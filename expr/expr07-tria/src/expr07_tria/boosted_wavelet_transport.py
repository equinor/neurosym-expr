from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from expr07_tria.batched_diagonal_spline_transport import (
    _DEGREE,
    BatchedDiagonalSplineTransport,
    _bounded_basis,
    _polynomial_coefficients,
    _polynomial_evaluate,
    _smoothing_matrix,
)
from expr07_tria.wavelet_features import (
    HaarFeatureSet,
    make_haar_features,
    project_haar,
)


@dataclass(frozen=True)
class _CandidateSplineData:
    """Residual-independent spline quantities for a candidate feature set."""

    design: Tensor
    knots: Tensor
    system: Tensor
    smoothing: Tensor
    left_design: Tensor
    right_design: Tensor
    left_derivative_design: Tensor
    right_derivative_design: Tensor


class BoostedWaveletSplineTransport(nn.Module):
    """Sparse triangular transport with boosted causal Haar parent features."""

    _learner_buffer_names = (
        "learner_outputs_",
        "learner_component_offsets_",
        "learner_starts_",
        "learner_widths_",
        "learner_levels_",
        "learner_knots_",
        "learner_coefficients_",
        "learner_polynomial_coefficients_",
        "learner_left_values_",
        "learner_right_values_",
        "learner_left_derivatives_",
        "learner_right_derivatives_",
        "training_nll_",
    )

    def __init__(
        self,
        *,
        max_wavelet_level: int | None = None,
        max_parent_distance: int | None = 256,
        max_learners_per_component: int = 4,
        candidate_count: int = 8,
        learning_rate: float = 0.25,
        log_smoothing: float = 2.0,
        min_loss_decrease: float = 1e-6,
        max_fit_iterations: int = 10,
        block_size: int = 1,
        fit_feature_batch_size: int = 4096,
    ) -> None:
        super().__init__()
        if max_wavelet_level is not None and max_wavelet_level < 0:
            raise ValueError("max_wavelet_level cannot be negative")
        if max_parent_distance is not None and max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
        if max_learners_per_component < 0:
            raise ValueError("max_learners_per_component cannot be negative")
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if fit_feature_batch_size < 1:
            raise ValueError("fit_feature_batch_size must be positive")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if min_loss_decrease < 0.0:
            raise ValueError("min_loss_decrease cannot be negative")

        self.max_wavelet_level = max_wavelet_level
        self.max_parent_distance = max_parent_distance
        self.max_learners_per_component = max_learners_per_component
        self.candidate_count = candidate_count
        self.learning_rate = learning_rate
        self.log_smoothing = log_smoothing
        self.min_loss_decrease = min_loss_decrease
        self.max_fit_iterations = max_fit_iterations
        self.block_size = block_size
        self.fit_feature_batch_size = fit_feature_batch_size
        self.diagonal = BatchedDiagonalSplineTransport(
            max_fit_iterations=max_fit_iterations,
        )
        for name in self._learner_buffer_names:
            self.register_buffer(name, None)
        self.n_samples_seen_: int | None = None
        self.n_features_in_: int | None = None
        self._learner_component_ranges: tuple[tuple[int, int], ...] = ()

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "n_samples_seen": self.n_samples_seen_,
            "n_features_in": self.n_features_in_,
            "max_wavelet_level": self.max_wavelet_level,
            "max_parent_distance": self.max_parent_distance,
            "max_learners_per_component": self.max_learners_per_component,
            "candidate_count": self.candidate_count,
            "learning_rate": self.learning_rate,
            "log_smoothing": self.log_smoothing,
            "min_loss_decrease": self.min_loss_decrease,
            "max_fit_iterations": self.max_fit_iterations,
            "block_size": self.block_size,
            "fit_feature_batch_size": self.fit_feature_batch_size,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.n_samples_seen_ = state["n_samples_seen"]
        self.n_features_in_ = state["n_features_in"]
        self.max_wavelet_level = state["max_wavelet_level"]
        self.max_parent_distance = state["max_parent_distance"]
        self.max_learners_per_component = state["max_learners_per_component"]
        self.candidate_count = state["candidate_count"]
        self.learning_rate = state["learning_rate"]
        self.log_smoothing = state["log_smoothing"]
        self.min_loss_decrease = state["min_loss_decrease"]
        self.max_fit_iterations = state["max_fit_iterations"]
        self.block_size = state.get("block_size", 1)
        self.fit_feature_batch_size = state.get("fit_feature_batch_size", 4096)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        offsets_key = f"{prefix}learner_component_offsets_"
        outputs_key = f"{prefix}learner_outputs_"
        mean_key = f"{prefix}diagonal.mean_"
        if (
            offsets_key not in state_dict
            and outputs_key in state_dict
            and mean_key in state_dict
        ):
            state_dict = dict(state_dict)
            state_dict[offsets_key] = self._component_offsets(
                state_dict[outputs_key],
                state_dict[mean_key].numel(),
            )
        for name in self._learner_buffer_names:
            key = f"{prefix}{name}"
            if getattr(self, name) is None and key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.learner_component_offsets_ is not None:
            self._rebuild_learner_component_cache()

    @staticmethod
    def _component_offsets(learner_outputs: Tensor, features: int) -> Tensor:
        counts = torch.bincount(learner_outputs, minlength=features)
        return torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))

    def _rebuild_learner_component_cache(self) -> None:
        assert self.learner_component_offsets_ is not None
        offsets = self.learner_component_offsets_.detach().cpu().tolist()
        self._learner_component_ranges = tuple(
            zip(offsets[:-1], offsets[1:], strict=True)
        )

    @property
    def dtype(self) -> torch.dtype:
        self._require_fitted()
        return self.diagonal.dtype

    @property
    def device(self) -> torch.device:
        self._require_fitted()
        return self.diagonal.device

    @property
    def learner_count_(self) -> int:
        self._require_fitted()
        assert self.learner_outputs_ is not None
        return self.learner_outputs_.numel()

    def _require_fitted(self) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")

    def _validate_input(self, value: Tensor, features: int, name: str) -> None:
        self._require_fitted()
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[1] != features:
            raise ValueError(f"{name} must have shape (N, {features})")
        if value.device != self.device:
            raise ValueError(
                f"{name} is on {value.device}, but the transport is on {self.device}"
            )
        if value.dtype != self.dtype:
            raise ValueError(
                f"{name} has dtype {value.dtype}, but the transport uses {self.dtype}"
            )
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")

    def _admissible_features(
        self,
        features: HaarFeatureSet,
        dependency_boundary: int,
    ) -> Tensor:
        admissible = features.ends <= dependency_boundary
        if self.max_parent_distance is not None:
            admissible = admissible & (
                features.ends
                > dependency_boundary - self.max_parent_distance
            )
        return torch.nonzero(admissible, as_tuple=False).flatten()

    @staticmethod
    def _screening_data(projections: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build normalized screening signals independent of the residual."""
        centered = projections - projections.mean(dim=0, keepdim=True)
        square = centered.square()
        absolute = centered.abs()
        signals = torch.stack(
            (
                centered,
                square - square.mean(dim=0, keepdim=True),
                absolute - absolute.mean(dim=0, keepdim=True),
            ),
            dim=2,
        )
        norms = torch.linalg.vector_norm(signals, dim=0).clamp_min(
            torch.finfo(projections.dtype).eps
        )
        finite_variation = torch.linalg.vector_norm(centered, dim=0) > (
            torch.finfo(projections.dtype).eps**0.5
        )
        return signals, norms, finite_variation

    def _prepare_candidate_splines(
        self,
        projections: Tensor,
    ) -> _CandidateSplineData:
        sample_count, candidate_count = projections.shape
        inner_knot_count = max(1, math.ceil(sample_count ** (1.0 / 3.0)))
        quantiles = torch.quantile(
            projections,
            projections.new_tensor((0.1, 0.9)),
            dim=0,
        )
        first, last = quantiles.unbind(dim=0)
        last = torch.maximum(
            last,
            first
            + torch.finfo(projections.dtype).eps**0.5
            * torch.maximum(torch.ones_like(first), first.abs()),
        )
        positions = torch.linspace(
            0.0,
            1.0,
            inner_knot_count + 2,
            dtype=projections.dtype,
            device=projections.device,
        ).unsqueeze(1)
        real_knots = first.unsqueeze(0) + (last - first).unsqueeze(0) * positions
        knots = torch.cat(
            (
                real_knots[:1].repeat(_DEGREE, 1),
                real_knots,
                real_knots[-1:].repeat(_DEGREE, 1),
            ),
            dim=0,
        ).T.contiguous()
        bounded = torch.maximum(
            torch.minimum(projections, last.unsqueeze(0)),
            first.unsqueeze(0),
        )
        design, _ = _bounded_basis(bounded, knots)
        left_design, left_derivative_design = _bounded_basis(
            first.unsqueeze(0),
            knots,
        )
        right_design, right_derivative_design = _bounded_basis(
            last.unsqueeze(0),
            knots,
        )
        design = torch.where(
            (projections < first.unsqueeze(0)).unsqueeze(2),
            left_design
            + (projections - first.unsqueeze(0)).unsqueeze(2)
            * left_derivative_design,
            design,
        )
        design = torch.where(
            (projections > last.unsqueeze(0)).unsqueeze(2),
            right_design
            + (projections - last.unsqueeze(0)).unsqueeze(2)
            * right_derivative_design,
            design,
        )
        by_candidate = design.permute(1, 0, 2).contiguous()
        basis_count = design.shape[2]
        gram = torch.bmm(by_candidate.transpose(1, 2), by_candidate)
        penalty = _smoothing_matrix(basis_count, design)
        scales = torch.sqrt(by_candidate.square().sum(dim=(1, 2)) / sample_count)
        smoothing = (
            math.exp(self.log_smoothing) * scales.square()
        )[:, None, None] * penalty.unsqueeze(0)
        system = gram + smoothing
        ridge = 1e-8 * torch.maximum(
            torch.ones(candidate_count, device=projections.device),
            torch.diagonal(system, dim1=1, dim2=2).abs().mean(dim=1),
        )
        identity = torch.eye(
            basis_count,
            dtype=projections.dtype,
            device=projections.device,
        )
        return _CandidateSplineData(
            design=by_candidate,
            knots=knots,
            system=system + ridge[:, None, None] * identity,
            smoothing=smoothing,
            left_design=left_design[0],
            right_design=right_design[0],
            left_derivative_design=left_derivative_design[0],
            right_derivative_design=right_derivative_design[0],
        )

    @staticmethod
    def _index_candidate_spline_data(
        candidates: _CandidateSplineData,
        indices: Tensor,
    ) -> _CandidateSplineData:
        return _CandidateSplineData(
            design=candidates.design[indices],
            knots=candidates.knots[indices],
            system=candidates.system[indices],
            smoothing=candidates.smoothing[indices],
            left_design=candidates.left_design[indices],
            right_design=candidates.right_design[indices],
            left_derivative_design=candidates.left_derivative_design[indices],
            right_derivative_design=candidates.right_derivative_design[indices],
        )

    @staticmethod
    def _combine_candidate_spline_data(
        candidates: list[_CandidateSplineData],
    ) -> _CandidateSplineData:
        return _CandidateSplineData(
            design=torch.cat([candidate.design for candidate in candidates]),
            knots=torch.cat([candidate.knots for candidate in candidates]),
            system=torch.cat([candidate.system for candidate in candidates]),
            smoothing=torch.cat(
                [candidate.smoothing for candidate in candidates]
            ),
            left_design=torch.cat(
                [candidate.left_design for candidate in candidates]
            ),
            right_design=torch.cat(
                [candidate.right_design for candidate in candidates]
            ),
            left_derivative_design=torch.cat(
                [candidate.left_derivative_design for candidate in candidates]
            ),
            right_derivative_design=torch.cat(
                [candidate.right_derivative_design for candidate in candidates]
            ),
        )

    def _fit_candidate_splines(
        self,
        candidate_data: _CandidateSplineData,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        by_candidate = candidate_data.design
        candidate_count = by_candidate.shape[0]
        if residual.ndim == 1:
            candidate_residuals = residual.expand(candidate_count, -1)
        elif residual.shape == by_candidate.shape[:2]:
            candidate_residuals = residual
        else:
            raise ValueError("residual must match the candidate spline batch")
        right_hand_side = -torch.bmm(
            by_candidate.transpose(1, 2),
            candidate_residuals.unsqueeze(2),
        )
        coefficients = torch.linalg.solve(
            candidate_data.system,
            right_hand_side,
        ).squeeze(2)
        coefficients = self.learning_rate * coefficients
        predictions = torch.bmm(
            by_candidate,
            coefficients.unsqueeze(2),
        ).squeeze(2)
        loss_decreases = 0.5 * (
            candidate_residuals.square().sum(dim=1)
            - (candidate_residuals + predictions).square().sum(dim=1)
        )
        loss_decreases = loss_decreases - 0.5 * torch.sum(
            coefficients
            * torch.bmm(
                candidate_data.smoothing,
                coefficients.unsqueeze(2),
            ).squeeze(2),
            dim=1,
        )
        left_values = torch.sum(candidate_data.left_design * coefficients, dim=1)
        right_values = torch.sum(candidate_data.right_design * coefficients, dim=1)
        left_derivatives = torch.sum(
            candidate_data.left_derivative_design * coefficients,
            dim=1,
        )
        right_derivatives = torch.sum(
            candidate_data.right_derivative_design * coefficients,
            dim=1,
        )
        return (
            loss_decreases,
            predictions,
            candidate_data.knots,
            coefficients,
            left_values,
            right_values,
            left_derivatives,
            right_derivatives,
        )

    def fit(self, X: Tensor) -> BoostedWaveletSplineTransport:
        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor")
        if not X.is_floating_point():
            raise TypeError("X must have a floating-point dtype")
        if X.ndim != 2:
            raise ValueError("X must have shape (N, D)")
        if not torch.all(torch.isfinite(X)):
            raise ValueError("X must contain only finite values")

        samples = X.detach()
        sample_count, dimension_count = samples.shape
        self.diagonal.fit(samples)
        assert self.diagonal.mean_ is not None and self.diagonal.scale_ is not None
        standardized = (
            samples - self.diagonal.mean_.unsqueeze(0)
        ) / self.diagonal.scale_.unsqueeze(0)
        outputs = self.diagonal(samples).detach()
        feature_set = make_haar_features(
            dimension_count,
            self.max_wavelet_level,
        ).to(samples.device)
        all_projections = project_haar(
            standardized,
            feature_set.starts,
            feature_set.widths,
        )
        if self.max_learners_per_component > 0:
            all_signals, all_norms, all_finite_variation = (
                self._screening_data(all_projections)
            )
            spline_batches = [
                self._prepare_candidate_splines(
                    all_projections[
                        :, start : start + self.fit_feature_batch_size
                    ]
                )
                for start in range(
                    0,
                    all_projections.shape[1],
                    self.fit_feature_batch_size,
                )
            ]
            all_splines = self._combine_candidate_spline_data(spline_batches)
        else:
            all_signals = None
            all_norms = None
            all_finite_variation = None
            all_splines = None

        learner_outputs: list[Tensor] = []
        learner_starts: list[Tensor] = []
        learner_widths: list[Tensor] = []
        learner_levels: list[Tensor] = []
        learner_knots: list[Tensor] = []
        learner_coefficients: list[Tensor] = []
        learner_left_values: list[Tensor] = []
        learner_right_values: list[Tensor] = []
        learner_left_derivatives: list[Tensor] = []
        learner_right_derivatives: list[Tensor] = []

        for block_start in range(0, dimension_count, self.block_size):
            if self.max_learners_per_component == 0:
                break
            assert (
                all_signals is not None
                and all_norms is not None
                and all_finite_variation is not None
                and all_splines is not None
            )
            block_end = min(block_start + self.block_size, dimension_count)
            admissible = self._admissible_features(feature_set, block_start)
            if admissible.numel() == 0:
                continue
            signals = all_signals[:, admissible]
            norms = all_norms[admissible]
            finite_variation = all_finite_variation[admissible]
            residual = outputs[:, block_start:block_end]
            component_count = block_end - block_start
            component_ids = torch.arange(
                block_start,
                block_end,
                dtype=torch.long,
                device=samples.device,
            )
            selected_count = min(self.candidate_count, admissible.numel())
            for _ in range(self.max_learners_per_component):
                scores = torch.max(
                    torch.abs(
                        torch.einsum("nct,nb->bct", signals, residual)
                    )
                    / norms.unsqueeze(0),
                    dim=2,
                ).values
                scores = torch.where(
                    finite_variation.unsqueeze(0),
                    scores,
                    torch.full_like(scores, -torch.inf),
                )
                screened = torch.topk(
                    scores,
                    selected_count,
                    dim=1,
                )
                candidates = admissible[screened.indices]
                flat_candidates = candidates.flatten()
                candidate_residuals = (
                    residual.T.repeat_interleave(selected_count, dim=0)
                )
                (
                    decreases,
                    predictions,
                    knots,
                    coefficients,
                    left_values,
                    right_values,
                    left_derivatives,
                    right_derivatives,
                ) = self._fit_candidate_splines(
                    self._index_candidate_spline_data(
                        all_splines,
                        flat_candidates,
                    ),
                    candidate_residuals,
                )
                decreases = decreases.reshape(component_count, selected_count)
                predictions = predictions.reshape(
                    component_count,
                    selected_count,
                    sample_count,
                )
                best = torch.argmax(decreases, dim=1)
                rows = torch.arange(component_count, device=samples.device)
                best_decreases = decreases[rows, best]
                accepted = (
                    torch.isfinite(screened.values[rows, best])
                    & torch.isfinite(best_decreases)
                    & (
                        best_decreases / sample_count
                        > self.min_loss_decrease
                    )
                )
                best_flat = rows * selected_count + best
                best_predictions = predictions[rows, best]
                residual = residual + (
                    best_predictions * accepted.unsqueeze(1)
                ).T

                accepted_indices = best_flat[accepted]
                features = flat_candidates[accepted_indices]
                learner_outputs.append(component_ids[accepted])
                learner_starts.append(feature_set.starts[features])
                learner_widths.append(feature_set.widths[features])
                learner_levels.append(feature_set.levels[features])
                learner_knots.append(knots[accepted_indices])
                learner_coefficients.append(coefficients[accepted_indices])
                learner_left_values.append(left_values[accepted_indices])
                learner_right_values.append(right_values[accepted_indices])
                learner_left_derivatives.append(
                    left_derivatives[accepted_indices]
                )
                learner_right_derivatives.append(
                    right_derivatives[accepted_indices]
                )
            outputs[:, block_start:block_end] = residual

        self.n_samples_seen_ = sample_count
        self.n_features_in_ = dimension_count
        learner_outputs_tensor = (
            torch.cat(learner_outputs)
            if learner_outputs
            else torch.empty(0, dtype=torch.long, device=samples.device)
        )
        if learner_outputs_tensor.numel() > 0:
            order = torch.argsort(learner_outputs_tensor, stable=True)
            self.learner_outputs_ = learner_outputs_tensor[order]
            self.learner_starts_ = torch.cat(learner_starts)[order]
            self.learner_widths_ = torch.cat(learner_widths)[order]
            self.learner_levels_ = torch.cat(learner_levels)[order]
            self.learner_knots_ = torch.cat(learner_knots)[order]
            self.learner_coefficients_ = torch.cat(learner_coefficients)[order]
            self.learner_polynomial_coefficients_ = _polynomial_coefficients(
                self.learner_knots_,
                self.learner_coefficients_,
            )
            self.learner_left_values_ = torch.cat(learner_left_values)[order]
            self.learner_right_values_ = torch.cat(learner_right_values)[order]
            self.learner_left_derivatives_ = torch.cat(
                learner_left_derivatives
            )[order]
            self.learner_right_derivatives_ = torch.cat(
                learner_right_derivatives
            )[order]
        else:
            inner_knot_count = max(1, math.ceil(sample_count ** (1.0 / 3.0)))
            basis_count = inner_knot_count + _DEGREE + 1
            interval_count = inner_knot_count + 1
            self.learner_outputs_ = torch.empty(
                0,
                dtype=torch.long,
                device=samples.device,
            )
            self.learner_starts_ = torch.empty_like(self.learner_outputs_)
            self.learner_widths_ = torch.empty_like(self.learner_outputs_)
            self.learner_levels_ = torch.empty_like(self.learner_outputs_)
            self.learner_knots_ = samples.new_empty(
                (0, basis_count + _DEGREE + 1)
            )
            self.learner_coefficients_ = samples.new_empty((0, basis_count))
            self.learner_polynomial_coefficients_ = samples.new_empty(
                (0, interval_count, 4)
            )
            self.learner_left_values_ = samples.new_empty(0)
            self.learner_right_values_ = samples.new_empty(0)
            self.learner_left_derivatives_ = samples.new_empty(0)
            self.learner_right_derivatives_ = samples.new_empty(0)

        assert self.learner_outputs_ is not None
        self.learner_component_offsets_ = self._component_offsets(
            self.learner_outputs_,
            dimension_count,
        )
        self._rebuild_learner_component_cache()

        _, diagonal_derivatives = self.diagonal._evaluate(standardized)
        assert diagonal_derivatives is not None
        self.training_nll_ = (
            0.5 * outputs.square().sum(dim=0)
            - torch.log(diagonal_derivatives).sum(dim=0)
        ).detach()
        return self

    def _evaluate_learner_values(
        self,
        projections: Tensor,
        learner_indices: Tensor | slice,
    ) -> Tensor:
        assert (
            self.learner_knots_ is not None
            and self.learner_polynomial_coefficients_ is not None
            and self.learner_left_values_ is not None
            and self.learner_right_values_ is not None
            and self.learner_left_derivatives_ is not None
            and self.learner_right_derivatives_ is not None
        )
        if projections.shape[1] == 0:
            return projections.new_empty((projections.shape[0], 0))
        knots = self.learner_knots_[learner_indices]
        polynomials = self.learner_polynomial_coefficients_[learner_indices]
        left = knots[:, _DEGREE]
        right = knots[:, -_DEGREE - 1]
        bounded = torch.maximum(
            torch.minimum(projections, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        offsets = (
            torch.arange(
                projections.shape[1],
                dtype=torch.long,
                device=projections.device,
            )
            * polynomials.shape[1]
        ).unsqueeze(0)
        values, _ = _polynomial_evaluate(
            bounded,
            knots,
            polynomials,
            offsets,
            with_derivatives=False,
        )
        below = projections < left.unsqueeze(0)
        above = projections > right.unsqueeze(0)
        values = torch.where(
            below,
            self.learner_left_values_[learner_indices].unsqueeze(0)
            + (projections - left.unsqueeze(0))
            * self.learner_left_derivatives_[learner_indices].unsqueeze(0),
            values,
        )
        return torch.where(
            above,
            self.learner_right_values_[learner_indices].unsqueeze(0)
            + (projections - right.unsqueeze(0))
            * self.learner_right_derivatives_[learner_indices].unsqueeze(0),
            values,
        )

    def _parent_offsets(self, standardized: Tensor) -> Tensor:
        assert (
            self.n_features_in_ is not None
            and self.learner_outputs_ is not None
            and self.learner_starts_ is not None
            and self.learner_widths_ is not None
        )
        if self.learner_outputs_.numel() == 0:
            return standardized.new_zeros(
                (standardized.shape[0], self.n_features_in_)
            )
        indices = torch.arange(
            self.learner_outputs_.numel(),
            dtype=torch.long,
            device=standardized.device,
        )
        projections = project_haar(
            standardized,
            self.learner_starts_,
            self.learner_widths_,
        )
        values = self._evaluate_learner_values(projections, indices)
        offsets = standardized.new_zeros(
            (standardized.shape[0], self.n_features_in_)
        )
        return offsets.index_add(1, self.learner_outputs_, values)

    def forward(self, X: Tensor) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        assert self.diagonal.mean_ is not None and self.diagonal.scale_ is not None
        self._validate_input(X, self.n_features_in_, "X")
        standardized = (X - self.diagonal.mean_) / self.diagonal.scale_
        return self.diagonal(X) + self._parent_offsets(standardized)

    def log_abs_det_jacobian(self, X: Tensor) -> Tensor:
        self._require_fitted()
        assert (
            self.n_features_in_ is not None
            and self.diagonal.mean_ is not None
            and self.diagonal.scale_ is not None
        )
        self._validate_input(X, self.n_features_in_, "X")
        standardized = (X - self.diagonal.mean_) / self.diagonal.scale_
        _, derivatives = self.diagonal._evaluate(standardized)
        assert derivatives is not None
        if torch.any(derivatives <= 100.0 * torch.finfo(X.dtype).eps):
            raise RuntimeError("the fitted map is not strictly monotone")
        return torch.log(derivatives / self.diagonal.scale_).sum(dim=1)

    def _diagonal_column_evaluate(
        self,
        x: Tensor,
        component: int,
    ) -> tuple[Tensor, Tensor]:
        assert (
            self.diagonal.knots_ is not None
            and self.diagonal.polynomial_coefficients_ is not None
            and self.diagonal.left_values_ is not None
            and self.diagonal.right_values_ is not None
            and self.diagonal.left_derivatives_ is not None
            and self.diagonal.right_derivatives_ is not None
        )
        knots = self.diagonal.knots_[component : component + 1]
        polynomials = self.diagonal.polynomial_coefficients_[
            component : component + 1
        ]
        left = knots[0, _DEGREE]
        right = knots[0, -_DEGREE - 1]
        bounded = x.clamp(min=left, max=right).unsqueeze(1)
        values, derivatives = _polynomial_evaluate(
            bounded,
            knots,
            polynomials,
            torch.zeros((1, 1), dtype=torch.long, device=x.device),
        )
        assert derivatives is not None
        values = values[:, 0]
        derivatives = derivatives[:, 0]
        below = x < left
        above = x > right
        values = torch.where(
            below,
            self.diagonal.left_values_[component]
            + (x - left) * self.diagonal.left_derivatives_[component],
            values,
        )
        values = torch.where(
            above,
            self.diagonal.right_values_[component]
            + (x - right) * self.diagonal.right_derivatives_[component],
            values,
        )
        derivatives = torch.where(
            below,
            self.diagonal.left_derivatives_[component],
            derivatives,
        )
        derivatives = torch.where(
            above,
            self.diagonal.right_derivatives_[component],
            derivatives,
        )
        return values, derivatives

    def _diagonal_block_evaluate(
        self,
        x: Tensor,
        component_start: int,
        component_end: int,
    ) -> tuple[Tensor, Tensor]:
        assert (
            self.diagonal.knots_ is not None
            and self.diagonal.polynomial_coefficients_ is not None
            and self.diagonal.left_values_ is not None
            and self.diagonal.right_values_ is not None
            and self.diagonal.left_derivatives_ is not None
            and self.diagonal.right_derivatives_ is not None
        )
        knots = self.diagonal.knots_[component_start:component_end]
        polynomials = self.diagonal.polynomial_coefficients_[
            component_start:component_end
        ]
        left = knots[:, _DEGREE]
        right = knots[:, -_DEGREE - 1]
        bounded = torch.maximum(
            torch.minimum(x, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        interval_offsets = (
            torch.arange(
                component_end - component_start,
                dtype=torch.long,
                device=x.device,
            )
            * polynomials.shape[1]
        ).unsqueeze(0)
        values, derivatives = _polynomial_evaluate(
            bounded,
            knots,
            polynomials,
            interval_offsets,
        )
        assert derivatives is not None
        below = x < left.unsqueeze(0)
        above = x > right.unsqueeze(0)
        left_values = self.diagonal.left_values_[
            component_start:component_end
        ].unsqueeze(0)
        right_values = self.diagonal.right_values_[
            component_start:component_end
        ].unsqueeze(0)
        left_derivatives = self.diagonal.left_derivatives_[
            component_start:component_end
        ].unsqueeze(0)
        right_derivatives = self.diagonal.right_derivatives_[
            component_start:component_end
        ].unsqueeze(0)
        values = torch.where(
            below,
            left_values + (x - left.unsqueeze(0)) * left_derivatives,
            values,
        )
        values = torch.where(
            above,
            right_values + (x - right.unsqueeze(0)) * right_derivatives,
            values,
        )
        derivatives = torch.where(below, left_derivatives, derivatives)
        derivatives = torch.where(above, right_derivatives, derivatives)
        return values, derivatives

    def _invert_diagonal_block(
        self,
        target: Tensor,
        component_start: int,
        component_end: int,
    ) -> Tensor:
        assert (
            self.diagonal.knots_ is not None
            and self.diagonal.left_values_ is not None
            and self.diagonal.right_values_ is not None
            and self.diagonal.left_derivatives_ is not None
            and self.diagonal.right_derivatives_ is not None
        )
        knots = self.diagonal.knots_[component_start:component_end]
        left = knots[:, _DEGREE].unsqueeze(0)
        right = knots[:, -_DEGREE - 1].unsqueeze(0)
        left_value = self.diagonal.left_values_[
            component_start:component_end
        ].unsqueeze(0)
        right_value = self.diagonal.right_values_[
            component_start:component_end
        ].unsqueeze(0)
        left_slope = self.diagonal.left_derivatives_[
            component_start:component_end
        ].unsqueeze(0)
        right_slope = self.diagonal.right_derivatives_[
            component_start:component_end
        ].unsqueeze(0)
        epsilon = 100.0 * torch.finfo(target.dtype).eps
        if torch.any((left_slope <= epsilon) | (right_slope <= epsilon)):
            raise RuntimeError("the fitted map has a non-positive tail slope")

        below = target < left_value
        above = target > right_value
        low = left.expand_as(target)
        high = right.expand_as(target)
        middle = left + (right - left) * (
            (target - left_value) / (right_value - left_value)
        )
        middle = torch.maximum(torch.minimum(middle, right), left)
        iterations = 24 if target.dtype == torch.float32 else 40
        for _ in range(iterations):
            values, slopes = self._diagonal_block_evaluate(
                middle,
                component_start,
                component_end,
            )
            low = torch.where(values < target, middle, low)
            high = torch.where(values >= target, middle, high)
            newton = middle - (values - target) / slopes
            valid = (
                torch.isfinite(newton)
                & (slopes > epsilon)
                & (newton >= low)
                & (newton <= high)
            )
            middle = torch.where(valid, newton, 0.5 * (low + high))

        left_tail = left + (target - left_value) / left_slope
        right_tail = right + (target - right_value) / right_slope
        result = torch.where(below, left_tail, middle)
        result = torch.where(above, right_tail, result)
        root = result.detach()
        root_value, root_slope = self._diagonal_block_evaluate(
            root,
            component_start,
            component_end,
        )
        return root + (target - root_value) / root_slope.detach()

    def _block_parent_offsets(
        self,
        history: Tensor,
        history_start: int,
        component_start: int,
        component_end: int,
    ) -> Tensor:
        assert (
            self.learner_starts_ is not None
            and self.learner_widths_ is not None
            and self.learner_outputs_ is not None
        )
        output_count = component_end - component_start
        learner_start = self._learner_component_ranges[component_start][0]
        learner_end = self._learner_component_ranges[component_end - 1][1]
        if learner_start == learner_end:
            return history.new_zeros((history.shape[0], output_count))
        learner_indices = slice(learner_start, learner_end)
        projections = project_haar(
            history,
            self.learner_starts_[learner_indices] - history_start,
            self.learner_widths_[learner_indices],
        )
        values = self._evaluate_learner_values(projections, learner_indices)
        offsets = history.new_zeros((history.shape[0], output_count))
        return offsets.index_add(
            1,
            self.learner_outputs_[learner_indices] - component_start,
            values,
        )

    def _inverse_from_prefix(self, X_star: Tensor, Z: Tensor) -> Tensor:
        assert (
            self.n_features_in_ is not None
            and self.diagonal.mean_ is not None
            and self.diagonal.scale_ is not None
        )
        prefix_count = X_star.shape[1]
        standardized_prefix = (
            X_star - self.diagonal.mean_[:prefix_count]
        ) / self.diagonal.scale_[:prefix_count]
        result_blocks = [standardized_prefix]
        history = standardized_prefix
        component_start = prefix_count
        dependency_boundary = (
            component_start // self.block_size
        ) * self.block_size
        history_start = 0
        if self.max_parent_distance is not None:
            history_start = max(
                0,
                dependency_boundary - self.max_parent_distance,
            )
            history = history[:, history_start:]

        while component_start < self.n_features_in_:
            dependency_boundary = (
                component_start // self.block_size
            ) * self.block_size
            component_end = min(
                dependency_boundary + self.block_size,
                self.n_features_in_,
            )
            offsets = self._block_parent_offsets(
                history,
                history_start,
                component_start,
                component_end,
            )
            block_target = Z[
                :,
                component_start - prefix_count : component_end - prefix_count,
            ]
            block = self._invert_diagonal_block(
                block_target - offsets,
                component_start,
                component_end,
            )
            result_blocks.append(block)
            history = torch.cat((history, block), dim=1)
            component_start = component_end
            if self.max_parent_distance is not None:
                retained_start = max(
                    0,
                    component_end - self.max_parent_distance,
                )
                drop = retained_start - history_start
                if drop > 0:
                    history = history[:, drop:]
                    history_start = retained_start

        standardized = torch.cat(result_blocks, dim=1)
        return (
            standardized * self.diagonal.scale_
            + self.diagonal.mean_
        )

    def inverse(self, Z: Tensor) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        self._validate_input(Z, self.n_features_in_, "Z")
        empty = Z.new_empty((Z.shape[0], 0))
        return self._inverse_from_prefix(empty, Z)

    def conditional_inverse(self, X_star: Tensor, Z: Tensor) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        if not isinstance(X_star, Tensor) or X_star.ndim != 2:
            raise ValueError("X_star must be a two-dimensional torch.Tensor")
        prefix_count = X_star.shape[1]
        if prefix_count >= self.n_features_in_:
            raise ValueError("X_star must condition fewer than all features")
        self._validate_input(X_star, prefix_count, "X_star")
        self._validate_input(Z, self.n_features_in_ - prefix_count, "Z")
        if X_star.shape[0] != Z.shape[0]:
            raise ValueError("X_star and Z must contain the same number of samples")
        return self._inverse_from_prefix(X_star, Z)

    def sample_exceedance(
        self,
        *,
        component: int,
        threshold: float | Tensor,
        sample_count: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._require_fitted()
        assert (
            self.n_features_in_ is not None
            and self.diagonal.mean_ is not None
            and self.diagonal.scale_ is not None
        )
        if component != 0:
            raise NotImplementedError(
                "exact exceedance sampling currently requires component=0"
            )
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        threshold_tensor = torch.as_tensor(
            threshold,
            dtype=self.dtype,
            device=self.device,
        )
        if threshold_tensor.numel() != 1 or not torch.isfinite(threshold_tensor):
            raise ValueError("threshold must be one finite scalar")
        standardized_threshold = (
            threshold_tensor - self.diagonal.mean_[0]
        ) / self.diagonal.scale_[0]
        boundary, _ = self._diagonal_column_evaluate(
            standardized_threshold.reshape(1),
            0,
        )
        lower_probability = 0.5 * (
            1.0 + torch.erf(boundary[0] / math.sqrt(2.0))
        )
        one = torch.ones((), dtype=self.dtype, device=self.device)
        upper_probability = torch.nextafter(one, torch.zeros_like(one))
        if lower_probability >= upper_probability:
            raise RuntimeError(
                "the exceedance probability is below numerical precision"
            )
        uniform = torch.rand(
            sample_count,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        probability = lower_probability + (
            upper_probability - lower_probability
        ) * uniform
        first_reference = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
        reference = torch.randn(
            (sample_count, self.n_features_in_),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        reference[:, 0] = first_reference
        return self.inverse(reference)
