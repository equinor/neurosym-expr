from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from nine.monotone_pspline import BatchedDiagonalSplineTransport
from nine.soft_tree_stage import SoftTreeTransportStage


class BoostedSoftTreeTransport(nn.Module):
    """Stage-wise triangular transport with soft-tree neural splines."""

    def __init__(
        self,
        *,
        max_stages: int = 4,
        max_depth: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        max_wavelet_level: int | None = None,
        learning_rate: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 30,
        min_temperature: float = 0.05,
        validation_fraction: float = 0.2,
        max_parent_distance: int | None = None,
        max_fit_iterations: int = 10,
        minimum_improvement_per_sample: float = 0.1,
        fine_tune_epochs: int = 25,
        fine_tune_learning_rate: float = 2.5e-3,
    ) -> None:
        super().__init__()
        if max_stages < 0:
            raise ValueError("max_stages cannot be negative")
        if max_parent_distance is not None and max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
        if minimum_improvement_per_sample < 0:
            raise ValueError("minimum_improvement_per_sample cannot be negative")
        if fine_tune_epochs < 0:
            raise ValueError("fine_tune_epochs cannot be negative")
        if fine_tune_learning_rate <= 0:
            raise ValueError("fine_tune_learning_rate must be positive")
        self.max_stages = max_stages
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.max_wavelet_level = max_wavelet_level
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_temperature = min_temperature
        self.validation_fraction = validation_fraction
        self.max_parent_distance = max_parent_distance
        self.max_fit_iterations = max_fit_iterations
        self.minimum_improvement_per_sample = minimum_improvement_per_sample
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.diagonal = BatchedDiagonalSplineTransport(
            max_fit_iterations=max_fit_iterations
        )
        self.stages = nn.ModuleList()
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int | None = None
        self.training_nll_: float | None = None
        self.criterion_improvements_: list[float] = []
        self.validation_improvements_: list[float] = []
        self.fine_tune_improvement_: float = 0.0
        self.stopping_reason_: str | None = None
        self.register_buffer("_placement_", torch.empty(0), persistent=False)
        self._placement_requested = False

    def _apply(self, fn: Any, recurse: bool = True) -> BoostedSoftTreeTransport:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    def _new_stage(self) -> SoftTreeTransportStage:
        return SoftTreeTransportStage(
            max_depth=self.max_depth,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            max_wavelet_level=self.max_wavelet_level,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_temperature=self.min_temperature,
            validation_fraction=self.validation_fraction,
            max_parent_distance=self.max_parent_distance,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "max_stages": self.max_stages,
            "max_depth": self.max_depth,
            "num_bins": self.num_bins,
            "tail_bound": self.tail_bound,
            "max_wavelet_level": self.max_wavelet_level,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_temperature": self.min_temperature,
            "validation_fraction": self.validation_fraction,
            "max_parent_distance": self.max_parent_distance,
            "max_fit_iterations": self.max_fit_iterations,
            "minimum_improvement_per_sample": self.minimum_improvement_per_sample,
            "fine_tune_epochs": self.fine_tune_epochs,
            "fine_tune_learning_rate": self.fine_tune_learning_rate,
            "n_features_in": self.n_features_in_,
            "n_samples_seen": self.n_samples_seen_,
            "training_nll": self.training_nll_,
            "criterion_improvements": self.criterion_improvements_,
            "validation_improvements": self.validation_improvements_,
            "fine_tune_improvement": self.fine_tune_improvement_,
            "stopping_reason": self.stopping_reason_,
            "stage_count": len(self.stages),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.max_stages = state["max_stages"]
        self.max_depth = state["max_depth"]
        self.num_bins = state["num_bins"]
        self.tail_bound = state["tail_bound"]
        self.max_wavelet_level = state["max_wavelet_level"]
        self.learning_rate = state["learning_rate"]
        self.max_epochs = state["max_epochs"]
        self.patience = state["patience"]
        self.min_temperature = state["min_temperature"]
        self.validation_fraction = state["validation_fraction"]
        self.max_parent_distance = state["max_parent_distance"]
        self.max_fit_iterations = state["max_fit_iterations"]
        self.minimum_improvement_per_sample = state.get(
            "minimum_improvement_per_sample",
            0.1,
        )
        self.fine_tune_epochs = state.get("fine_tune_epochs", 0)
        self.fine_tune_learning_rate = state.get(
            "fine_tune_learning_rate",
            2.5e-3,
        )
        self.n_features_in_ = state["n_features_in"]
        self.n_samples_seen_ = state["n_samples_seen"]
        self.training_nll_ = state["training_nll"]
        self.criterion_improvements_ = state["criterion_improvements"]
        self.validation_improvements_ = state.get("validation_improvements", [])
        self.fine_tune_improvement_ = state.get("fine_tune_improvement", 0.0)
        self.stopping_reason_ = state["stopping_reason"]
        self.diagonal.max_fit_iterations = self.max_fit_iterations

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        extra = state_dict.get(f"{prefix}_extra_state", {})
        saved_stage_count = extra.get("stage_count", 0)
        if extra:
            self.set_extra_state(extra)
        while len(self.stages) > saved_stage_count:
            self.stages.pop(-1)
        while len(self.stages) < saved_stage_count:
            stage = self._new_stage()
            if self._placement_requested:
                stage = stage.to(
                    device=self._placement_.device,
                    dtype=self._placement_.dtype,
                )
            self.stages.append(stage)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def stage_count_(self) -> int:
        return len(self.stages)

    @property
    def leaf_count_(self) -> int:
        return sum(stage.leaf_count for stage in self.stages)

    def fit(self, values: Tensor) -> BoostedSoftTreeTransport:
        self.diagonal.fit(values)
        self.n_samples_seen_, self.n_features_in_ = values.shape
        with torch.no_grad():
            current = self.diagonal(values)
        self.stages = nn.ModuleList()
        self.criterion_improvements_ = []
        self.validation_improvements_ = []
        self.fine_tune_improvement_ = 0.0
        current_nll = float(0.5 * current.square().sum().item())
        validation_count = (
            max(1, round(values.shape[0] * self.validation_fraction))
            if self.validation_fraction > 0 and values.shape[0] >= 10
            else 0
        )
        validation_mask = torch.zeros(
            values.shape[0],
            dtype=torch.bool,
            device=values.device,
        )
        if validation_count:
            validation_mask[
                torch.linspace(
                    0,
                    values.shape[0] - 1,
                    validation_count,
                    device=values.device,
                ).round().long()
            ] = True
        current_validation_nll = (
            float(
                (0.5 * current[validation_mask].square()).sum().item()
            )
            if validation_count
            else current_nll
        )

        for _ in range(self.max_stages):
            candidate = self._new_stage()
            mapped, logdet = candidate._fit_with_output(current)
            if candidate.is_identity:
                self.stopping_reason_ = "candidate_stage_is_identity"
                break
            candidate_nll = float(
                (0.5 * mapped.square().sum(dim=1) - logdet).sum().item()
            )
            improvement = current_nll - candidate_nll
            if validation_count:
                candidate_validation_nll = float(
                    (
                        0.5
                        * mapped[validation_mask].square().sum(dim=1)
                        - logdet[validation_mask]
                    ).sum().item()
                )
                validation_improvement = (
                    current_validation_nll - candidate_validation_nll
                )
            else:
                validation_improvement = improvement
            tolerance = 100.0 * torch.finfo(values.dtype).eps * (
                1.0 + abs(current_validation_nll)
            )
            minimum_training_improvement = (
                self.minimum_improvement_per_sample * values.shape[0]
            )
            if (
                validation_improvement <= tolerance
                or improvement <= minimum_training_improvement
            ):
                self.stopping_reason_ = "no_global_criterion_improvement"
                break
            self.stages.append(candidate)
            self.criterion_improvements_.append(improvement)
            self.validation_improvements_.append(validation_improvement)
            current = mapped.detach()
            current_nll = candidate_nll
            current_validation_nll = candidate_validation_nll
        else:
            self.stopping_reason_ = "search_budget_exhausted"

        self._fine_tune(values, validation_mask)
        with torch.no_grad():
            self.training_nll_ = float(-self.log_prob(values).sum().item())
        return self

    def _fine_tune(self, values: Tensor, validation_mask: Tensor) -> None:
        if self.fine_tune_epochs == 0 or not self.stages:
            return
        parameters = list(self.stages.parameters())
        if not parameters:
            return
        training_mask = ~validation_mask
        scoring_mask = validation_mask if torch.any(validation_mask) else training_mask

        def objective(mask: Tensor) -> Tensor:
            reference, logdet = self.forward_with_logdet(values[mask])
            loss = (
                0.5 * reference.square().sum(dim=1) - logdet
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("soft-tree fine-tuning produced a non-finite loss")
            return loss

        with torch.no_grad():
            initial_score = objective(scoring_mask)
        best_score = float(initial_score.item())
        best_parameters = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.fine_tune_learning_rate,
        )

        for _ in range(self.fine_tune_epochs):
            optimizer.zero_grad()
            objective(training_mask).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            with torch.no_grad():
                score = float(objective(scoring_mask).item())
            tolerance = 100.0 * torch.finfo(values.dtype).eps * (
                1.0 + abs(best_score)
            )
            if score < best_score - tolerance:
                best_score = score
                for best, parameter in zip(
                    best_parameters,
                    parameters,
                    strict=True,
                ):
                    best.copy_(parameter)

        with torch.no_grad():
            for parameter, best in zip(
                parameters,
                best_parameters,
                strict=True,
            ):
                parameter.copy_(best)
        self.fine_tune_improvement_ = float(initial_score.item()) - best_score

    def _validate(self, values: Tensor, name: str) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")
        if not isinstance(values, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have shape (N, {self.n_features_in_})")
        assert self.diagonal.mean_ is not None
        if values.device != self.diagonal.mean_.device:
            raise ValueError(f"{name} and the transport must be on the same device")
        if values.dtype != self.diagonal.mean_.dtype:
            raise ValueError(f"{name} and the transport must have the same dtype")

    def forward(self, values: Tensor) -> Tensor:
        self._validate(values, "values")
        assert self.diagonal.mean_ is not None and self.diagonal.scale_ is not None
        standardized = (values - self.diagonal.mean_) / self.diagonal.scale_
        result = self.diagonal._evaluate(
            standardized,
            with_derivatives=False,
        )[0]
        for stage in self.stages:
            result = stage._forward(result, compute_logdet=False)[0]
        return result

    def forward_with_logdet(self, values: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(values, "values")
        assert self.diagonal.mean_ is not None and self.diagonal.scale_ is not None
        standardized = (values - self.diagonal.mean_) / self.diagonal.scale_
        result, derivative = self.diagonal._evaluate(standardized)
        assert derivative is not None
        logdet = torch.log(derivative / self.diagonal.scale_).sum(dim=1)
        for stage in self.stages:
            result, stage_logdet = stage._forward(
                result,
                compute_logdet=True,
            )
            assert stage_logdet is not None
            logdet = logdet + stage_logdet
        return result, logdet

    def log_abs_det_jacobian(self, values: Tensor) -> Tensor:
        return self.forward_with_logdet(values)[1]

    def log_prob(self, values: Tensor) -> Tensor:
        reference, logdet = self.forward_with_logdet(values)
        return (
            -0.5 * (reference.square() + math.log(2.0 * math.pi)).sum(dim=1)
            + logdet
        )

    def inverse(self, reference: Tensor) -> Tensor:
        self._validate(reference, "reference")
        result = reference
        for stage in reversed(self.stages):
            result = stage._inverse_from_reference(result, 0)
        return self.diagonal._inverse_dimensions(result, start=0)

    def conditional_inverse(
        self,
        prefix: Tensor,
        reference_suffix: Tensor,
    ) -> Tensor:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")
        if prefix.ndim != 2 or reference_suffix.ndim != 2:
            raise ValueError("prefix and reference_suffix must be matrices")
        if (
            prefix.shape[0] != reference_suffix.shape[0]
            or prefix.shape[1] + reference_suffix.shape[1] != self.n_features_in_
        ):
            raise ValueError("prefix and reference_suffix have incompatible shapes")
        if prefix.device != reference_suffix.device:
            raise ValueError("prefix and reference_suffix must be on the same device")
        if prefix.dtype != reference_suffix.dtype:
            raise ValueError("prefix and reference_suffix must have the same dtype")
        assert (
            self.diagonal.mean_ is not None
            and self.diagonal.scale_ is not None
        )
        if prefix.device != self.diagonal.mean_.device:
            raise ValueError(
                "prefix, reference_suffix, and the transport must be on the same device"
            )
        if prefix.dtype != self.diagonal.mean_.dtype:
            raise ValueError(
                "prefix, reference_suffix, and the transport must have the same dtype"
            )
        if prefix.shape[1] == 0:
            current = prefix.clone()
        else:
            standardized_prefix = (
                prefix - self.diagonal.mean_[: prefix.shape[1]]
            ) / self.diagonal.scale_[: prefix.shape[1]]
            current = self.diagonal._evaluate_dimensions(
                standardized_prefix,
                start=0,
                with_derivatives=False,
            )[0]
        stage_prefixes: list[Tensor] = []
        for stage in self.stages:
            stage_prefixes.append(current)
            current = stage._evaluate_prefix(
                current,
                compute_logdet=False,
            )[0]
        suffix = reference_suffix
        for stage, stage_prefix in zip(
            reversed(self.stages),
            reversed(stage_prefixes),
            strict=True,
        ):
            stage_reference = torch.cat((stage_prefix, suffix), dim=1)
            suffix = stage._inverse_from_reference(
                stage_reference,
                prefix.shape[1],
            )[:, prefix.shape[1] :]
        reconstructed_suffix = self.diagonal._inverse_suffix(
            suffix,
            prefix.shape[1],
        )
        return torch.cat((prefix, reconstructed_suffix), dim=1)


__all__ = ["BoostedSoftTreeTransport"]
