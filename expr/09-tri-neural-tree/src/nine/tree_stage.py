from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from nine.hard_tree import HardTreeSpline

_DENSE_COMPONENT_LIMIT = 512
_SCREEN_COMPONENT_BATCH_SIZE = 2048
_SCREEN_FEATURE_COUNT = 8
_SCREEN_ELEMENT_BUDGET = 1_000_000


class HardTreeTransportStage(nn.Module):
    """One lower-triangular stage of causal hard trees."""

    def __init__(
        self,
        *,
        max_wavelet_level: int | None = None,
        max_leaves: int = 32,
        max_fit_iterations: int = 10,
        max_parent_distance: int | None = None,
    ) -> None:
        super().__init__()
        if max_parent_distance is not None and max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
        self.max_wavelet_level = max_wavelet_level
        self.max_leaves = max_leaves
        self.max_fit_iterations = max_fit_iterations
        self.max_parent_distance = max_parent_distance
        self.components = nn.ModuleList()
        self.register_buffer(
            "component_indices_",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer("conditioning_mean_", torch.empty(0))
        self.register_buffer("conditioning_scale_", torch.empty(0))
        self.n_features_in_: int | None = None
        self.training_nll_: float | None = None
        self.criterion_: float | None = None
        self.stopping_reason_: str | None = None
        self._placement_requested = False

    def _apply(self, fn: Any, recurse: bool = True) -> HardTreeTransportStage:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "n_features_in": self.n_features_in_,
            "training_nll": self.training_nll_,
            "criterion": self.criterion_,
            "stopping_reason": self.stopping_reason_,
            "component_count": len(self.components),
            "max_wavelet_level": self.max_wavelet_level,
            "max_leaves": self.max_leaves,
            "max_fit_iterations": self.max_fit_iterations,
            "max_parent_distance": self.max_parent_distance,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.n_features_in_ = state["n_features_in"]
        self.training_nll_ = state["training_nll"]
        self.criterion_ = state["criterion"]
        self.stopping_reason_ = state["stopping_reason"]
        self.max_wavelet_level = state["max_wavelet_level"]
        self.max_leaves = state["max_leaves"]
        self.max_fit_iterations = state["max_fit_iterations"]
        self.max_parent_distance = state.get("max_parent_distance")

    def _new_component(self) -> HardTreeSpline:
        return HardTreeSpline(
            max_wavelet_level=self.max_wavelet_level,
            max_leaves=self.max_leaves,
            max_fit_iterations=self.max_fit_iterations,
        )

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
        saved_component_count = extra.get("component_count", 0)
        while len(self.components) > saved_component_count:
            self.components.pop(-1)
        while len(self.components) < saved_component_count:
            component = self._new_component()
            if self._placement_requested:
                component = component.to(
                    device=self.conditioning_mean_.device,
                    dtype=self.conditioning_mean_.dtype,
                )
            self.components.append(component)
        for name in ("conditioning_mean_", "conditioning_scale_"):
            key = f"{prefix}{name}"
            if key in state_dict:
                saved = state_dict[key]
                current = getattr(self, name)
                if self._placement_requested:
                    setattr(
                        self,
                        name,
                        torch.empty(
                            saved.shape,
                            dtype=current.dtype,
                            device=current.device,
                        ),
                    )
                else:
                    setattr(self, name, torch.empty_like(saved))
        key = f"{prefix}component_indices_"
        if key in state_dict:
            saved = state_dict[key]
            self.component_indices_ = torch.empty(
                saved.shape,
                dtype=torch.long,
                device=self.conditioning_mean_.device,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _haar_feature_count(dimension: int, max_level: int | None) -> int:
        if dimension < 1:
            return 0
        available_level = dimension.bit_length() - 1
        final_level = (
            available_level
            if max_level is None
            else min(max_level, available_level)
        )
        return dimension + sum(
            dimension // (1 << level)
            for level in range(1, final_level + 1)
        )

    def _screen_components(
        self,
        standardized: Tensor,
    ) -> list[tuple[int, Tensor]]:
        sample_count, dimension_count = standardized.shape
        minimum = HardTreeSpline._minimum_leaf_size(sample_count)
        if sample_count < 2 * minimum:
            return []

        squared = standardized.square()
        squared = squared - squared.mean(dim=0)
        squared_scale = squared.std(dim=0, correction=0).clamp_min(
            torch.finfo(standardized.dtype).eps
        )
        squared = squared / squared_scale
        feature_rows = torch.arange(
            dimension_count,
            device=standardized.device,
        ).unsqueeze(1)
        split_counts = torch.arange(
            1,
            sample_count + 1,
            dtype=standardized.dtype,
            device=standardized.device,
        ).view(-1, 1, 1)
        right_counts = sample_count - split_counts
        selected_component_batches: list[Tensor] = []
        selected_feature_batches: list[Tensor] = []
        component_batch_size = min(
            _SCREEN_COMPONENT_BATCH_SIZE,
            max(
                1,
                _SCREEN_ELEMENT_BUDGET
                // (sample_count * _SCREEN_FEATURE_COUNT),
            ),
            max(
                1,
                _SCREEN_ELEMENT_BUDGET
                // min(
                    dimension_count,
                    (self.max_parent_distance or dimension_count)
                    + _SCREEN_COMPONENT_BATCH_SIZE,
                ),
            ),
        )

        for start in range(1, dimension_count, component_batch_size):
            stop = min(start + component_batch_size, dimension_count)
            targets = standardized[:, start:stop]
            source_start = (
                0
                if self.max_parent_distance is None
                else max(0, start - self.max_parent_distance)
            )
            source_features = standardized[:, source_start:stop]
            score = source_features.T @ targets
            score.abs_()
            scale_score = source_features.T @ squared[:, start:stop]
            scale_score.abs_()
            torch.maximum(score, scale_score, out=score)
            score.div_(sample_count)
            del scale_score
            component_columns = torch.arange(
                start,
                stop,
                device=standardized.device,
            ).unsqueeze(0)
            local_feature_rows = feature_rows[source_start:stop]
            unavailable = local_feature_rows >= component_columns
            if self.max_parent_distance is not None:
                unavailable = unavailable | (
                    local_feature_rows
                    < component_columns - self.max_parent_distance
                )
            score.masked_fill_(unavailable, -torch.inf)
            candidate_count = min(_SCREEN_FEATURE_COUNT, score.shape[0])
            feature_indices = (
                score.topk(candidate_count, dim=0).indices.T + source_start
            )

            feature_values = standardized[:, feature_indices]
            order = feature_values.argsort(dim=0)
            ordered_targets = torch.gather(
                targets.unsqueeze(2).expand(-1, -1, candidate_count),
                0,
                order,
            )
            cumulative = ordered_targets.cumsum(dim=0)
            cumulative_squares = ordered_targets.square().cumsum(dim=0)
            totals = cumulative[-1]
            total_squares = cumulative_squares[-1]
            left_variance = (
                (
                    cumulative_squares
                    - cumulative.square() / split_counts
                )
                / split_counts
            ).clamp_min(torch.finfo(standardized.dtype).eps)
            safe_right_counts = right_counts.clamp_min(1)
            right_variance = (
                (
                    total_squares
                    - cumulative_squares
                    - (totals - cumulative).square() / safe_right_counts
                )
                / safe_right_counts
            ).clamp_min(torch.finfo(standardized.dtype).eps)
            parent_variance = (
                (total_squares - totals.square() / sample_count) / sample_count
            ).clamp_min(torch.finfo(standardized.dtype).eps)
            gain = 0.5 * (
                sample_count * parent_variance.log()
                - split_counts * left_variance.log()
                - right_counts * right_variance.log()
            )
            gain[: minimum - 1] = -torch.inf
            gain[sample_count - minimum :] = -torch.inf
            valid_features = feature_indices < torch.arange(
                start,
                stop,
                device=standardized.device,
            ).unsqueeze(1)
            if self.max_parent_distance is not None:
                valid_features = valid_features & (
                    feature_indices
                    >= torch.arange(
                        start,
                        stop,
                        device=standardized.device,
                    ).unsqueeze(1)
                    - self.max_parent_distance
                )
            gain.masked_fill_(~valid_features.unsqueeze(0), -torch.inf)
            best_gain = gain.amax(dim=(0, 2))
            search_feature_counts = torch.tensor(
                [
                    self._haar_feature_count(
                        (
                            component
                            if self.max_parent_distance is None
                            else min(component, self.max_parent_distance)
                        ),
                        self.max_wavelet_level,
                    )
                    for component in range(start, stop)
                ],
                dtype=standardized.dtype,
                device=standardized.device,
            )
            threshold_count = min(
                16,
                max(3, int(sample_count**0.5)),
                sample_count - 2 * minimum + 1,
            )
            search_cost = 2.0 * (
                search_feature_counts.clamp_min(1).log()
                + math.log(max(1, threshold_count))
            )
            viable = torch.nonzero(
                best_gain > search_cost,
                as_tuple=False,
            ).flatten()
            selected_component_batches.append(viable + start)
            selected_feature_batches.append(feature_indices[viable])

        if not selected_component_batches:
            return []
        selected_components = torch.cat(selected_component_batches)
        selected_features = torch.cat(selected_feature_batches)
        return [
            (
                component,
                features[
                    (features < component)
                    & (
                        features
                        >= component
                        - (
                            component
                            if self.max_parent_distance is None
                            else self.max_parent_distance
                        )
                    )
                ],
            )
            for component, features in zip(
                selected_components.tolist(),
                selected_features,
            )
        ]

    def fit(self, values: Tensor) -> HardTreeTransportStage:
        if not isinstance(values, Tensor) or not values.is_floating_point():
            raise TypeError("values must be a floating-point torch.Tensor")
        if values.ndim != 2:
            raise ValueError("values must have shape (N, D)")
        if not torch.all(torch.isfinite(values)):
            raise ValueError("values must be finite")
        samples = values.detach()
        self.n_features_in_ = samples.shape[1]
        self.conditioning_mean_ = samples.mean(dim=0)
        self.conditioning_scale_ = samples.std(dim=0, correction=0).clamp_min(
            torch.finfo(samples.dtype).eps
        )
        standardized = (
            samples - self.conditioning_mean_
        ) / self.conditioning_scale_
        self.components = nn.ModuleList()
        if samples.shape[1] <= _DENSE_COMPONENT_LIMIT:
            component_candidates = [
                (component, None)
                for component in range(samples.shape[1])
            ]
        else:
            component_candidates = self._screen_components(standardized)
        fitted_indices: list[int] = []
        for component, feature_indices in component_candidates:
            tree = self._new_component()
            fit_options: dict[str, Any] = {}
            if feature_indices is not None:
                search_feature_count = self._haar_feature_count(
                    (
                        component
                        if self.max_parent_distance is None
                        else min(component, self.max_parent_distance)
                    ),
                    self.max_wavelet_level,
                )
                fit_options = {
                    "candidate_starts": feature_indices,
                    "candidate_widths": torch.ones_like(feature_indices),
                    "search_feature_count": search_feature_count,
                }
            if feature_indices is not None and samples.device.type == "mps":
                local_conditioning = standardized[:, feature_indices].to("cpu")
                local_targets = samples[:, component : component + 1].to("cpu")
                local_starts = torch.arange(feature_indices.numel())
                tree.fit(
                    local_conditioning,
                    local_targets,
                    candidate_starts=local_starts,
                    candidate_widths=torch.ones_like(local_starts),
                    search_feature_count=search_feature_count,
                )
                tree.feature_starts_ = feature_indices.to("cpu")[
                    tree.feature_starts_
                ]
                tree.condition_dimension_ = component
                tree = tree.to(device=samples.device, dtype=samples.dtype)
            else:
                tree.fit(
                    standardized[:, :component],
                    samples[:, component : component + 1],
                    **fit_options,
                )
            if samples.shape[1] <= _DENSE_COMPONENT_LIMIT or not tree.is_identity:
                self.components.append(tree)
                fitted_indices.append(component)
        self.component_indices_ = torch.tensor(
            fitted_indices,
            dtype=torch.long,
            device=samples.device,
        )
        mapped, logdet = self.forward_with_logdet(samples)
        self.training_nll_ = float(
            (0.5 * mapped.square().sum(dim=1) - logdet).sum().item()
        )
        self.criterion_ = float(
            sum(float(tree.criterion_.item()) for tree in self.components)
        )
        self.stopping_reason_ = (
            "candidate_stage_is_identity"
            if self.is_identity
            else "no_improving_split"
        )
        return self

    @property
    def is_identity(self) -> bool:
        return all(tree.is_identity for tree in self.components)

    def _validate(self, values: Tensor, name: str) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the stage")
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have shape (N, {self.n_features_in_})")
        if values.device != self.conditioning_mean_.device:
            raise ValueError(f"{name} and the stage must be on the same device")
        if values.dtype != self.conditioning_mean_.dtype:
            raise ValueError(f"{name} and the stage must have the same dtype")

    def forward_with_logdet(self, values: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(values, "values")
        standardized = (
            values - self.conditioning_mean_
        ) / self.conditioning_scale_
        result = values.clone()
        logdet = values.new_zeros(values.shape[0])
        for component_tensor, tree in zip(self.component_indices_, self.components):
            component = int(component_tensor.item())
            mapped, log_derivative = tree.forward_with_log_derivative(
                standardized[:, :component],
                values[:, component : component + 1],
            )
            result[:, component : component + 1] = mapped
            logdet = logdet + log_derivative[:, 0]
        return result, logdet

    def forward(self, values: Tensor) -> Tensor:
        return self.forward_with_logdet(values)[0]

    def log_abs_det_jacobian(self, values: Tensor) -> Tensor:
        return self.forward_with_logdet(values)[1]

    def inverse(self, reference: Tensor) -> Tensor:
        self._validate(reference, "reference")
        reconstructed = reference.clone()
        for component_tensor, tree in zip(self.component_indices_, self.components):
            component = int(component_tensor.item())
            conditioning = (
                reconstructed[:, :component] - self.conditioning_mean_[:component]
            ) / self.conditioning_scale_[:component]
            reconstructed[:, component : component + 1] = tree.inverse(
                conditioning,
                reference[:, component : component + 1],
            )
        return reconstructed

    def conditional_inverse(
        self,
        prefix: Tensor,
        reference_suffix: Tensor,
    ) -> Tensor:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the stage")
        if prefix.ndim != 2 or reference_suffix.ndim != 2:
            raise ValueError("prefix and reference_suffix must be matrices")
        if (
            prefix.shape[0] != reference_suffix.shape[0]
            or prefix.shape[1] + reference_suffix.shape[1] != self.n_features_in_
        ):
            raise ValueError("prefix and reference_suffix have incompatible shapes")
        reconstructed = torch.cat((prefix, reference_suffix), dim=1)
        for component_tensor, tree in zip(self.component_indices_, self.components):
            component = int(component_tensor.item())
            if component < prefix.shape[1]:
                continue
            conditioning = (
                reconstructed[:, :component] - self.conditioning_mean_[:component]
            ) / self.conditioning_scale_[:component]
            suffix_index = component - prefix.shape[1]
            reconstructed[:, component : component + 1] = tree.inverse(
                conditioning,
                reference_suffix[:, suffix_index : suffix_index + 1],
            )
        return reconstructed
