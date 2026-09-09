from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from nine.soft_tree import (
    SoftTreeRationalQuadraticSpline,
    _batched_transform,
    _fit_soft_tree_batch,
    _transform_from_parameters,
)
from nine.tree_stage import HardTreeTransportStage
from nine.wavelet_features import (
    _batched_project_haar_from_prefix,
    _haar_prefix,
    _project_haar_from_prefix,
    make_haar_features,
)

_PARAMETER_NAMES = (
    "node_thresholds",
    "node_raw_temperatures",
    "leaf_width_logits",
    "leaf_height_logits",
    "leaf_derivative_logits",
)
_FIT_ELEMENT_BUDGET = 1_000_000


class _SoftTreeComponentView:
    def __init__(self, stage: SoftTreeTransportStage, index: int) -> None:
        self._stage = stage
        self._index = index

    @property
    def feature_starts_(self) -> Tensor:
        return self._stage.feature_starts_[self._index]

    @property
    def feature_widths_(self) -> Tensor:
        return self._stage.feature_widths_[self._index]

    @property
    def node_features_(self) -> Tensor:
        return torch.arange(
            self.feature_starts_.numel(),
            device=self.feature_starts_.device,
        )

    @property
    def leaf_count(self) -> int:
        return 1 << self._stage.max_depth

    @property
    def is_identity(self) -> bool:
        return False

    @property
    def condition_dimension_(self) -> int:
        return self._stage._component_indices[self._index]

    @property
    def training_nll_(self) -> float | None:
        return self._stage.component_training_nll_[self._index]

    @training_nll_.setter
    def training_nll_(self, value: float | None) -> None:
        self._stage.component_training_nll_[self._index] = value

    @property
    def validation_nll_(self) -> float | None:
        return self._stage.component_validation_nll_[self._index]

    @validation_nll_.setter
    def validation_nll_(self, value: float | None) -> None:
        self._stage.component_validation_nll_[self._index] = value

    def _parameters(self) -> list[Tensor]:
        return [
            getattr(self._stage, name)[self._index : self._index + 1]
            for name in _PARAMETER_NAMES
        ]

    def _transform(
        self,
        conditioning: Tensor,
        values: Tensor,
        *,
        inverse: bool,
        compute_logdet: bool,
    ) -> tuple[Tensor, Tensor | None]:
        if (
            conditioning.ndim != 2
            or conditioning.shape[1] != self.condition_dimension_
        ):
            raise ValueError(
                f"conditioning must have shape (N, {self.condition_dimension_})"
            )
        if values.shape != (conditioning.shape[0], 1):
            raise ValueError("values must have shape (N, 1)")
        projection_prefix = _haar_prefix(conditioning)
        projections = _project_haar_from_prefix(
            projection_prefix,
            self.feature_starts_,
            self.feature_widths_,
        )
        return _transform_from_parameters(
            projections,
            values,
            [parameter[0] for parameter in self._parameters()],
            max_depth=self._stage.max_depth,
            min_temperature=self._stage.min_temperature,
            tail_bound=self._stage.tail_bound,
            inverse=inverse,
            compute_logdet=compute_logdet,
        )

    def forward_with_log_derivative(
        self,
        conditioning: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mapped, logdet = self._transform(
            conditioning,
            targets,
            inverse=False,
            compute_logdet=True,
        )
        assert logdet is not None
        return mapped, logdet

    def forward(self, conditioning: Tensor, targets: Tensor) -> Tensor:
        return self._transform(
            conditioning,
            targets,
            inverse=False,
            compute_logdet=False,
        )[0]

    def inverse(self, conditioning: Tensor, reference: Tensor) -> Tensor:
        return self._transform(
            conditioning,
            reference,
            inverse=True,
            compute_logdet=False,
        )[0]


class SoftTreeTransportStage(nn.Module):
    """One lower-triangular stage of soft-tree neural splines."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        max_wavelet_level: int | None = None,
        learning_rate: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 30,
        min_temperature: float = 0.05,
        validation_fraction: float = 0.2,
        max_parent_distance: int | None = 256,
    ) -> None:
        super().__init__()
        if max_parent_distance is not None and max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
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
        node_count = (1 << max_depth) - 1
        leaf_count = 1 << max_depth
        self.node_thresholds = nn.Parameter(torch.empty((0, node_count)))
        self.node_raw_temperatures = nn.Parameter(torch.empty((0, node_count)))
        self.leaf_width_logits = nn.Parameter(
            torch.empty((0, leaf_count, num_bins))
        )
        self.leaf_height_logits = nn.Parameter(
            torch.empty((0, leaf_count, num_bins))
        )
        self.leaf_derivative_logits = nn.Parameter(
            torch.empty((0, leaf_count, num_bins - 1))
        )
        self.register_buffer(
            "component_indices_",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "feature_starts_",
            torch.empty((0, node_count), dtype=torch.long),
        )
        self.register_buffer(
            "feature_widths_",
            torch.empty((0, node_count), dtype=torch.long),
        )
        self.register_buffer("conditioning_mean_", torch.empty(0))
        self.register_buffer("conditioning_scale_", torch.empty(0))
        self._component_indices: tuple[int, ...] = ()
        self.component_training_nll_: list[float | None] = []
        self.component_validation_nll_: list[float | None] = []
        self.n_features_in_: int | None = None
        self.training_nll_: float | None = None
        self.stopping_reason_: str | None = None
        self._placement_requested = False
        self._singleton_features = True

    def _apply(self, fn: Any, recurse: bool = True) -> SoftTreeTransportStage:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    def _new_component(self) -> SoftTreeRationalQuadraticSpline:
        return SoftTreeRationalQuadraticSpline(
            max_depth=self.max_depth,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            max_wavelet_level=self.max_wavelet_level,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_temperature=self.min_temperature,
            validation_fraction=self.validation_fraction,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
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
            "n_features_in": self.n_features_in_,
            "training_nll": self.training_nll_,
            "stopping_reason": self.stopping_reason_,
            "component_count": len(self._component_indices),
            "component_training_nll": self.component_training_nll_,
            "component_validation_nll": self.component_validation_nll_,
            "representation_version": 2,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
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
        self.n_features_in_ = state["n_features_in"]
        self.training_nll_ = state["training_nll"]
        self.stopping_reason_ = state["stopping_reason"]
        component_count = state.get("component_count", 0)
        self.component_training_nll_ = list(
            state.get("component_training_nll", [None] * component_count)
        )
        self.component_validation_nll_ = list(
            state.get("component_validation_nll", [None] * component_count)
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
        if extra:
            self.set_extra_state(extra)
        saved_component_count = extra.get("component_count", 0)
        legacy = bool(extra) and extra.get("representation_version", 1) < 2
        if legacy:
            if saved_component_count:
                for name in _PARAMETER_NAMES:
                    state_dict[f"{prefix}{name}"] = torch.stack(
                        [
                            state_dict[f"{prefix}components.{index}.{name}"]
                            for index in range(saved_component_count)
                        ]
                    )
                for name in ("feature_starts_", "feature_widths_"):
                    state_dict[f"{prefix}{name}"] = torch.stack(
                        [
                            state_dict[f"{prefix}components.{index}.{name}"]
                            for index in range(saved_component_count)
                        ]
                    )
            else:
                reference = state_dict[f"{prefix}conditioning_mean_"]
                node_count = (1 << self.max_depth) - 1
                leaf_count = 1 << self.max_depth
                state_dict[f"{prefix}node_thresholds"] = reference.new_empty(
                    (0, node_count)
                )
                state_dict[f"{prefix}node_raw_temperatures"] = reference.new_empty(
                    (0, node_count)
                )
                state_dict[f"{prefix}leaf_width_logits"] = reference.new_empty(
                    (0, leaf_count, self.num_bins)
                )
                state_dict[f"{prefix}leaf_height_logits"] = reference.new_empty(
                    (0, leaf_count, self.num_bins)
                )
                state_dict[f"{prefix}leaf_derivative_logits"] = reference.new_empty(
                    (0, leaf_count, self.num_bins - 1)
                )
                state_dict[f"{prefix}feature_starts_"] = torch.empty(
                    (0, node_count),
                    dtype=torch.long,
                    device=reference.device,
                )
                state_dict[f"{prefix}feature_widths_"] = torch.empty(
                    (0, node_count),
                    dtype=torch.long,
                    device=reference.device,
                )
            component_states = [
                state_dict.get(f"{prefix}components.{index}._extra_state", {})
                for index in range(saved_component_count)
            ]
            if saved_component_count and f"{prefix}component_indices_" not in state_dict:
                try:
                    component_indices = [
                        state["condition_dimension"]
                        for state in component_states
                    ]
                except KeyError as error:
                    error_msgs.append(
                        f"{prefix}legacy component state is missing "
                        "'condition_dimension'"
                    )
                    component_indices = []
                state_dict[f"{prefix}component_indices_"] = torch.tensor(
                    component_indices,
                    dtype=torch.long,
                    device=state_dict[f"{prefix}conditioning_mean_"].device,
                )
            self.component_training_nll_ = [
                state.get("training_nll") for state in component_states
            ]
            self.component_validation_nll_ = [
                state.get("validation_nll") for state in component_states
            ]
            for key in list(state_dict):
                if key.startswith(f"{prefix}components."):
                    del state_dict[key]

        for name in _PARAMETER_NAMES:
            key = f"{prefix}{name}"
            if key not in state_dict:
                continue
            saved = state_dict[key]
            current = getattr(self, name)
            dtype = current.dtype if self._placement_requested else saved.dtype
            device = current.device if self._placement_requested else saved.device
            setattr(
                self,
                name,
                nn.Parameter(torch.empty(saved.shape, dtype=dtype, device=device)),
            )
        for name in (
            "conditioning_mean_",
            "conditioning_scale_",
            "feature_starts_",
            "feature_widths_",
        ):
            key = f"{prefix}{name}"
            if key in state_dict:
                saved = state_dict[key]
                current = getattr(self, name)
                dtype = (
                    current.dtype
                    if self._placement_requested and saved.is_floating_point()
                    else saved.dtype
                )
                device = current.device if self._placement_requested else saved.device
                setattr(
                    self,
                    name,
                    torch.empty(saved.shape, dtype=dtype, device=device),
                )
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
        self._component_indices = tuple(self.component_indices_.tolist())
        self._singleton_features = bool(torch.all(self.feature_widths_ == 1))
        if any(
            left >= right
            for left, right in zip(
                self._component_indices,
                self._component_indices[1:],
            )
        ):
            error_msgs.append(f"{prefix}component indices must be strictly increasing")

    @property
    def components(self) -> tuple[_SoftTreeComponentView, ...]:
        return tuple(
            _SoftTreeComponentView(self, index)
            for index in range(len(self._component_indices))
        )

    @property
    def is_identity(self) -> bool:
        return len(self._component_indices) == 0

    @property
    def leaf_count(self) -> int:
        return len(self._component_indices) * (1 << self.max_depth)

    def _clear_components(self, reference: Tensor) -> None:
        node_count = (1 << self.max_depth) - 1
        leaf_count = 1 << self.max_depth
        self.node_thresholds = nn.Parameter(reference.new_empty((0, node_count)))
        self.node_raw_temperatures = nn.Parameter(
            reference.new_empty((0, node_count))
        )
        self.leaf_width_logits = nn.Parameter(
            reference.new_empty((0, leaf_count, self.num_bins))
        )
        self.leaf_height_logits = nn.Parameter(
            reference.new_empty((0, leaf_count, self.num_bins))
        )
        self.leaf_derivative_logits = nn.Parameter(
            reference.new_empty((0, leaf_count, self.num_bins - 1))
        )
        self.feature_starts_ = torch.empty(
            (0, node_count),
            dtype=torch.long,
            device=reference.device,
        )
        self.feature_widths_ = torch.empty_like(self.feature_starts_)
        self.component_indices_ = torch.empty(
            0,
            dtype=torch.long,
            device=reference.device,
        )
        self._component_indices = ()
        self.component_training_nll_ = []
        self.component_validation_nll_ = []
        self._singleton_features = True

    def _install_components(
        self,
        trees: list[SoftTreeRationalQuadraticSpline],
        indices: list[int],
    ) -> None:
        accepted = [tree for tree in trees if not tree.is_identity]
        accepted_indices = [
            index
            for index, tree in zip(indices, trees, strict=True)
            if not tree.is_identity
        ]
        if not accepted:
            return
        for name in _PARAMETER_NAMES:
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.stack(
                        [getattr(tree, name).detach() for tree in accepted]
                    )
                ),
            )
        self.feature_starts_ = torch.stack(
            [tree.feature_starts_ for tree in accepted]
        )
        self.feature_widths_ = torch.stack(
            [tree.feature_widths_ for tree in accepted]
        )
        self.component_indices_ = torch.tensor(
            accepted_indices,
            dtype=torch.long,
            device=self.conditioning_mean_.device,
        )
        self._component_indices = tuple(accepted_indices)
        self.component_training_nll_ = [
            tree.training_nll_ for tree in accepted
        ]
        self.component_validation_nll_ = [
            tree.validation_nll_ for tree in accepted
        ]
        self._singleton_features = bool(torch.all(self.feature_widths_ == 1))

    def _candidate_features(
        self,
        component: int,
        screened: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        if screened is not None:
            starts = screened.to(device=device)
            return starts, torch.ones_like(starts)
        if self.max_parent_distance is None:
            return None, None
        features = make_haar_features(
            component,
            self.max_wavelet_level,
        ).to(device)
        keep = (
            (features.starts >= component - self.max_parent_distance)
            & (features.ends <= component)
        )
        return features.starts[keep], features.widths[keep]

    def fit(self, values: Tensor) -> SoftTreeTransportStage:
        self._fit_with_output(values)
        return self

    def _fit_with_output(self, values: Tensor) -> tuple[Tensor, Tensor]:
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
        self._clear_components(samples)

        screener = HardTreeTransportStage(
            max_wavelet_level=self.max_wavelet_level,
            max_parent_distance=self.max_parent_distance,
        )
        candidates = screener._screen_components(standardized)

        candidate_trees: list[SoftTreeRationalQuadraticSpline] = []
        candidate_indices: list[int] = []
        candidate_conditioning: list[Tensor] = []
        candidate_targets: list[Tensor] = []
        candidate_starts: list[Tensor | None] = []
        candidate_widths: list[Tensor | None] = []
        for component_index, screened_features in candidates:
            starts, widths = self._candidate_features(
                component_index,
                screened_features,
                samples.device,
            )
            if starts is not None and starts.numel() == 0:
                continue
            tree = self._new_component().to(
                device=samples.device,
                dtype=samples.dtype,
            )
            candidate_trees.append(tree)
            candidate_indices.append(component_index)
            candidate_conditioning.append(standardized[:, :component_index])
            candidate_targets.append(
                samples[:, component_index : component_index + 1]
            )
            candidate_starts.append(starts)
            candidate_widths.append(widths)

        per_component_elements = samples.shape[0] * max(
            (1 << self.max_depth) * self.num_bins,
            1,
        )
        fit_batch_size = max(1, _FIT_ELEMENT_BUDGET // per_component_elements)
        all_singletons = all(widths is not None for widths in candidate_widths)
        if all_singletons and candidate_widths:
            all_singletons = bool(
                torch.all(
                    torch.cat(
                        [
                            widths
                            for widths in candidate_widths
                            if widths is not None
                        ]
                    )
                    == 1
                )
            )
        projection_prefix = None if all_singletons else _haar_prefix(standardized)
        for start in range(0, len(candidate_trees), fit_batch_size):
            stop = start + fit_batch_size
            _fit_soft_tree_batch(
                candidate_trees[start:stop],
                candidate_conditioning[start:stop],
                candidate_targets[start:stop],
                candidate_starts[start:stop],
                candidate_widths[start:stop],
                projection_prefix=projection_prefix,
            )

        self._install_components(candidate_trees, candidate_indices)
        with torch.no_grad():
            mapped, logdet = self.forward_with_logdet(samples)
        self.training_nll_ = float(
            (0.5 * mapped.square().sum(dim=1) - logdet).sum().item()
        )
        self.stopping_reason_ = (
            "candidate_stage_is_identity"
            if self.is_identity
            else "component_search_complete"
        )
        return mapped, logdet

    def _validate(self, values: Tensor, name: str) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the stage")
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have shape (N, {self.n_features_in_})")
        if values.device != self.conditioning_mean_.device:
            raise ValueError(f"{name} and the stage must be on the same device")
        if values.dtype != self.conditioning_mean_.dtype:
            raise ValueError(f"{name} and the stage must have the same dtype")

    def _evaluate_components(
        self,
        values: Tensor,
        *,
        compute_logdet: bool,
    ) -> tuple[Tensor, Tensor | None]:
        return self._evaluate_prefix(values, compute_logdet=compute_logdet)

    def _evaluate_prefix(
        self,
        values: Tensor,
        *,
        compute_logdet: bool,
    ) -> tuple[Tensor, Tensor | None]:
        component_count = sum(
            component < values.shape[1]
            for component in self._component_indices
        )
        standardized = (
            values - self.conditioning_mean_[: values.shape[1]]
        ) / self.conditioning_scale_[: values.shape[1]]
        if component_count == 0:
            return (
                values.clone(),
                values.new_zeros((values.shape[0], 0)) if compute_logdet else None,
            )

        component_slice = slice(0, component_count)
        starts = self.feature_starts_[component_slice]
        widths = self.feature_widths_[component_slice]
        if self._singleton_features:
            projections = standardized[:, starts]
        else:
            projections = _batched_project_haar_from_prefix(
                _haar_prefix(standardized),
                starts,
                widths,
            )
        parameters = [
            getattr(self, name)[component_slice]
            for name in _PARAMETER_NAMES
        ]
        component_indices = self.component_indices_[component_slice]
        mapped, component_logdet = _batched_transform(
            projections,
            values[:, component_indices],
            parameters,
            max_depth=self.max_depth,
            min_temperature=self.min_temperature,
            tail_bound=self.tail_bound,
            compute_logdet=compute_logdet,
        )
        result = values.clone()
        result[:, component_indices] = mapped
        return result, component_logdet

    def _forward(
        self,
        values: Tensor,
        *,
        compute_logdet: bool,
    ) -> tuple[Tensor, Tensor | None]:
        result, component_logdet = self._evaluate_components(
            values,
            compute_logdet=compute_logdet,
        )
        return (
            result,
            None if component_logdet is None else component_logdet.sum(dim=1),
        )

    def forward_with_logdet(self, values: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(values, "values")
        mapped, logdet = self._forward(values, compute_logdet=True)
        assert logdet is not None
        return mapped, logdet

    def forward(self, values: Tensor) -> Tensor:
        self._validate(values, "values")
        return self._forward(values, compute_logdet=False)[0]

    def forward_prefix(self, values: Tensor) -> Tensor:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the stage")
        if values.ndim != 2 or values.shape[1] > self.n_features_in_:
            raise ValueError(
                f"values must have shape (N, P) with P <= {self.n_features_in_}"
            )
        if values.device != self.conditioning_mean_.device:
            raise ValueError("values and the stage must be on the same device")
        if values.dtype != self.conditioning_mean_.dtype:
            raise ValueError("values and the stage must have the same dtype")
        return self._evaluate_prefix(values, compute_logdet=False)[0]

    def log_abs_det_jacobian(self, values: Tensor) -> Tensor:
        return self.forward_with_logdet(values)[1]

    def _inverse_from_reference(
        self,
        reference: Tensor,
        prefix_size: int,
    ) -> Tensor:
        reconstructed = reference.clone()
        projection_prefix = (
            None
            if self._singleton_features
            else reference.new_zeros(
                (reference.shape[0], reference.shape[1] + 1)
            )
        )
        projected_dimension = 0
        for index, component in enumerate(self._component_indices):
            if component < prefix_size:
                continue
            if self._singleton_features:
                starts = self.feature_starts_[index]
                projections = (
                    reconstructed[:, starts] - self.conditioning_mean_[starts]
                ) / self.conditioning_scale_[starts]
            else:
                assert projection_prefix is not None
                standardized = (
                    reconstructed[:, projected_dimension:component]
                    - self.conditioning_mean_[projected_dimension:component]
                ) / self.conditioning_scale_[projected_dimension:component]
                projection_prefix[:, projected_dimension + 1 : component + 1] = (
                    projection_prefix[:, projected_dimension : projected_dimension + 1]
                    + standardized.cumsum(dim=1)
                )
                projections = _project_haar_from_prefix(
                    projection_prefix,
                    self.feature_starts_[index],
                    self.feature_widths_[index],
                )
            reconstructed[:, component : component + 1] = (
                _transform_from_parameters(
                    projections,
                    reference[:, component : component + 1],
                    [
                        getattr(self, name)[index]
                        for name in _PARAMETER_NAMES
                    ],
                    max_depth=self.max_depth,
                    min_temperature=self.min_temperature,
                    tail_bound=self.tail_bound,
                    inverse=True,
                    compute_logdet=False,
                )[0]
            )
            projected_dimension = component
        return reconstructed

    def inverse(self, reference: Tensor) -> Tensor:
        self._validate(reference, "reference")
        return self._inverse_from_reference(reference, 0)

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
        reference = torch.cat((prefix, reference_suffix), dim=1)
        return self._inverse_from_reference(
            reference,
            prefix.shape[1],
        )


__all__ = ["SoftTreeTransportStage"]
