from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nine.rational_quadratic_spline import (
    _rational_quadratic_spline,
    identity_parameters,
)
from nine.wavelet_features import (
    HaarFeatureSet,
    _project_haar_from_prefix,
    make_haar_features,
    project_haar,
)


class SoftTreeRationalQuadraticSpline(nn.Module):
    """A causal soft binary tree that predicts one monotone RQ spline."""

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
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if tail_bound <= 0:
            raise ValueError("tail_bound must be positive")
        if num_bins * 1e-3 >= 2.0 * tail_bound:
            raise ValueError(
                "num_bins * minimum bin size must be less than 2 * tail_bound"
            )
        if max_wavelet_level is not None and max_wavelet_level < 0:
            raise ValueError("max_wavelet_level cannot be negative")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive")
        if not 0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")

        self.max_depth = max_depth
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.max_wavelet_level = max_wavelet_level
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_temperature = min_temperature
        self.validation_fraction = validation_fraction

        node_count = (1 << max_depth) - 1
        leaf_count = 1 << max_depth
        width, height, derivative = identity_parameters(
            num_bins,
            (leaf_count,),
        )
        self.node_thresholds = nn.Parameter(torch.zeros(node_count))
        initial_temperature = math.log(math.expm1(1.0 - min_temperature))
        self.node_raw_temperatures = nn.Parameter(
            torch.full((node_count,), initial_temperature)
        )
        self.leaf_width_logits = nn.Parameter(width)
        self.leaf_height_logits = nn.Parameter(height)
        self.leaf_derivative_logits = nn.Parameter(derivative)
        self.register_buffer("feature_starts_", torch.empty(0, dtype=torch.long))
        self.register_buffer("feature_widths_", torch.empty(0, dtype=torch.long))
        self.register_buffer("node_features_", torch.empty(0, dtype=torch.long))
        self.condition_dimension_: int | None = None
        self.training_nll_: float | None = None
        self.validation_nll_: float | None = None
        self.stopping_reason_: str | None = None
        self.accepted_ = False
        self._placement_requested = False

    def _apply(
        self,
        fn: Any,
        recurse: bool = True,
    ) -> SoftTreeRationalQuadraticSpline:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    @property
    def leaf_count(self) -> int:
        return 1 << self.max_depth

    @property
    def is_identity(self) -> bool:
        return not self.accepted_

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
            "condition_dimension": self.condition_dimension_,
            "training_nll": self.training_nll_,
            "validation_nll": self.validation_nll_,
            "stopping_reason": self.stopping_reason_,
            "accepted": self.accepted_,
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
        self.condition_dimension_ = state["condition_dimension"]
        self.training_nll_ = state["training_nll"]
        self.validation_nll_ = state["validation_nll"]
        self.stopping_reason_ = state["stopping_reason"]
        self.accepted_ = state["accepted"]

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
        parameter_names = (
            "node_thresholds",
            "node_raw_temperatures",
            "leaf_width_logits",
            "leaf_height_logits",
            "leaf_derivative_logits",
        )
        for name in parameter_names:
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
        for name in ("feature_starts_", "feature_widths_", "node_features_"):
            key = f"{prefix}{name}"
            if key in state_dict:
                saved = state_dict[key]
                current = getattr(self, name)
                device = current.device if self._placement_requested else saved.device
                setattr(
                    self,
                    name,
                    torch.empty(saved.shape, dtype=saved.dtype, device=device),
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

    def _features(
        self,
        conditioning: Tensor,
        candidate_starts: Tensor | None,
        candidate_widths: Tensor | None,
        projection_prefix: Tensor | None = None,
    ) -> tuple[HaarFeatureSet, Tensor]:
        if (candidate_starts is None) != (candidate_widths is None):
            raise ValueError(
                "candidate_starts and candidate_widths must be provided together"
            )
        if candidate_starts is None:
            features = make_haar_features(
                conditioning.shape[1],
                self.max_wavelet_level,
            ).to(conditioning.device)
        else:
            assert candidate_widths is not None
            if (
                candidate_starts.ndim != 1
                or candidate_widths.ndim != 1
                or candidate_starts.shape != candidate_widths.shape
                or candidate_starts.device != conditioning.device
                or candidate_widths.device != conditioning.device
                or candidate_starts.numel() == 0
            ):
                raise ValueError("candidate feature metadata must be device vectors")
            features = HaarFeatureSet(
                candidate_starts,
                candidate_widths,
                torch.zeros_like(candidate_starts),
            )
        if torch.all(features.widths == 1):
            projections = conditioning[:, features.starts]
        elif projection_prefix is not None:
            projections = _project_haar_from_prefix(
                projection_prefix,
                features.starts,
                features.widths,
            )
        else:
            projections = project_haar(
                conditioning,
                features.starts,
                features.widths,
            )
        return features, projections

    def _initialize_topology(
        self,
        projections: Tensor,
        targets: Tensor,
    ) -> None:
        centered = targets[:, 0] - targets[:, 0].mean()
        squared = centered.square() - centered.square().mean()
        score = torch.maximum(
            (projections.T @ centered).abs(),
            (projections.T @ squared).abs(),
        )
        node_count = self.node_thresholds.numel()
        selected = torch.topk(
            score,
            min(node_count, score.numel()),
        ).indices
        if selected.numel() < node_count:
            selected = selected.repeat(
                math.ceil(node_count / selected.numel())
            )
        self.node_features_ = selected[:node_count]
        selected_values = projections[:, self.node_features_]
        quantiles = (
            torch.arange(node_count, device=targets.device, dtype=targets.dtype)
            .remainder(3)
            .add(1)
            .div(4)
        )
        with torch.no_grad():
            self.node_thresholds.copy_(
                torch.stack(
                    [
                        torch.quantile(selected_values[:, index], quantiles[index])
                        for index in range(node_count)
                    ]
                )
            )
            identity = identity_parameters(
                self.num_bins,
                (self.leaf_count,),
                dtype=targets.dtype,
                device=targets.device,
            )
            self.leaf_width_logits.copy_(identity[0])
            self.leaf_height_logits.copy_(identity[1])
            self.leaf_derivative_logits.copy_(identity[2])
            offsets = torch.linspace(
                -1.0,
                1.0,
                self.leaf_count,
                dtype=targets.dtype,
                device=targets.device,
            ).unsqueeze(1)
            shape = torch.linspace(
                -1.0,
                1.0,
                self.num_bins,
                dtype=targets.dtype,
                device=targets.device,
            ).unsqueeze(0)
            self.leaf_height_logits.add_(1e-3 * offsets * shape)

    def _project(self, conditioning: Tensor) -> Tensor:
        return project_haar(
            conditioning,
            self.feature_starts_,
            self.feature_widths_,
        )

    def leaf_probabilities(self, conditioning: Tensor) -> Tensor:
        self._validate_conditioning(conditioning)
        if self.is_identity or self.node_features_.numel() == 0:
            return conditioning.new_ones((conditioning.shape[0], 1))
        projections = self._project(conditioning)
        return self._leaf_probabilities_from_projections(projections)

    def _leaf_probabilities_from_projections(
        self,
        projections: Tensor,
    ) -> Tensor:
        return _leaf_probabilities_from_parameters(
            projections,
            self.node_thresholds,
            self.node_raw_temperatures,
            max_depth=self.max_depth,
            min_temperature=self.min_temperature,
        )

    def _blended_parameters(
        self,
        conditioning: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        probabilities = self.leaf_probabilities(conditioning)
        return self._blend_leaf_parameters(probabilities)

    def _blend_leaf_parameters(
        self,
        probabilities: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            probabilities @ self.leaf_width_logits,
            probabilities @ self.leaf_height_logits,
            probabilities @ self.leaf_derivative_logits,
        )

    def _validate_conditioning(self, conditioning: Tensor) -> None:
        if self.condition_dimension_ is None:
            raise RuntimeError("fit must be called before applying the tree")
        if conditioning.ndim != 2 or conditioning.shape[1] != self.condition_dimension_:
            raise ValueError(
                f"conditioning must have shape (N, {self.condition_dimension_})"
            )
        if conditioning.dtype != self.node_thresholds.dtype:
            raise ValueError("conditioning and tree must have the same dtype")
        if conditioning.device != self.node_thresholds.device:
            raise ValueError("conditioning and tree must be on the same device")

    def _transform(
        self,
        conditioning: Tensor,
        values: Tensor,
        *,
        inverse: bool,
        compute_log_derivative: bool,
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_conditioning(conditioning)
        if values.ndim != 2 or values.shape != (conditioning.shape[0], 1):
            raise ValueError("values must have shape (N, 1)")
        if self.is_identity:
            log_derivative = torch.zeros_like(values) if compute_log_derivative else None
            return values, log_derivative
        projections = self._project(conditioning)
        return self._transform_from_projections(
            projections,
            values,
            inverse=inverse,
            compute_log_derivative=compute_log_derivative,
        )

    def _transform_from_projections(
        self,
        projections: Tensor,
        values: Tensor,
        *,
        inverse: bool,
        compute_log_derivative: bool,
    ) -> tuple[Tensor, Tensor | None]:
        return _transform_from_parameters(
            projections,
            values,
            (
                self.node_thresholds,
                self.node_raw_temperatures,
                self.leaf_width_logits,
                self.leaf_height_logits,
                self.leaf_derivative_logits,
            ),
            max_depth=self.max_depth,
            min_temperature=self.min_temperature,
            inverse=inverse,
            tail_bound=self.tail_bound,
            compute_logdet=compute_log_derivative,
        )

    def forward_with_log_derivative(
        self,
        conditioning: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mapped, log_derivative = self._transform(
            conditioning,
            targets,
            inverse=False,
            compute_log_derivative=True,
        )
        assert log_derivative is not None
        return mapped, log_derivative

    def forward(self, conditioning: Tensor, targets: Tensor) -> Tensor:
        return self._transform(
            conditioning,
            targets,
            inverse=False,
            compute_log_derivative=False,
        )[0]

    def inverse(self, conditioning: Tensor, reference: Tensor) -> Tensor:
        return self._transform(
            conditioning,
            reference,
            inverse=True,
            compute_log_derivative=False,
        )[0]

    def fit(
        self,
        conditioning: Tensor,
        targets: Tensor,
        *,
        candidate_starts: Tensor | None = None,
        candidate_widths: Tensor | None = None,
    ) -> SoftTreeRationalQuadraticSpline:
        _fit_soft_tree_batch(
            [self],
            [conditioning],
            [targets],
            [candidate_starts],
            [candidate_widths],
        )
        return self


def _leaf_probabilities_from_parameters(
    projections: Tensor,
    thresholds: Tensor,
    raw_temperatures: Tensor,
    *,
    max_depth: int,
    min_temperature: float,
) -> Tensor:
    temperatures = min_temperature + F.softplus(raw_temperatures)
    gates = torch.sigmoid(
        (projections - thresholds.unsqueeze(0)) / temperatures.unsqueeze(0)
    )
    probabilities = projections.new_ones((projections.shape[0], 1))
    node_offset = 0
    for depth in range(max_depth):
        level_count = 1 << depth
        level_gates = gates[:, node_offset : node_offset + level_count]
        probabilities = torch.stack(
            (
                probabilities * (1.0 - level_gates),
                probabilities * level_gates,
            ),
            dim=2,
        ).flatten(1)
        node_offset += level_count
    return probabilities


def _transform_from_parameters(
    projections: Tensor,
    values: Tensor,
    parameters: Sequence[Tensor],
    *,
    max_depth: int,
    min_temperature: float,
    tail_bound: float,
    inverse: bool,
    compute_logdet: bool,
) -> tuple[Tensor, Tensor | None]:
    thresholds, raw_temperatures, widths, heights, derivatives = parameters
    probabilities = _leaf_probabilities_from_parameters(
        projections,
        thresholds,
        raw_temperatures,
        max_depth=max_depth,
        min_temperature=min_temperature,
    )
    transformed, logabsdet = _rational_quadratic_spline(
        values[:, 0],
        probabilities @ widths,
        probabilities @ heights,
        probabilities @ derivatives,
        inverse=inverse,
        tail_bound=tail_bound,
        compute_logabsdet=compute_logdet,
    )
    return (
        transformed.unsqueeze(1),
        None if logabsdet is None else logabsdet.unsqueeze(1),
    )


def _batched_leaf_probabilities(
    projections: Tensor,
    thresholds: Tensor,
    raw_temperatures: Tensor,
    *,
    max_depth: int,
    min_temperature: float,
) -> Tensor:
    temperatures = min_temperature + F.softplus(raw_temperatures)
    gates = torch.sigmoid(
        (projections - thresholds.unsqueeze(0)) / temperatures.unsqueeze(0)
    )
    probabilities = projections.new_ones((*projections.shape[:2], 1))
    node_offset = 0
    for depth in range(max_depth):
        level_count = 1 << depth
        level_gates = gates[:, :, node_offset : node_offset + level_count]
        probabilities = torch.stack(
            (
                probabilities * (1.0 - level_gates),
                probabilities * level_gates,
            ),
            dim=3,
        ).flatten(2)
        node_offset += level_count
    return probabilities


def _batched_transform(
    projections: Tensor,
    targets: Tensor,
    parameters: Sequence[Tensor],
    *,
    max_depth: int,
    min_temperature: float,
    tail_bound: float,
    inverse: bool = False,
    compute_logdet: bool = True,
) -> tuple[Tensor, Tensor | None]:
    thresholds, raw_temperatures, widths, heights, derivatives = parameters
    probabilities = _batched_leaf_probabilities(
        projections,
        thresholds,
        raw_temperatures,
        max_depth=max_depth,
        min_temperature=min_temperature,
    )
    return _rational_quadratic_spline(
        targets,
        torch.einsum("ncl,clb->ncb", probabilities, widths),
        torch.einsum("ncl,clb->ncb", probabilities, heights),
        torch.einsum("ncl,clb->ncb", probabilities, derivatives),
        inverse=inverse,
        tail_bound=tail_bound,
        compute_logabsdet=compute_logdet,
    )


def _clip_batched_grad_norm_(
    parameters: Sequence[Tensor],
    max_norm: float,
) -> None:
    tree_count = parameters[0].shape[0]
    squared_norm = parameters[0].new_zeros(tree_count)
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm.add_(parameter.grad.flatten(1).square().sum(dim=1))
    norms = squared_norm.sqrt()
    scales = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            shape = (tree_count,) + (1,) * (parameter.grad.ndim - 1)
            parameter.grad.mul_(scales.view(shape))


def _fit_soft_tree_batch(
    trees: Sequence[SoftTreeRationalQuadraticSpline],
    conditionings: Sequence[Tensor],
    targets: Sequence[Tensor],
    candidate_starts: Sequence[Tensor | None],
    candidate_widths: Sequence[Tensor | None],
    *,
    projection_prefix: Tensor | None = None,
) -> None:
    if not trees:
        return
    if not (
        len(trees)
        == len(conditionings)
        == len(targets)
        == len(candidate_starts)
        == len(candidate_widths)
    ):
        raise ValueError("batched tree fitting arguments must have equal lengths")

    reference = trees[0]
    shared_configuration = (
        reference.max_depth,
        reference.num_bins,
        reference.tail_bound,
        reference.learning_rate,
        reference.max_epochs,
        reference.patience,
        reference.min_temperature,
        reference.validation_fraction,
    )
    sample_count = targets[0].shape[0]
    for tree, conditioning, target in zip(
        trees,
        conditionings,
        targets,
        strict=True,
    ):
        if conditioning.ndim != 2:
            raise ValueError("conditioning must have shape (N, P)")
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError("targets must have shape (N, 1)")
        if conditioning.shape[0] != target.shape[0]:
            raise ValueError("conditioning and targets must have equal row counts")
        if conditioning.device != target.device or conditioning.dtype != target.dtype:
            raise ValueError("conditioning and targets must share device and dtype")
        if not torch.all(torch.isfinite(conditioning)) or not torch.all(
            torch.isfinite(target)
        ):
            raise ValueError("training values must be finite")
        configuration = (
            tree.max_depth,
            tree.num_bins,
            tree.tail_bound,
            tree.learning_rate,
            tree.max_epochs,
            tree.patience,
            tree.min_temperature,
            tree.validation_fraction,
        )
        if configuration != shared_configuration:
            raise ValueError("batched trees must share their training configuration")
        if target.shape[0] != sample_count:
            raise ValueError("batched trees must have equal sample counts")
        tree.condition_dimension_ = conditioning.shape[1]
        tree.accepted_ = False
        if conditioning.shape[1] == 0:
            if len(trees) != 1:
                raise ValueError("batched trees must have causal features")
            tree.stopping_reason_ = "no_causal_features"
            return

    target_reference = targets[0]
    validation_count = (
        max(1, round(sample_count * reference.validation_fraction))
        if reference.validation_fraction > 0 and sample_count >= 10
        else 0
    )
    validation_mask = torch.zeros(
        sample_count,
        dtype=torch.bool,
        device=target_reference.device,
    )
    if validation_count:
        validation_mask[
            torch.linspace(
                0,
                sample_count - 1,
                validation_count,
                device=target_reference.device,
            ).round().long()
        ] = True
    training_mask = ~validation_mask
    scoring_mask = validation_mask if validation_count else training_mask

    selected_projections = []
    for tree, conditioning, target, starts, widths in zip(
        trees,
        conditionings,
        targets,
        candidate_starts,
        candidate_widths,
        strict=True,
    ):
        tree.to(device=target.device, dtype=target.dtype)
        features, projections = tree._features(
            conditioning,
            starts,
            widths,
            projection_prefix,
        )
        tree._initialize_topology(
            projections[training_mask],
            target[training_mask],
        )
        selected_features = tree.node_features_.clone()
        tree.feature_starts_ = features.starts[selected_features]
        tree.feature_widths_ = features.widths[selected_features]
        tree.node_features_ = torch.arange(
            selected_features.numel(),
            dtype=torch.long,
            device=target.device,
        )
        selected_projections.append(projections[:, selected_features])

    projection_batch = torch.stack(selected_projections, dim=1)
    target_batch = torch.cat(targets, dim=1)
    training_projections = projection_batch[training_mask]
    training_targets = target_batch[training_mask]
    scoring_projections = projection_batch[scoring_mask]
    scoring_targets = target_batch[scoring_mask]
    parameters = [
        nn.Parameter(
            torch.stack([getattr(tree, name).detach() for tree in trees], dim=0)
        )
        for name in (
            "node_thresholds",
            "node_raw_temperatures",
            "leaf_width_logits",
            "leaf_height_logits",
            "leaf_derivative_logits",
        )
    ]
    optimizer = torch.optim.Adam(parameters, lr=reference.learning_rate)
    identity_scores = (0.5 * target_batch[scoring_mask].square()).mean(dim=0)
    best_scores = identity_scores.clone()
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    improved_once = torch.zeros(
        len(trees),
        dtype=torch.bool,
        device=target_reference.device,
    )
    stale_epochs = torch.zeros(
        len(trees),
        dtype=torch.long,
        device=target_reference.device,
    )
    active = torch.ones_like(improved_once)

    for _ in range(reference.max_epochs):
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        active_parameters = [
            parameter[active_indices] for parameter in parameters
        ]
        optimizer.zero_grad()
        mapped, logdet = _batched_transform(
            training_projections[:, active_indices],
            training_targets[:, active_indices],
            active_parameters,
            max_depth=reference.max_depth,
            min_temperature=reference.min_temperature,
            tail_bound=reference.tail_bound,
        )
        assert logdet is not None
        losses = (0.5 * mapped.square() - logdet).mean(dim=0)
        regularization = 1e-5 * (
            active_parameters[2].square().mean(dim=(1, 2))
            + active_parameters[3].square().mean(dim=(1, 2))
        )
        objectives = losses + regularization
        if not torch.all(torch.isfinite(objectives)):
            raise RuntimeError("soft-tree training produced a non-finite objective")
        objectives.sum().backward()
        _clip_batched_grad_norm_(parameters, 10.0)
        inactive = ~active
        if torch.any(inactive):
            for parameter in parameters:
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    moment = state.get(name)
                    if moment is not None:
                        moment[inactive] = 0
        optimizer.step()

        with torch.no_grad():
            scored, score_logdet = _batched_transform(
                scoring_projections[:, active_indices],
                scoring_targets[:, active_indices],
                [parameter[active_indices] for parameter in parameters],
                max_depth=reference.max_depth,
                min_temperature=reference.min_temperature,
                tail_bound=reference.tail_bound,
            )
            assert score_logdet is not None
            scores = (0.5 * scored.square() - score_logdet).mean(dim=0)
            if not torch.all(torch.isfinite(scores)):
                raise RuntimeError("soft-tree scoring produced a non-finite objective")
            active_best_scores = best_scores[active_indices]
            tolerances = 100.0 * torch.finfo(target_reference.dtype).eps * (
                1.0 + active_best_scores.abs()
            )
            improved_active = scores < active_best_scores - tolerances
            improved = torch.zeros_like(active)
            improved[active_indices] = improved_active
            best_scores[active_indices] = torch.where(
                improved_active,
                scores,
                active_best_scores,
            )
            improved_once |= improved
            stale_epochs[active_indices] = torch.where(
                improved_active,
                torch.zeros_like(active_indices),
                stale_epochs[active_indices] + 1,
            )
            for best, parameter in zip(
                best_parameters,
                parameters,
                strict=True,
            ):
                best[improved] = parameter.detach()[improved]
            active = stale_epochs < reference.patience
        if not torch.any(active):
            break

    parameter_names = (
        "node_thresholds",
        "node_raw_temperatures",
        "leaf_width_logits",
        "leaf_height_logits",
        "leaf_derivative_logits",
    )
    with torch.no_grad():
        mapped, logdet = _batched_transform(
            projection_batch,
            target_batch,
            best_parameters,
            max_depth=reference.max_depth,
            min_temperature=reference.min_temperature,
            tail_bound=reference.tail_bound,
        )
        assert logdet is not None
        training_scores = (0.5 * mapped.square() - logdet).sum(dim=0)
        identity_training_scores = (0.5 * target_batch.square()).sum(dim=0)

    for index, tree in enumerate(trees):
        if not bool(improved_once[index]):
            tree.accepted_ = False
            tree.stopping_reason_ = "no_validation_improvement"
            tree.validation_nll_ = float(identity_scores[index].item())
            tree.training_nll_ = float(identity_training_scores[index].item())
            continue
        with torch.no_grad():
            for name, best in zip(parameter_names, best_parameters, strict=True):
                getattr(tree, name).copy_(best[index])
        tree.accepted_ = True
        tree.validation_nll_ = float(best_scores[index].item())
        tree.training_nll_ = float(training_scores[index].item())
        tree.stopping_reason_ = "early_stopping"


__all__ = ["SoftTreeRationalQuadraticSpline"]
