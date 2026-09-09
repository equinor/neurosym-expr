from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from nine.monotone_pspline import BatchedDiagonalSplineTransport
from nine.wavelet_features import HaarFeatureSet, make_haar_features, project_haar

_SPLIT_FINALIST_COUNT = 4


@dataclass
class _BuildNode:
    indices: Tensor
    spline: BatchedDiagonalSplineTransport | None
    criterion: float
    feature: int = -1
    threshold: float = 0.0
    left: _BuildNode | None = None
    right: _BuildNode | None = None


@dataclass
class _Split:
    improvement: float
    feature: int
    threshold: float
    left_indices: Tensor
    right_indices: Tensor
    left_spline: BatchedDiagonalSplineTransport
    right_spline: BatchedDiagonalSplineTransport
    left_criterion: float
    right_criterion: float


class HardTreeSpline(nn.Module):
    """A hard tree over causal Haar features with monotone spline leaves."""

    def __init__(
        self,
        *,
        max_wavelet_level: int | None = None,
        max_leaves: int = 32,
        max_fit_iterations: int = 10,
    ) -> None:
        super().__init__()
        if max_wavelet_level is not None and max_wavelet_level < 0:
            raise ValueError("max_wavelet_level cannot be negative")
        if max_leaves < 1:
            raise ValueError("max_leaves must be positive")
        if max_fit_iterations < 1:
            raise ValueError("max_fit_iterations must be positive")
        self.max_wavelet_level = max_wavelet_level
        self.max_leaves = max_leaves
        self.max_fit_iterations = max_fit_iterations
        self.leaves = nn.ModuleList()
        self.register_buffer("feature_starts_", torch.empty(0, dtype=torch.long))
        self.register_buffer("feature_widths_", torch.empty(0, dtype=torch.long))
        self.register_buffer("node_features_", torch.empty(0, dtype=torch.long))
        self.register_buffer("node_thresholds_", torch.empty(0))
        self.register_buffer("node_left_", torch.empty(0, dtype=torch.long))
        self.register_buffer("node_right_", torch.empty(0, dtype=torch.long))
        self.register_buffer("node_leaf_", torch.empty(0, dtype=torch.long))
        self.register_buffer("criterion_", torch.tensor(float("inf")))
        self.condition_dimension_: int | None = None
        self.singleton_features_ = False
        self.training_nll_: float | None = None
        self.stopping_reason_: str | None = None
        self._placement_requested = False

    def _apply(self, fn: Any, recurse: bool = True) -> HardTreeSpline:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "condition_dimension": self.condition_dimension_,
            "singleton_features": self.singleton_features_,
            "training_nll": self.training_nll_,
            "stopping_reason": self.stopping_reason_,
            "leaf_count": len(self.leaves),
            "max_wavelet_level": self.max_wavelet_level,
            "max_leaves": self.max_leaves,
            "max_fit_iterations": self.max_fit_iterations,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.condition_dimension_ = state["condition_dimension"]
        self.singleton_features_ = state.get("singleton_features", False)
        self.training_nll_ = state["training_nll"]
        self.stopping_reason_ = state["stopping_reason"]
        self.max_wavelet_level = state["max_wavelet_level"]
        self.max_leaves = state["max_leaves"]
        self.max_fit_iterations = state["max_fit_iterations"]

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
        saved_leaf_count = extra.get("leaf_count", 0)
        while len(self.leaves) > saved_leaf_count:
            self.leaves.pop(-1)
        while len(self.leaves) < saved_leaf_count:
            leaf = BatchedDiagonalSplineTransport()
            if self._placement_requested:
                leaf = leaf.to(
                    device=self.node_thresholds_.device,
                    dtype=self.node_thresholds_.dtype,
                )
            self.leaves.append(leaf)
        for name in (
            "feature_starts_",
            "feature_widths_",
            "node_features_",
            "node_thresholds_",
            "node_left_",
            "node_right_",
            "node_leaf_",
            "criterion_",
        ):
            key = f"{prefix}{name}"
            if key in state_dict:
                saved = state_dict[key]
                current = getattr(self, name)
                if self._placement_requested:
                    dtype = current.dtype if saved.is_floating_point() else saved.dtype
                    setattr(
                        self,
                        name,
                        torch.empty(
                            saved.shape,
                            dtype=dtype,
                            device=current.device,
                        ),
                    )
                else:
                    setattr(self, name, torch.empty_like(saved))
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
    def _identity_criterion(values: Tensor) -> float:
        return float(values.square().sum().item())

    @staticmethod
    def _spline_statistics(
        spline: BatchedDiagonalSplineTransport,
        values: Tensor,
        total_count: int,
    ) -> tuple[float, float]:
        assert (
            spline.mean_ is not None
            and spline.scale_ is not None
            and spline.effective_dof_ is not None
        )
        standardized = (values - spline.mean_) / spline.scale_
        reference, derivative = spline._evaluate(standardized)
        assert derivative is not None
        nll = (
            0.5 * reference.square()
            - torch.log(derivative / spline.scale_)
        ).sum()
        dof = spline.effective_dof_.sum()
        criterion = 2.0 * nll + math.log(total_count) * dof
        return float(nll.item()), float(criterion.item())

    def _fit_spline(
        self,
        values: Tensor,
        total_count: int,
    ) -> tuple[BatchedDiagonalSplineTransport, float]:
        spline = BatchedDiagonalSplineTransport(
            max_fit_iterations=self.max_fit_iterations
        ).fit(values)
        _, criterion = self._spline_statistics(spline, values, total_count)
        return spline, criterion

    def _fit_spline_batch(
        self,
        values: Tensor,
        total_count: int,
    ) -> tuple[BatchedDiagonalSplineTransport, Tensor]:
        spline = BatchedDiagonalSplineTransport(
            max_fit_iterations=self.max_fit_iterations
        ).fit(values)
        assert spline.mean_ is not None and spline.scale_ is not None
        assert spline.effective_dof_ is not None
        standardized = (values - spline.mean_) / spline.scale_
        reference, derivative = spline._evaluate(standardized)
        assert derivative is not None
        nll = (
            0.5 * reference.square()
            - torch.log(derivative / spline.scale_)
        ).sum(dim=0)
        criterion = 2.0 * nll + math.log(total_count) * spline.effective_dof_
        return spline, criterion

    @staticmethod
    def _minimum_leaf_size(sample_count: int) -> int:
        basis_count = math.ceil(sample_count ** (1.0 / 3.0)) + 4
        return max(8, basis_count + 1)

    @staticmethod
    def _thresholds(values: Tensor, minimum_leaf_size: int) -> Tensor:
        count = values.numel()
        available = count - 2 * minimum_leaf_size + 1
        if available <= 0:
            return values.new_empty(0)
        threshold_count = min(16, max(3, int(math.sqrt(count))), available)
        lower = minimum_leaf_size / count
        upper = 1.0 - lower
        quantiles = torch.linspace(
            lower,
            upper,
            threshold_count,
            dtype=values.dtype,
            device=values.device,
        )
        return torch.unique(torch.quantile(values, quantiles))

    def _best_split(
        self,
        node: _BuildNode,
        projections: Tensor,
        targets: Tensor,
        total_count: int,
        search_feature_count: int,
    ) -> _Split | None:
        minimum = self._minimum_leaf_size(node.indices.numel())
        if node.indices.numel() < 2 * minimum:
            return None
        feature_count = projections.shape[1]
        candidates: list[tuple[int, Tensor, float, Tensor]] = []
        node_targets = targets[node.indices, 0]
        node_targets = node_targets - node_targets.mean()
        node_target_squares = node_targets.square()
        total_sum = node_targets.sum()
        total_square_sum = node_target_squares.sum()
        variance_floor = (
            torch.finfo(targets.dtype).eps
            * node_target_squares.mean()
        ).clamp_min(torch.finfo(targets.dtype).tiny)
        for feature in range(feature_count):
            feature_values = projections[node.indices, feature]
            thresholds = self._thresholds(feature_values, minimum)
            if thresholds.numel() == 0:
                continue
            search_cost = 2.0 * (
                math.log(max(1, search_feature_count))
                + math.log(max(1, thresholds.numel()))
            )
            left_masks = feature_values.unsqueeze(1) <= thresholds.unsqueeze(0)
            left_count_tensor = left_masks.sum(dim=0)
            left_count_values = left_count_tensor.to(targets.dtype)
            right_count_values = node.indices.numel() - left_count_values
            mask_values = left_masks.to(targets.dtype)
            left_sum = mask_values.T @ node_targets
            left_square_sum = mask_values.T @ node_target_squares
            left_variance = (
                (
                    left_square_sum
                    - left_sum.square() / left_count_values
                )
                / left_count_values
            ).clamp_min(variance_floor)
            right_sum = total_sum - left_sum
            right_variance = (
                (
                    total_square_sum
                    - left_square_sum
                    - right_sum.square() / right_count_values
                )
                / right_count_values
            ).clamp_min(variance_floor)
            proxy_criterion = (
                left_count_values * left_variance.log()
                + right_count_values * right_variance.log()
                + search_cost
            )
            valid = (left_count_tensor >= minimum) & (
                left_count_tensor <= node.indices.numel() - minimum
            )
            for threshold_index in torch.nonzero(
                valid,
                as_tuple=False,
            ).flatten().tolist():
                candidates.append(
                    (
                        feature,
                        thresholds[threshold_index],
                        search_cost,
                        proxy_criterion[threshold_index],
                    )
                )

        if not candidates:
            return None

        finalist_count = min(_SPLIT_FINALIST_COUNT, len(candidates))
        finalist_indices = torch.topk(
            torch.stack([candidate[3] for candidate in candidates]),
            finalist_count,
            largest=False,
        ).indices.tolist()
        partitions: list[tuple[int, Tensor, Tensor, Tensor, float]] = []
        for finalist_index in finalist_indices:
            feature, threshold, search_cost, _ = candidates[finalist_index]
            left_mask = projections[node.indices, feature] <= threshold
            partitions.append(
                (
                    feature,
                    threshold,
                    node.indices[left_mask],
                    node.indices[~left_mask],
                    search_cost,
                )
            )
        criteria = targets.new_full((len(partitions), 2), torch.inf)
        fitted_batches: dict[
            int,
            tuple[BatchedDiagonalSplineTransport, list[tuple[int, int]]],
        ] = {}
        fitted_individual: dict[tuple[int, int], BatchedDiagonalSplineTransport] = {}
        groups: dict[int, list[tuple[int, int]]] = {}
        for partition_index, partition in enumerate(partitions):
            for side in range(2):
                indices = partition[2 + side]
                groups.setdefault(indices.numel(), []).append(
                    (partition_index, side)
                )
        for child_count, children in groups.items():
            child_values = torch.stack(
                [
                    targets[partitions[index][2 + side], 0]
                    for index, side in children
                ],
                dim=1,
            )
            try:
                spline_batch, batch_criteria = self._fit_spline_batch(
                    child_values,
                    total_count,
                )
                rows, columns = zip(*children, strict=True)
                criteria[list(rows), list(columns)] = batch_criteria
                fitted_batches[child_count] = (spline_batch, children)
                continue
            except ValueError:
                pass
            except RuntimeError as error:
                if str(error) not in {
                    "could not find a descent direction",
                    "line search failed",
                }:
                    raise
            for partition_index, side in children:
                try:
                    spline, criterion = self._fit_spline(
                        targets[partitions[partition_index][2 + side]],
                        total_count,
                    )
                except ValueError:
                    continue
                except RuntimeError as error:
                    if str(error) not in {
                        "could not find a descent direction",
                        "line search failed",
                    }:
                        raise
                    continue
                criteria[partition_index, side] = criterion
                fitted_individual[(side, partition_index)] = spline

        search_costs = targets.new_tensor(
            [partition[4] for partition in partitions]
        )
        improvements = node.criterion - criteria.sum(dim=1) - search_costs
        best_index = int(torch.argmax(improvements).item())
        best_improvement = float(improvements[best_index].item())
        tolerance = 100.0 * torch.finfo(targets.dtype).eps * (
            1.0 + abs(node.criterion)
        )
        if not math.isfinite(best_improvement) or best_improvement <= tolerance:
            return None

        def selected_spline(side: int) -> BatchedDiagonalSplineTransport:
            individual = fitted_individual.get((side, best_index))
            if individual is not None:
                return individual
            child_count = partitions[best_index][2 + side].numel()
            batch, children = fitted_batches[child_count]
            return batch.select_dimension(children.index((best_index, side)))

        feature, threshold, left_indices, right_indices, _ = partitions[best_index]
        return _Split(
            improvement=best_improvement,
            feature=feature,
            threshold=float(threshold.item()),
            left_indices=left_indices,
            right_indices=right_indices,
            left_spline=selected_spline(0),
            right_spline=selected_spline(1),
            left_criterion=float(criteria[best_index, 0].item()),
            right_criterion=float(criteria[best_index, 1].item()),
        )

    def fit(
        self,
        conditioning: Tensor,
        targets: Tensor,
        *,
        candidate_starts: Tensor | None = None,
        candidate_widths: Tensor | None = None,
        search_feature_count: int | None = None,
    ) -> HardTreeSpline:
        if conditioning.ndim != 2:
            raise ValueError("conditioning must have shape (N, P)")
        if targets.ndim != 2 or targets.shape[1] != 1:
            raise ValueError("targets must have shape (N, 1)")
        if conditioning.shape[0] != targets.shape[0]:
            raise ValueError("conditioning and targets must have equal row counts")
        if conditioning.device != targets.device or conditioning.dtype != targets.dtype:
            raise ValueError("conditioning and targets must share device and dtype")
        if not torch.all(torch.isfinite(conditioning)) or not torch.all(
            torch.isfinite(targets)
        ):
            raise ValueError("training values must be finite")

        sample_count, condition_dimension = conditioning.shape
        self.leaves = nn.ModuleList()
        self.condition_dimension_ = condition_dimension
        self.singleton_features_ = False
        all_indices = torch.arange(sample_count, device=targets.device)
        identity_criterion = self._identity_criterion(targets)
        if condition_dimension == 0:
            self.feature_starts_ = torch.empty(
                0,
                dtype=torch.long,
                device=targets.device,
            )
            self.feature_widths_ = torch.empty_like(self.feature_starts_)
            self.node_features_ = torch.tensor(
                [-1],
                dtype=torch.long,
                device=targets.device,
            )
            self.node_thresholds_ = targets.new_zeros(1)
            self.node_left_ = torch.tensor(
                [-1],
                dtype=torch.long,
                device=targets.device,
            )
            self.node_right_ = self.node_left_.clone()
            self.node_leaf_ = self.node_left_.clone()
            self.criterion_ = targets.new_tensor(identity_criterion)
            self.training_nll_ = 0.5 * identity_criterion
            self.stopping_reason_ = "no_causal_features"
            return self
        if (candidate_starts is None) != (candidate_widths is None):
            raise ValueError(
                "candidate_starts and candidate_widths must be provided together"
            )
        if candidate_starts is None:
            features = make_haar_features(
                condition_dimension,
                self.max_wavelet_level,
            ).to(conditioning.device)
        else:
            if (
                candidate_starts.ndim != 1
                or candidate_widths is None
                or candidate_widths.ndim != 1
                or candidate_starts.shape != candidate_widths.shape
                or candidate_starts.device != conditioning.device
                or candidate_widths.device != conditioning.device
            ):
                raise ValueError(
                    "candidate feature metadata must be equal-length device vectors"
                )
            features = HaarFeatureSet(
                starts=candidate_starts,
                widths=candidate_widths,
                levels=torch.zeros_like(candidate_starts),
            )
            self.singleton_features_ = bool(torch.all(candidate_widths == 1).item())
        self.feature_starts_ = features.starts
        self.feature_widths_ = features.widths
        projections = (
            conditioning[:, features.starts]
            if self.singleton_features_
            else project_haar(
                conditioning,
                features.starts,
                features.widths,
            )
        )
        root = _BuildNode(all_indices, None, identity_criterion)
        terminal_nodes = [root]
        tree_criterion = identity_criterion
        full_search_feature_count = (
            projections.shape[1]
            if search_feature_count is None
            else search_feature_count
        )

        while len(terminal_nodes) < self.max_leaves:
            candidates = [
                (
                    node,
                    self._best_split(
                        node,
                        projections,
                        targets,
                        sample_count,
                        full_search_feature_count,
                    ),
                )
                for node in terminal_nodes
            ]
            candidates = [
                (node, split)
                for node, split in candidates
                if split is not None
            ]
            if not candidates:
                self.stopping_reason_ = "no_improving_split"
                break
            node, split = max(candidates, key=lambda item: item[1].improvement)
            assert split is not None
            node.feature = split.feature
            node.threshold = split.threshold
            node.left = _BuildNode(
                split.left_indices,
                split.left_spline,
                split.left_criterion,
            )
            node.right = _BuildNode(
                split.right_indices,
                split.right_spline,
                split.right_criterion,
            )
            tree_criterion -= split.improvement
            terminal_nodes = [
                terminal for terminal in terminal_nodes if terminal is not node
            ]
            terminal_nodes.extend((node.left, node.right))
        else:
            self.stopping_reason_ = "search_budget_exhausted"

        self.leaves = nn.ModuleList()
        node_features: list[int] = []
        node_thresholds: list[float] = []
        node_left: list[int] = []
        node_right: list[int] = []
        node_leaf: list[int] = []

        def flatten(node: _BuildNode) -> int:
            index = len(node_features)
            node_features.append(node.feature)
            node_thresholds.append(node.threshold)
            node_left.append(-1)
            node_right.append(-1)
            node_leaf.append(-1)
            if node.left is None or node.right is None:
                if node.spline is not None:
                    node_leaf[index] = len(self.leaves)
                    self.leaves.append(node.spline)
                return index
            left_index = flatten(node.left)
            right_index = flatten(node.right)
            node_left[index] = left_index
            node_right[index] = right_index
            return index

        flatten(root)
        self.node_features_ = torch.tensor(
            node_features,
            dtype=torch.long,
            device=targets.device,
        )
        self.node_thresholds_ = targets.new_tensor(node_thresholds)
        self.node_left_ = torch.tensor(
            node_left,
            dtype=torch.long,
            device=targets.device,
        )
        self.node_right_ = torch.tensor(
            node_right,
            dtype=torch.long,
            device=targets.device,
        )
        self.node_leaf_ = torch.tensor(
            node_leaf,
            dtype=torch.long,
            device=targets.device,
        )
        self.criterion_ = targets.new_tensor(tree_criterion)
        transformed, log_derivative = self.forward_with_log_derivative(
            conditioning,
            targets,
        )
        self.training_nll_ = float(
            (0.5 * transformed.square() - log_derivative).sum().item()
        )
        return self

    @property
    def is_identity(self) -> bool:
        return len(self.leaves) == 0

    def _route(self, conditioning: Tensor) -> Tensor:
        if self.condition_dimension_ is None:
            raise RuntimeError("fit must be called before routing")
        if conditioning.ndim != 2 or conditioning.shape[1] != self.condition_dimension_:
            raise ValueError(
                f"conditioning must have shape (N, {self.condition_dimension_})"
            )
        if self.is_identity:
            return torch.full(
                (conditioning.shape[0],),
                -1,
                dtype=torch.long,
                device=conditioning.device,
            )
        projections = (
            conditioning[:, self.feature_starts_]
            if self.singleton_features_
            else project_haar(
                conditioning,
                self.feature_starts_,
                self.feature_widths_,
            )
        )
        nodes = torch.zeros(
            conditioning.shape[0],
            dtype=torch.long,
            device=conditioning.device,
        )
        unresolved = self.node_leaf_[nodes] < 0
        while bool(torch.any(unresolved)):
            rows = torch.nonzero(unresolved, as_tuple=False).flatten()
            active = nodes[rows]
            feature = self.node_features_[active]
            go_left = (
                projections[rows, feature] <= self.node_thresholds_[active]
            )
            nodes[rows] = torch.where(
                go_left,
                self.node_left_[active],
                self.node_right_[active],
            )
            unresolved = self.node_leaf_[nodes] < 0
        return self.node_leaf_[nodes]

    def forward_with_log_derivative(
        self,
        conditioning: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if targets.ndim != 2 or targets.shape[1] != 1:
            raise ValueError("targets must have shape (N, 1)")
        if self.is_identity:
            return targets, torch.zeros_like(targets)
        leaf_ids = self._route(conditioning)
        result = torch.empty_like(targets)
        log_derivative = torch.empty_like(targets)
        for leaf_id, spline in enumerate(self.leaves):
            selected = leaf_ids == leaf_id
            if not bool(torch.any(selected)):
                continue
            values = targets[selected]
            assert spline.mean_ is not None and spline.scale_ is not None
            mapped, derivative = spline._evaluate(
                (values - spline.mean_) / spline.scale_
            )
            assert derivative is not None
            result[selected] = mapped
            log_derivative[selected] = torch.log(derivative / spline.scale_)
        return result, log_derivative

    def forward(self, conditioning: Tensor, targets: Tensor) -> Tensor:
        return self.forward_with_log_derivative(conditioning, targets)[0]

    def inverse(self, conditioning: Tensor, reference: Tensor) -> Tensor:
        if reference.ndim != 2 or reference.shape[1] != 1:
            raise ValueError("reference must have shape (N, 1)")
        if self.is_identity:
            return reference
        leaf_ids = self._route(conditioning)
        result = torch.empty_like(reference)
        for leaf_id, spline in enumerate(self.leaves):
            selected = leaf_ids == leaf_id
            if bool(torch.any(selected)):
                result[selected] = spline.inverse(reference[selected])
        return result
