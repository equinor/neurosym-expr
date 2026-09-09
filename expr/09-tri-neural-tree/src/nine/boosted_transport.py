from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from nine.monotone_pspline import BatchedDiagonalSplineTransport
from nine.tree_stage import HardTreeTransportStage


class BoostedHardTreeTransport(nn.Module):
    """Self-selecting composition of a diagonal map and hard-tree stages."""

    def __init__(
        self,
        *,
        max_stages: int = 8,
        max_leaves: int = 32,
        max_wavelet_level: int | None = None,
        max_fit_iterations: int = 10,
        max_parent_distance: int | None = None,
    ) -> None:
        super().__init__()
        if max_stages < 0:
            raise ValueError("max_stages cannot be negative")
        if max_parent_distance is not None and max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
        self.max_stages = max_stages
        self.max_leaves = max_leaves
        self.max_wavelet_level = max_wavelet_level
        self.max_fit_iterations = max_fit_iterations
        self.max_parent_distance = max_parent_distance
        self.diagonal = BatchedDiagonalSplineTransport(
            max_fit_iterations=max_fit_iterations
        )
        self.stages = nn.ModuleList()
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int | None = None
        self.training_nll_: float | None = None
        self.criterion_improvements_: list[float] = []
        self.stopping_reason_: str | None = None
        self.register_buffer("_placement_", torch.empty(0), persistent=False)
        self._placement_requested = False

    def _apply(self, fn: Any, recurse: bool = True) -> BoostedHardTreeTransport:
        result = super()._apply(fn, recurse=recurse)
        self._placement_requested = True
        return result

    def _new_stage(self) -> HardTreeTransportStage:
        return HardTreeTransportStage(
            max_wavelet_level=self.max_wavelet_level,
            max_leaves=self.max_leaves,
            max_fit_iterations=self.max_fit_iterations,
            max_parent_distance=self.max_parent_distance,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "max_stages": self.max_stages,
            "max_leaves": self.max_leaves,
            "max_wavelet_level": self.max_wavelet_level,
            "max_fit_iterations": self.max_fit_iterations,
            "max_parent_distance": self.max_parent_distance,
            "n_features_in": self.n_features_in_,
            "n_samples_seen": self.n_samples_seen_,
            "training_nll": self.training_nll_,
            "criterion_improvements": self.criterion_improvements_,
            "stopping_reason": self.stopping_reason_,
            "stage_count": len(self.stages),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.max_stages = state["max_stages"]
        self.max_leaves = state["max_leaves"]
        self.max_wavelet_level = state["max_wavelet_level"]
        self.max_fit_iterations = state["max_fit_iterations"]
        self.max_parent_distance = state.get("max_parent_distance")
        self.n_features_in_ = state["n_features_in"]
        self.n_samples_seen_ = state["n_samples_seen"]
        self.training_nll_ = state["training_nll"]
        self.criterion_improvements_ = state["criterion_improvements"]
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
        return sum(
            len(component.leaves)
            for stage in self.stages
            for component in stage.components
        )

    def fit(self, values: Tensor) -> BoostedHardTreeTransport:
        self.diagonal.fit(values)
        self.n_samples_seen_, self.n_features_in_ = values.shape
        current = self.diagonal(values).detach()
        self.stages = nn.ModuleList()
        self.criterion_improvements_ = []
        current_nll = float(0.5 * current.square().sum().item())

        for _ in range(self.max_stages):
            candidate = self._new_stage().fit(current)
            if candidate.is_identity:
                self.stopping_reason_ = "candidate_stage_is_identity"
                break
            mapped, logdet = candidate.forward_with_logdet(current)
            candidate_nll = float(
                (0.5 * mapped.square().sum(dim=1) - logdet).sum().item()
            )
            improvement = current_nll - candidate_nll
            tolerance = 100.0 * torch.finfo(values.dtype).eps * (
                1.0 + abs(current_nll)
            )
            if improvement <= tolerance:
                self.stopping_reason_ = "no_global_criterion_improvement"
                break
            self.stages.append(candidate)
            self.criterion_improvements_.append(improvement)
            current = mapped.detach()
            current_nll = candidate_nll
        else:
            self.stopping_reason_ = "search_budget_exhausted"

        reference, logdet = self._forward_with_logdet(values)
        self.training_nll_ = float(
            (
                0.5 * reference.square()
                + 0.5 * math.log(2.0 * math.pi)
            ).sum().item()
            - logdet.sum().item()
        )
        return self

    def _validate(self, values: Tensor, name: str) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have shape (N, {self.n_features_in_})")

    def forward(self, values: Tensor) -> Tensor:
        return self._forward_with_logdet(values)[0]

    def _forward_with_logdet(self, values: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(values, "values")
        assert self.diagonal.mean_ is not None and self.diagonal.scale_ is not None
        standardized = (values - self.diagonal.mean_) / self.diagonal.scale_
        result, derivative = self.diagonal._evaluate(standardized)
        assert derivative is not None
        logdet = torch.log(derivative / self.diagonal.scale_).sum(dim=1)
        for stage in self.stages:
            result, stage_logdet = stage.forward_with_logdet(result)
            logdet = logdet + stage_logdet
        return result, logdet

    def log_abs_det_jacobian(self, values: Tensor) -> Tensor:
        return self._forward_with_logdet(values)[1]

    def log_prob(self, values: Tensor) -> Tensor:
        reference, logdet = self._forward_with_logdet(values)
        return (
            -0.5 * (reference.square() + math.log(2.0 * math.pi)).sum(dim=1)
            + logdet
        )

    def inverse(self, reference: Tensor) -> Tensor:
        self._validate(reference, "reference")
        result = reference
        for stage in reversed(self.stages):
            result = stage.inverse(result)
        return self.diagonal.inverse(result)

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
        assert self.diagonal.mean_ is not None
        full = self.diagonal.mean_.unsqueeze(0).expand(prefix.shape[0], -1).clone()
        full[:, : prefix.shape[1]] = prefix
        current = self.diagonal(full)
        for stage in self.stages:
            current = stage(current)
        reference = torch.cat(
            (current[:, : prefix.shape[1]], reference_suffix),
            dim=1,
        )
        return self.inverse(reference)
