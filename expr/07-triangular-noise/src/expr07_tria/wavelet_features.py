from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HaarFeatureSet:
    """A causal dictionary of one-dimensional Haar projection features."""

    starts: Tensor
    widths: Tensor
    levels: Tensor

    @property
    def ends(self) -> Tensor:
        return self.starts + self.widths

    def to(self, device: torch.device) -> HaarFeatureSet:
        return HaarFeatureSet(
            starts=self.starts.to(device=device),
            widths=self.widths.to(device=device),
            levels=self.levels.to(device=device),
        )


def make_haar_features(
    dimension_count: int,
    max_level: int | None = None,
) -> HaarFeatureSet:
    """Create singleton scaling atoms and aligned Haar detail atoms."""
    if dimension_count < 1:
        raise ValueError("dimension_count must be positive")
    if max_level is not None and max_level < 0:
        raise ValueError("max_level cannot be negative")

    available_level = int(math.log2(dimension_count))
    final_level = available_level if max_level is None else min(
        max_level,
        available_level,
    )
    starts = list(range(dimension_count))
    widths = [1] * dimension_count
    levels = [0] * dimension_count
    for level in range(1, final_level + 1):
        width = 1 << level
        for start in range(0, dimension_count - width + 1, width):
            starts.append(start)
            widths.append(width)
            levels.append(level)

    return HaarFeatureSet(
        starts=torch.tensor(starts, dtype=torch.long),
        widths=torch.tensor(widths, dtype=torch.long),
        levels=torch.tensor(levels, dtype=torch.long),
    )


def project_haar(
    values: Tensor,
    starts: Tensor,
    widths: Tensor,
) -> Tensor:
    """Evaluate selected normalized Haar atoms on ``(samples, dimensions)``."""
    if values.ndim != 2:
        raise ValueError("values must have shape (N, D)")
    if starts.ndim != 1 or widths.ndim != 1 or starts.shape != widths.shape:
        raise ValueError("starts and widths must be equal-length vectors")
    if starts.device != values.device or widths.device != values.device:
        raise ValueError("feature metadata must be on the values device")
    if torch.any(starts < 0) or torch.any(widths < 1):
        raise ValueError("feature supports must be non-negative and non-empty")
    if torch.any(starts + widths > values.shape[1]):
        raise ValueError("feature support extends beyond the input")
    if starts.numel() == 0:
        return values.new_empty((values.shape[0], 0))

    prefix = torch.cat(
        (values.new_zeros((values.shape[0], 1)), values.cumsum(dim=1)),
        dim=1,
    )
    singletons = widths == 1
    safe_half = torch.div(widths, 2, rounding_mode="floor").clamp(min=1)
    middle = starts + safe_half
    ends = starts + widths
    left_sum = prefix[:, middle] - prefix[:, starts]
    right_sum = prefix[:, ends] - prefix[:, middle]
    details = (left_sum - right_sum) / widths.to(values.dtype).sqrt().unsqueeze(0)
    singleton_values = values[:, starts]
    return torch.where(singletons.unsqueeze(0), singleton_values, details)
