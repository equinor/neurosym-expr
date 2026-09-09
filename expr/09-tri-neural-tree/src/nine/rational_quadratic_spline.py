"""Vectorized monotone rational-quadratic splines with linear identity tails."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def identity_parameters(
    num_bins: int,
    batch_shape: Sequence[int] | torch.Size = (),
    *,
    min_derivative: float = 1e-3,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return logits for an exact identity spline.

    The returned shapes are ``(*batch_shape, num_bins)`` for widths and
    heights and ``(*batch_shape, num_bins - 1)`` for internal derivatives.
    Boundary derivatives are fixed to one by :func:`rational_quadratic_spline`.
    """
    _validate_num_bins(num_bins)
    _validate_minimum("min_derivative", min_derivative)
    if min_derivative >= 1.0:
        raise ValueError("min_derivative must be less than 1 for identity parameters")
    if not isinstance(batch_shape, (Sequence, torch.Size)):
        raise TypeError("batch_shape must be a sequence of non-negative integers")
    shape = tuple(batch_shape)
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise ValueError("batch_shape must contain only non-negative integers")
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point dtype")

    options = {"dtype": dtype, "device": device}
    widths = torch.zeros((*shape, num_bins), **options)
    heights = torch.zeros_like(widths)
    target = torch.as_tensor(1.0 - min_derivative, **options)
    derivative_logit = target + torch.log(-torch.expm1(-target))
    derivatives = derivative_logit.expand(*shape, num_bins - 1).clone()
    return widths, heights, derivatives


def identity_rational_quadratic_spline_parameters(
    num_bins: int,
    batch_shape: Sequence[int] | torch.Size = (),
    *,
    min_derivative: float = 1e-3,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Descriptive alias for :func:`identity_parameters`."""
    return identity_parameters(
        num_bins,
        batch_shape,
        min_derivative=min_derivative,
        dtype=dtype,
        device=device,
    )


def rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    *,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Apply a monotone rational-quadratic spline elementwise.

    Parameter tensors have one final bin dimension and leading dimensions
    broadcastable to ``inputs.shape``. Width and height logits have ``K``
    entries, while derivative logits have ``K - 1`` entries for the internal
    knots. The two boundary derivatives are fixed to one, making the tails
    continuously differentiable linear identities.

    Returns the transformed values and their elementwise log absolute
    derivative. In inverse mode the latter is the inverse derivative.
    """
    tensors = (
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
    )
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("inputs and all spline parameters must be tensors")
    if not inputs.is_floating_point():
        raise TypeError("inputs must have a floating-point dtype")
    for name, parameter in zip(
        ("unnormalized_widths", "unnormalized_heights", "unnormalized_derivatives"),
        tensors[1:],
        strict=True,
    ):
        if not parameter.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if parameter.dtype != inputs.dtype:
            raise TypeError(f"{name} must have the same dtype as inputs")
        if parameter.device != inputs.device:
            raise ValueError(f"{name} must be on the same device as inputs")
        if parameter.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension")
        if not torch.isfinite(parameter).all():
            raise ValueError(f"{name} must contain only finite values")
    if torch.isnan(inputs).any():
        raise ValueError("inputs must not contain NaN values")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")

    num_bins = unnormalized_widths.shape[-1]
    _validate_num_bins(num_bins)
    if unnormalized_heights.shape[-1] != num_bins:
        raise ValueError("width and height logits must have the same final dimension")
    if unnormalized_derivatives.shape[-1] != num_bins - 1:
        raise ValueError("derivative logits must have num_bins - 1 entries")

    tail_bound = _validate_positive_finite("tail_bound", tail_bound)
    min_bin_width = _validate_minimum("min_bin_width", min_bin_width)
    min_bin_height = _validate_minimum("min_bin_height", min_bin_height)
    min_derivative = _validate_minimum("min_derivative", min_derivative)
    interval = 2.0 * tail_bound
    if num_bins * min_bin_width >= interval:
        raise ValueError("num_bins * min_bin_width must be less than 2 * tail_bound")
    if num_bins * min_bin_height >= interval:
        raise ValueError("num_bins * min_bin_height must be less than 2 * tail_bound")

    try:
        output_shape = torch.broadcast_shapes(
            inputs.shape,
            unnormalized_widths.shape[:-1],
            unnormalized_heights.shape[:-1],
            unnormalized_derivatives.shape[:-1],
        )
    except RuntimeError as error:
        raise ValueError(
            "parameter leading dimensions must be broadcastable to inputs.shape"
        ) from error
    if output_shape != inputs.shape:
        raise ValueError(
            "parameter leading dimensions may not expand the shape of inputs"
        )

    outputs, logabsdet = _rational_quadratic_spline(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=inverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        compute_logabsdet=True,
    )
    assert logabsdet is not None
    return outputs, logabsdet


def _rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    *,
    inverse: bool,
    tail_bound: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    compute_logabsdet: bool,
) -> tuple[Tensor, Tensor | None]:
    """Evaluate a spline whose arguments have already been validated."""
    num_bins = unnormalized_widths.shape[-1]
    interval = 2.0 * tail_bound
    widths = min_bin_width + (
        interval - num_bins * min_bin_width
    ) * torch.softmax(unnormalized_widths, dim=-1)
    heights = min_bin_height + (
        interval - num_bins * min_bin_height
    ) * torch.softmax(unnormalized_heights, dim=-1)
    internal_derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    widths = torch.broadcast_to(widths, (*inputs.shape, num_bins))
    heights = torch.broadcast_to(heights, (*inputs.shape, num_bins))
    internal_derivatives = torch.broadcast_to(
        internal_derivatives, (*inputs.shape, num_bins - 1)
    )

    left = inputs.new_full((*inputs.shape, 1), -tail_bound)
    right = inputs.new_full((*inputs.shape, 1), tail_bound)
    cumulative_widths = torch.cat(
        (left, left + torch.cumsum(widths, dim=-1)[..., :-1], right), dim=-1
    )
    cumulative_heights = torch.cat(
        (left, left + torch.cumsum(heights, dim=-1)[..., :-1], right), dim=-1
    )
    widths = cumulative_widths[..., 1:] - cumulative_widths[..., :-1]
    heights = cumulative_heights[..., 1:] - cumulative_heights[..., :-1]
    boundary = inputs.new_ones((*inputs.shape, 1))
    derivatives = torch.cat(
        (boundary, internal_derivatives, boundary), dim=-1
    )

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    bounded_inputs = torch.where(
        inside, inputs, inputs.clamp(min=-tail_bound, max=tail_bound)
    )
    knots = cumulative_heights if inverse else cumulative_widths
    bin_indices = torch.sum(
        bounded_inputs.unsqueeze(-1) >= knots[..., 1:-1], dim=-1
    )

    x_left = _gather(cumulative_widths, bin_indices)
    y_left = _gather(cumulative_heights, bin_indices)
    bin_width = _gather(widths, bin_indices)
    bin_height = _gather(heights, bin_indices)
    derivative_left = _gather(derivatives, bin_indices)
    derivative_right = _gather(derivatives, bin_indices + 1)
    slope = bin_height / bin_width

    if inverse:
        y_delta = bounded_inputs - y_left
        derivative_sum = derivative_left + derivative_right - 2.0 * slope
        a = y_delta * derivative_sum + bin_height * (slope - derivative_left)
        b = bin_height * derivative_left - y_delta * derivative_sum
        c = -slope * y_delta
        discriminant = b.square() - 4.0 * a * c
        sqrt_discriminant = torch.sqrt(discriminant.clamp_min(0.0))
        linear = a.abs() <= torch.finfo(inputs.dtype).eps
        safe_b = torch.where(linear, b, torch.ones_like(b))
        quadratic_denominator = -b - sqrt_discriminant
        safe_quadratic_denominator = torch.where(
            linear,
            torch.ones_like(quadratic_denominator),
            quadratic_denominator,
        )
        linear_root = -c / safe_b
        quadratic_root = (2.0 * c) / safe_quadratic_denominator
        theta = torch.where(linear, linear_root, quadratic_root)
        theta = theta.clamp(0.0, 1.0)
        outputs_inside = x_left + theta * bin_width
    else:
        theta = (bounded_inputs - x_left) / bin_width
        outputs_inside = _evaluate_spline(
            theta,
            y_left,
            bin_height,
            slope,
            derivative_left,
            derivative_right,
        )

    outputs = torch.where(inside, outputs_inside, inputs)
    if not compute_logabsdet:
        return outputs, None

    denominator = slope + (
        derivative_left + derivative_right - 2.0 * slope
    ) * theta * (1.0 - theta)
    derivative_numerator = slope.square() * (
        derivative_right * theta.square()
        + 2.0 * slope * theta * (1.0 - theta)
        + derivative_left * (1.0 - theta).square()
    )
    logabsdet_inside = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    if inverse:
        logabsdet_inside = -logabsdet_inside

    logabsdet = torch.where(inside, logabsdet_inside, torch.zeros_like(inputs))
    return outputs, logabsdet


def _evaluate_spline(
    theta: Tensor,
    y_left: Tensor,
    bin_height: Tensor,
    slope: Tensor,
    derivative_left: Tensor,
    derivative_right: Tensor,
) -> Tensor:
    numerator = bin_height * (
        slope * theta.square() + derivative_left * theta * (1.0 - theta)
    )
    denominator = slope + (
        derivative_left + derivative_right - 2.0 * slope
    ) * theta * (1.0 - theta)
    return y_left + numerator / denominator


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)


def _validate_num_bins(num_bins: int) -> None:
    if not isinstance(num_bins, int):
        raise TypeError("num_bins must be an integer")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")


def _validate_positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_minimum(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


__all__ = [
    "identity_parameters",
    "identity_rational_quadratic_spline_parameters",
    "rational_quadratic_spline",
]
