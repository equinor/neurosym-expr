from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_DEGREE = 3
_INNER_MAX_ITERATIONS = 100
_INVERSE_ITERATIONS = 60


def _bounded_basis(x: Tensor, knots: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate cubic B-spline bases and their derivatives."""
    knot_count = knots.shape[1]
    basis_count = knot_count - _DEGREE - 1
    values = (
        (x.unsqueeze(-1) >= knots[:, :-1].unsqueeze(0))
        & (x.unsqueeze(-1) < knots[:, 1:].unsqueeze(0))
    ).to(x.dtype)

    at_right = x >= knots[:, basis_count].unsqueeze(0)
    values[..., -1] = torch.where(at_right, torch.ones_like(x), values[..., -1])

    lower_values: Tensor | None = None
    for level in range(1, _DEGREE + 1):
        size = knot_count - level - 1
        left_denominator = (
            knots[:, level : level + size] - knots[:, :size]
        ).unsqueeze(0)
        right_denominator = (
            knots[:, level + 1 : level + 1 + size]
            - knots[:, 1 : 1 + size]
        ).unsqueeze(0)

        safe_left = torch.where(
            left_denominator != 0,
            left_denominator,
            torch.ones_like(left_denominator),
        )
        safe_right = torch.where(
            right_denominator != 0,
            right_denominator,
            torch.ones_like(right_denominator),
        )
        left = torch.where(
            left_denominator != 0,
            (x.unsqueeze(-1) - knots[:, :size].unsqueeze(0))
            / safe_left
            * values[..., :size],
            torch.zeros_like(values[..., :size]),
        )
        right = torch.where(
            right_denominator != 0,
            (
                knots[:, level + 1 : level + 1 + size].unsqueeze(0)
                - x.unsqueeze(-1)
            )
            / safe_right
            * values[..., 1 : size + 1],
            torch.zeros_like(values[..., :size]),
        )
        values = left + right
        if level == _DEGREE - 1:
            lower_values = values

    assert lower_values is not None
    left_denominator = (
        knots[:, _DEGREE : _DEGREE + basis_count] - knots[:, :basis_count]
    )
    right_denominator = (
        knots[:, _DEGREE + 1 : _DEGREE + 1 + basis_count]
        - knots[:, 1 : 1 + basis_count]
    )
    left_scale = torch.where(
        left_denominator != 0,
        _DEGREE / left_denominator,
        torch.zeros_like(left_denominator),
    )
    right_scale = torch.where(
        right_denominator != 0,
        _DEGREE / right_denominator,
        torch.zeros_like(right_denominator),
    )
    derivatives = (
        left_scale.unsqueeze(0) * lower_values[..., :basis_count]
        - right_scale.unsqueeze(0) * lower_values[..., 1 : basis_count + 1]
    )

    right_values = torch.zeros_like(values)
    right_values[..., -1] = 1.0
    right_derivatives = torch.zeros_like(derivatives)
    final_slope = _DEGREE / (
        knots[:, basis_count] - knots[:, basis_count - 1]
    )
    right_derivatives[..., -2] = -final_slope.unsqueeze(0)
    right_derivatives[..., -1] = final_slope.unsqueeze(0)
    values = torch.where(at_right.unsqueeze(-1), right_values, values)
    derivatives = torch.where(
        at_right.unsqueeze(-1),
        right_derivatives,
        derivatives,
    )
    return values, derivatives


def _bounded_evaluate(
    x: Tensor,
    knots: Tensor,
    coefficients: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate only the four non-zero cubic basis functions."""
    basis_count = knots.shape[1] - _DEGREE - 1
    left = knots[:, _DEGREE]
    right = knots[:, basis_count]
    interval_count = basis_count - _DEGREE
    spans = (
        torch.floor((x - left) / (right - left) * interval_count).to(torch.long)
        + _DEGREE
    ).clamp(min=_DEGREE, max=basis_count - 1)

    levels = torch.arange(1, _DEGREE + 1, device=x.device)
    dimensions = torch.arange(knots.shape[0], device=x.device)[None, :, None]
    left_distances = x.unsqueeze(2) - knots[
        dimensions,
        spans.unsqueeze(2) + 1 - levels,
    ]
    right_distances = (
        knots[dimensions, spans.unsqueeze(2) + levels] - x.unsqueeze(2)
    )

    local_basis = torch.ones_like(x).unsqueeze(2)
    lower_basis: Tensor | None = None
    for level in range(1, _DEGREE + 1):
        saved = torch.zeros_like(x)
        next_basis = []
        for index in range(level):
            denominator = (
                right_distances[..., index]
                + left_distances[..., level - index - 1]
            )
            ratio = torch.where(
                denominator != 0,
                local_basis[..., index] / denominator,
                torch.zeros_like(denominator),
            )
            next_basis.append(saved + right_distances[..., index] * ratio)
            saved = left_distances[..., level - index - 1] * ratio
        next_basis.append(saved)
        local_basis = torch.stack(next_basis, dim=2)
        if level == _DEGREE - 1:
            lower_basis = local_basis

    assert lower_basis is not None
    offsets = torch.arange(-_DEGREE, 1, device=x.device)
    knot_indices = spans.unsqueeze(2) + offsets
    alpha_denominator = (
        knots[dimensions, knot_indices + _DEGREE]
        - knots[dimensions, knot_indices]
    )
    beta_denominator = (
        knots[dimensions, knot_indices + _DEGREE + 1]
        - knots[dimensions, knot_indices + 1]
    )
    alpha = torch.where(
        alpha_denominator != 0,
        _DEGREE / alpha_denominator,
        torch.zeros_like(alpha_denominator),
    )
    beta = torch.where(
        beta_denominator != 0,
        _DEGREE / beta_denominator,
        torch.zeros_like(beta_denominator),
    )
    local_derivatives = (
        alpha * F.pad(lower_basis, (1, 0))
        - beta * F.pad(lower_basis, (0, 1))
    )
    local_coefficients = coefficients[dimensions, knot_indices]
    return (
        torch.sum(local_basis * local_coefficients, dim=2),
        torch.sum(local_derivatives * local_coefficients, dim=2),
    )


def _positive_increments(raw_increments: Tensor) -> Tensor:
    return torch.cat(
        (
            raw_increments[:, :1],
            F.softplus(raw_increments[:, 1:]),
        ),
        dim=1,
    )


def _coefficients(raw_increments: Tensor) -> Tensor:
    return torch.cumsum(_positive_increments(raw_increments), dim=1)


def _smoothing_matrix(size: int, reference: Tensor) -> Tensor:
    identity = torch.eye(size, dtype=reference.dtype, device=reference.device)
    differences = torch.diff(identity, n=2, dim=1)
    return differences @ differences.T


def _objective_derivatives(
    raw_increments: Tensor,
    quadratic: Tensor,
    derivative_basis: Tensor,
    sample_count: int,
) -> tuple[Tensor, Tensor]:
    increments = _positive_increments(raw_increments)
    derivative = torch.bmm(
        derivative_basis,
        increments.unsqueeze(2),
    ).squeeze(2)
    reciprocal = derivative.reciprocal()
    increment_gradient = (
        torch.bmm(quadratic, increments.unsqueeze(2)).squeeze(2)
        - torch.bmm(
            derivative_basis.transpose(1, 2),
            reciprocal.unsqueeze(2),
        ).squeeze(2)
        / sample_count
    )

    increment_hessian = quadratic + torch.bmm(
        derivative_basis.transpose(1, 2),
        derivative_basis
        * (reciprocal.square() / sample_count).unsqueeze(2),
    )

    sigmoid = torch.sigmoid(raw_increments[:, 1:])
    slopes = torch.cat(
        (torch.ones_like(raw_increments[:, :1]), sigmoid),
        dim=1,
    )
    curvatures = torch.cat(
        (
            torch.zeros_like(raw_increments[:, :1]),
            sigmoid * (1.0 - sigmoid),
        ),
        dim=1,
    )
    hessian = (
        increment_hessian * slopes.unsqueeze(1) * slopes.unsqueeze(2)
        + torch.diag_embed(curvatures * increment_gradient)
    )
    return slopes * increment_gradient, hessian


def _objective(
    raw_increments: Tensor,
    quadratic: Tensor,
    derivative_basis: Tensor,
    sample_count: int,
) -> Tensor:
    increments = _positive_increments(raw_increments)
    derivative = torch.bmm(
        derivative_basis,
        increments.unsqueeze(2),
    ).squeeze(2)
    quadratic_increments = torch.bmm(
        quadratic,
        increments.unsqueeze(2),
    ).squeeze(2)
    return (
        0.5 * torch.sum(increments * quadratic_increments, dim=1)
        - torch.log(derivative).sum(dim=1) / sample_count
    )


class BatchedDiagonalSplineTransport(nn.Module):
    """Cubic P-spline transport for about 12,000 independent parameters.

    Input tensors have shape ``(samples, parameters)`` and must contain at
    least 64 samples. Fitting and map evaluation stay on the input device.
    """

    _buffer_names = (
        "mean_",
        "scale_",
        "knots_",
        "raw_coefficients_",
        "effective_dof_",
        "aicc_",
        "nll_",
        "left_values_",
        "right_values_",
        "left_derivatives_",
        "right_derivatives_",
    )

    def __init__(self) -> None:
        super().__init__()
        for name in self._buffer_names:
            self.register_buffer(name, None)
        self.n_samples_seen_: int | None = None
        self.n_features_in_: int | None = None

    def get_extra_state(self) -> dict[str, int | None]:
        return {
            "n_samples_seen": self.n_samples_seen_,
            "n_features_in": self.n_features_in_,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.n_samples_seen_ = state["n_samples_seen"]
        self.n_features_in_ = state["n_features_in"]

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
        for name in self._buffer_names:
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

    @property
    def dtype(self) -> torch.dtype:
        self._require_fitted()
        assert self.mean_ is not None
        return self.mean_.dtype

    @property
    def device(self) -> torch.device:
        self._require_fitted()
        assert self.mean_ is not None
        return self.mean_.device

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")

    def _validate_input(self, value: Tensor, name: str) -> None:
        self._require_fitted()
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{name} must have shape (N, {self.n_features_in_})"
            )
        if value.device != self.device:
            raise ValueError(
                f"{name} is on {value.device}, but the transport is on "
                f"{self.device}"
            )
        if value.dtype != self.dtype:
            raise ValueError(
                f"{name} has dtype {value.dtype}, but the transport uses "
                f"{self.dtype}"
            )

    def fit(self, X: Tensor) -> BatchedDiagonalSplineTransport:
        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor")
        if not X.is_floating_point():
            raise TypeError("X must have a floating-point dtype")
        if X.ndim != 2:
            raise ValueError("X must have shape (N, D)")
        if X.shape[0] < 64:
            raise ValueError("at least 64 samples are required")
        if not torch.all(torch.isfinite(X)):
            raise ValueError("X must contain only finite values")

        samples = X.detach()
        sample_count, dimension_count = samples.shape
        mean = samples.mean(dim=0)
        scale = samples.std(dim=0, correction=0)
        if torch.any(scale <= torch.finfo(samples.dtype).eps):
            raise ValueError("every input dimension must have non-zero variance")

        standardized = (samples - mean) / scale
        inner_knot_count = max(
            1,
            math.ceil(sample_count ** (1.0 / 3.0)),
        )
        quantiles = torch.quantile(
            standardized,
            standardized.new_tensor((0.1, 0.9)),
            dim=0,
        )
        first, last = quantiles.unbind(dim=0)
        if torch.any(first >= last):
            raise ValueError("every input dimension must have varying quantiles")

        positions = torch.linspace(
            0.0,
            1.0,
            inner_knot_count + 2,
            dtype=samples.dtype,
            device=samples.device,
        ).unsqueeze(1)
        real_knots = first.unsqueeze(0) + (
            last - first
        ).unsqueeze(0) * positions
        knots = torch.cat(
            (
                real_knots[:1].repeat(_DEGREE, 1),
                real_knots,
                real_knots[-1:].repeat(_DEGREE, 1),
            ),
            dim=0,
        ).T.contiguous()

        left = knots[:, _DEGREE]
        right = knots[:, -_DEGREE - 1]
        bounded = torch.maximum(
            torch.minimum(standardized, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        design, derivative_basis = _bounded_basis(bounded, knots)
        basis_count = design.shape[2]
        design_by_dimension = design.permute(1, 0, 2).contiguous()
        derivative_by_dimension = derivative_basis.permute(1, 0, 2).contiguous()
        del standardized, bounded
        gram = torch.bmm(
            design_by_dimension.transpose(1, 2),
            design_by_dimension,
        )
        penalty = _smoothing_matrix(basis_count, design)
        del design
        scales = torch.sqrt(
            design_by_dimension.square().sum(dim=(1, 2)) / sample_count
        )
        smoothing = (
            math.exp(2.0) * scales.square()
        ).unsqueeze(1).unsqueeze(2) * penalty.unsqueeze(0)
        cumulative_derivative_basis = (
            derivative_by_dimension.flip(2).cumsum(2).flip(2)
        )
        del derivative_basis
        transformed_gram = gram.flip((1, 2)).cumsum(1).cumsum(2).flip((1, 2))
        transformed_smoothing = (
            smoothing.flip((1, 2)).cumsum(1).cumsum(2).flip((1, 2))
        )
        quadratic = (
            transformed_gram + transformed_smoothing
        ) / sample_count
        identity = torch.eye(
            basis_count,
            dtype=samples.dtype,
            device=samples.device,
        )
        raw_increments = torch.full(
            (dimension_count, basis_count),
            1e-6,
            dtype=samples.dtype,
            device=samples.device,
        )

        tolerance = max(
            1e-8,
            10.0 * torch.finfo(samples.dtype).eps**0.5,
        )
        for iteration in range(_INNER_MAX_ITERATIONS):
            gradient, hessian = _objective_derivatives(
                raw_increments,
                quadratic,
                cumulative_derivative_basis,
                sample_count,
            )
            curvature = torch.maximum(
                torch.ones(dimension_count, device=samples.device),
                torch.diagonal(hessian, dim1=1, dim2=2).abs().mean(dim=1),
            )
            damping = 1e-8 * curvature
            selected = torch.zeros(
                dimension_count,
                dtype=torch.bool,
                device=samples.device,
            )
            direction = torch.zeros_like(raw_increments)
            for _ in range(12):
                candidate = torch.linalg.solve(
                    hessian + damping[:, None, None] * identity,
                    -gradient.unsqueeze(2),
                ).squeeze(2)
                valid = (
                    torch.isfinite(candidate).all(dim=1)
                    & ((gradient * candidate).sum(dim=1) < 0)
                    & ~selected
                )
                direction = torch.where(valid[:, None], candidate, direction)
                selected = selected | valid
                damping = torch.where(selected, damping, damping * 10.0)
                if bool(torch.all(selected)):
                    break
            if not bool(torch.all(selected)):
                raise RuntimeError("could not find a descent direction")

            objective = _objective(
                raw_increments,
                quadratic,
                cumulative_derivative_basis,
                sample_count,
            )
            objective_tolerance = (
                10.0
                * torch.finfo(samples.dtype).eps
                * torch.maximum(torch.ones_like(objective), objective.abs())
            )
            slope = (gradient * direction).sum(dim=1)
            accepted = torch.zeros_like(selected)
            step = torch.ones_like(objective)
            accepted_step = torch.zeros_like(step)
            for _ in range(25):
                candidate = raw_increments + step[:, None] * direction
                candidate_objective = _objective(
                    candidate,
                    quadratic,
                    cumulative_derivative_basis,
                    sample_count,
                )
                accept = (
                    torch.isfinite(candidate_objective)
                    & (
                        candidate_objective
                        <= objective
                        + 1e-4 * step * slope
                        + objective_tolerance
                    )
                    & ~accepted
                )
                raw_increments = torch.where(
                    accept[:, None],
                    candidate,
                    raw_increments,
                )
                accepted_step = torch.where(accept, step, accepted_step)
                accepted = accepted | accept
                step = torch.where(accepted, step, step * 0.5)
                if bool(torch.all(accepted)):
                    break
            if not bool(torch.all(accepted)):
                raise RuntimeError("line search failed")

            if (iteration + 1) % 10 == 0:
                changes = torch.linalg.vector_norm(
                    accepted_step[:, None] * direction,
                    dim=1,
                )
                sizes = torch.linalg.vector_norm(raw_increments, dim=1)
                if bool(torch.all(changes <= tolerance * (1.0 + sizes))):
                    break

        coefficients = _coefficients(raw_increments).detach()
        _, unpenalized_hessian = _objective_derivatives(
            raw_increments,
            transformed_gram,
            cumulative_derivative_basis,
            1,
        )
        _, penalized_hessian = _objective_derivatives(
            raw_increments,
            transformed_gram + transformed_smoothing,
            cumulative_derivative_basis,
            1,
        )
        ridge = 1e-8 * torch.maximum(
            torch.ones(dimension_count, device=samples.device),
            torch.diagonal(
                penalized_hessian,
                dim1=1,
                dim2=2,
            ).abs().mean(dim=1),
        )
        effective_dof = torch.diagonal(
            torch.linalg.solve(
                penalized_hessian + ridge[:, None, None] * identity,
                unpenalized_hessian,
            ),
            dim1=1,
            dim2=2,
        ).sum(dim=1)
        mapped = torch.bmm(
            design_by_dimension,
            coefficients.unsqueeze(2),
        ).squeeze(2)
        derivatives = torch.bmm(
            derivative_by_dimension,
            coefficients.unsqueeze(2),
        ).squeeze(2)
        nll = (
            0.5 * mapped.square().sum(dim=1)
            - torch.log(derivatives).sum(dim=1)
        )
        correction = effective_dof * (effective_dof + 1.0) / torch.clamp(
            sample_count - effective_dof - 1.0,
            min=1e-12,
        )
        left_design, left_derivative_design = _bounded_basis(
            left.unsqueeze(0),
            knots,
        )
        right_design, right_derivative_design = _bounded_basis(
            right.unsqueeze(0),
            knots,
        )

        self.n_samples_seen_ = sample_count
        self.n_features_in_ = dimension_count
        self.mean_ = mean
        self.scale_ = scale
        self.knots_ = knots
        self.raw_coefficients_ = raw_increments.detach()
        self.effective_dof_ = effective_dof.detach()
        self.aicc_ = (2.0 * (nll + effective_dof + correction)).detach()
        self.nll_ = nll.detach()
        self.left_values_ = torch.sum(
            left_design[0] * coefficients,
            dim=1,
        )
        self.right_values_ = torch.sum(
            right_design[0] * coefficients,
            dim=1,
        )
        self.left_derivatives_ = torch.sum(
            left_derivative_design[0] * coefficients,
            dim=1,
        )
        self.right_derivatives_ = torch.sum(
            right_derivative_design[0] * coefficients,
            dim=1,
        )
        return self

    def _evaluate(
        self,
        x: Tensor,
        coefficients: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert (
            self.knots_ is not None
            and self.raw_coefficients_ is not None
            and self.left_values_ is not None
            and self.right_values_ is not None
            and self.left_derivatives_ is not None
            and self.right_derivatives_ is not None
        )
        left = self.knots_[:, _DEGREE]
        right = self.knots_[:, -_DEGREE - 1]
        bounded = torch.maximum(
            torch.minimum(x, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        if coefficients is None:
            coefficients = _coefficients(self.raw_coefficients_)
        values, derivatives = _bounded_evaluate(
            bounded,
            self.knots_,
            coefficients,
        )

        below = x < left.unsqueeze(0)
        above = x > right.unsqueeze(0)
        values = torch.where(
            below,
            self.left_values_.unsqueeze(0)
            + (x - left.unsqueeze(0)) * self.left_derivatives_.unsqueeze(0),
            values,
        )
        values = torch.where(
            above,
            self.right_values_.unsqueeze(0)
            + (x - right.unsqueeze(0)) * self.right_derivatives_.unsqueeze(0),
            values,
        )
        derivatives = torch.where(
            below,
            self.left_derivatives_.unsqueeze(0),
            derivatives,
        )
        derivatives = torch.where(
            above,
            self.right_derivatives_.unsqueeze(0),
            derivatives,
        )
        return values, derivatives

    def forward(self, X: Tensor) -> Tensor:
        self._validate_input(X, "X")
        assert self.mean_ is not None and self.scale_ is not None
        return self._evaluate((X - self.mean_) / self.scale_)[0]

    def inverse(self, Z: Tensor) -> Tensor:
        self._validate_input(Z, "Z")
        assert (
            self.knots_ is not None
            and self.mean_ is not None
            and self.scale_ is not None
            and self.left_values_ is not None
            and self.right_values_ is not None
            and self.left_derivatives_ is not None
            and self.right_derivatives_ is not None
        )
        left = self.knots_[:, _DEGREE]
        right = self.knots_[:, -_DEGREE - 1]
        epsilon = 100.0 * torch.finfo(Z.dtype).eps
        if torch.any(self.left_derivatives_ <= epsilon) or torch.any(
            self.right_derivatives_ <= epsilon
        ):
            raise RuntimeError("the fitted map has a non-positive tail slope")

        below = Z < self.left_values_.unsqueeze(0)
        above = Z > self.right_values_.unsqueeze(0)
        low = left.unsqueeze(0).expand_as(Z)
        high = right.unsqueeze(0).expand_as(Z)
        middle = left.unsqueeze(0) + (right - left).unsqueeze(0) * (
            (Z - self.left_values_.unsqueeze(0))
            / (self.right_values_ - self.left_values_).unsqueeze(0)
        )
        middle = torch.maximum(
            torch.minimum(middle, right.unsqueeze(0)),
            left.unsqueeze(0),
        )

        assert self.raw_coefficients_ is not None
        coefficients = _coefficients(self.raw_coefficients_)
        iterations = 24 if Z.dtype == torch.float32 else 40
        for _ in range(min(_INVERSE_ITERATIONS, iterations)):
            values, slopes = _bounded_evaluate(
                middle,
                self.knots_,
                coefficients,
            )
            low = torch.where(values < Z, middle, low)
            high = torch.where(values >= Z, middle, high)
            root_tolerance = (
                4.0
                * torch.finfo(Z.dtype).eps
                * (1.0 + middle.abs())
                * slopes.abs()
            )
            converged = below | above | (
                (values - Z).abs() <= root_tolerance
            )
            if bool(torch.all(converged)):
                break
            newton = middle - (values - Z) / slopes
            valid = (
                torch.isfinite(newton)
                & (slopes > epsilon)
                & (newton >= low)
                & (newton <= high)
            )
            middle = torch.where(valid, newton, 0.5 * (low + high))

        left_tail = left.unsqueeze(0) + (
            Z - self.left_values_.unsqueeze(0)
        ) / self.left_derivatives_.unsqueeze(0)
        right_tail = right.unsqueeze(0) + (
            Z - self.right_values_.unsqueeze(0)
        ) / self.right_derivatives_.unsqueeze(0)
        result = torch.where(below, left_tail, middle)
        result = torch.where(above, right_tail, result)

        root = result.detach()
        root_value, root_slope = self._evaluate(root, coefficients)
        standardized = root + (Z - root_value) / root_slope.detach()
        return standardized * self.scale_ + self.mean_

    @property
    def coefficients_(self) -> Tensor:
        self._require_fitted()
        assert self.raw_coefficients_ is not None
        return self.raw_coefficients_
