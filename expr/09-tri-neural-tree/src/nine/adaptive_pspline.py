from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Sparsity = Tensor | Sequence[Sequence[int]] | Literal["diagonal"] | None


@dataclass(frozen=True)
class _LocalBasisEvaluation:
    x: Tensor
    knot_indices: Tensor
    values: Tensor
    below: Tensor
    above: Tensor


class _LinearTailBasis(nn.Module):
    def __init__(self, degree: int, knots: Tensor) -> None:
        super().__init__()
        self.degree = degree
        self.register_buffer("knots", knots)
        self.nbasis = self.knots.numel() - self.degree - 1
        self.register_buffer("left", self.knots[self.degree].clone())
        self.register_buffer("right", self.knots[-self.degree - 1].clone())
        self.width = float((self.right - self.left).detach().cpu())
        self.register_buffer(
            "_levels",
            torch.arange(1, self.degree + 1, device=self.knots.device),
            persistent=False,
        )
        self.register_buffer(
            "_offsets",
            torch.arange(-self.degree, 1, device=self.knots.device),
            persistent=False,
        )
        edges = torch.stack((self.left, self.right))
        edge_values, edge_derivatives = self._bounded_design_and_derivative(edges)
        self.register_buffer("left_values", edge_values[0])
        self.register_buffer("right_values", edge_values[1])
        self.register_buffer("left_derivatives", edge_derivatives[0])
        self.register_buffer("right_derivatives", edge_derivatives[1])

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
        legacy_key = f"{prefix}basis.knots"
        if legacy_key in state_dict:
            knots_key = f"{prefix}knots"
            if knots_key in state_dict and not torch.equal(
                state_dict[legacy_key],
                state_dict[knots_key],
            ):
                error_msgs.append(
                    f'inconsistent spline knots in "{knots_key}" and '
                    f'legacy key "{legacy_key}"'
                )
            if isinstance(state_dict, MutableMapping):
                state_dict.pop(legacy_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _bounded_design(self, x: Tensor) -> Tensor:
        knot_indices, local_values, _ = self._bounded_local_design(
            x,
            derivative=False,
        )
        values = x.new_zeros((x.numel(), self.nbasis))
        values.scatter_(1, knot_indices, local_values)
        return values

    def _bounded_derivative_design(self, x: Tensor) -> Tensor:
        knot_indices, _, local_derivatives = self._bounded_local_design(
            x,
            derivative=True,
        )
        assert local_derivatives is not None
        derivatives = x.new_zeros((x.numel(), self.nbasis))
        derivatives.scatter_(1, knot_indices, local_derivatives)
        return derivatives

    def _bounded_local_design(
        self,
        x: Tensor,
        *,
        derivative: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        values = x.reshape(-1)
        spans = torch.searchsorted(self.knots, values, side="right") - 1
        spans = torch.where(values <= self.left, self.degree, spans)
        spans = torch.where(values >= self.right, self.nbasis - 1, spans)
        spans = spans.clamp(min=self.degree, max=self.nbasis - 1)

        left = values.unsqueeze(1) - self.knots[
            spans.unsqueeze(1) + 1 - self._levels
        ]
        right = (
            self.knots[spans.unsqueeze(1) + self._levels] - values.unsqueeze(1)
        )
        local_basis = torch.ones_like(values).unsqueeze(1)
        lower_basis = local_basis if derivative else None

        for level in range(1, self.degree + 1):
            saved = torch.zeros_like(values)
            next_basis = []
            for index in range(level):
                denominator = right[:, index] + left[:, level - index - 1]
                ratio = torch.where(
                    denominator != 0,
                    local_basis[:, index] / denominator,
                    torch.zeros_like(denominator),
                )
                next_basis.append(saved + right[:, index] * ratio)
                saved = left[:, level - index - 1] * ratio
            next_basis.append(saved)
            local_basis = torch.stack(next_basis, dim=1)
            if derivative and level == self.degree - 1:
                lower_basis = local_basis

        knot_indices = spans.unsqueeze(1) + self._offsets
        if not derivative:
            return knot_indices, local_basis, None

        assert lower_basis is not None
        alpha_denominator = (
            self.knots[knot_indices + self.degree] - self.knots[knot_indices]
        )
        beta_denominator = (
            self.knots[knot_indices + self.degree + 1]
            - self.knots[knot_indices + 1]
        )
        alpha = torch.where(
            alpha_denominator != 0,
            self.degree / alpha_denominator,
            torch.zeros_like(alpha_denominator),
        )
        beta = torch.where(
            beta_denominator != 0,
            self.degree / beta_denominator,
            torch.zeros_like(beta_denominator),
        )
        lower_left = F.pad(lower_basis, (1, 0))
        lower_right = F.pad(lower_basis, (0, 1))
        local_derivatives = alpha * lower_left - beta * lower_right
        return knot_indices, local_basis, local_derivatives

    def _bounded_design_and_derivative(self, x: Tensor) -> tuple[Tensor, Tensor]:
        knot_indices, local_values, local_derivatives = self._bounded_local_design(
            x,
            derivative=True,
        )
        assert local_derivatives is not None
        values = x.new_zeros((x.numel(), self.nbasis))
        derivatives = torch.zeros_like(values)
        values.scatter_(1, knot_indices, local_values)
        derivatives.scatter_(1, knot_indices, local_derivatives)
        return values, derivatives

    def design(self, x: Tensor) -> Tensor:
        bounded = x.clamp(min=self.left, max=self.right)
        values = self._bounded_design(bounded)

        below = x < self.left
        above = x > self.right
        low_tail = (
            self.left_values + (x - self.left).unsqueeze(1) * self.left_derivatives
        )
        high_tail = (
            self.right_values + (x - self.right).unsqueeze(1) * self.right_derivatives
        )
        values = torch.where(below.unsqueeze(1), low_tail, values)
        return torch.where(above.unsqueeze(1), high_tail, values)

    def design_and_derivative(self, x: Tensor) -> tuple[Tensor, Tensor]:
        bounded = x.clamp(min=self.left, max=self.right)
        values, derivatives = self._bounded_design_and_derivative(bounded)

        below = x < self.left
        above = x > self.right
        low_tail = (
            self.left_values + (x - self.left).unsqueeze(1) * self.left_derivatives
        )
        high_tail = (
            self.right_values + (x - self.right).unsqueeze(1) * self.right_derivatives
        )
        values = torch.where(below.unsqueeze(1), low_tail, values)
        values = torch.where(above.unsqueeze(1), high_tail, values)
        derivatives = torch.where(
            below.unsqueeze(1),
            self.left_derivatives.expand_as(derivatives),
            derivatives,
        )
        derivatives = torch.where(
            above.unsqueeze(1),
            self.right_derivatives.expand_as(derivatives),
            derivatives,
        )
        return values, derivatives

    def derivative_design(self, x: Tensor) -> Tensor:
        bounded = x.clamp(min=self.left, max=self.right)
        derivatives = self._bounded_derivative_design(bounded)
        derivatives = torch.where(
            (x < self.left).unsqueeze(1),
            self.left_derivatives.expand_as(derivatives),
            derivatives,
        )
        return torch.where(
            (x > self.right).unsqueeze(1),
            self.right_derivatives.expand_as(derivatives),
            derivatives,
        )

    def evaluate_and_derivative(
        self,
        x: Tensor,
        coefficients: Tensor,
    ) -> tuple[Tensor, Tensor]:
        bounded = x.clamp(min=self.left, max=self.right)
        values, derivatives = self._bounded_evaluate_and_derivative(
            bounded,
            coefficients,
        )

        below = x < self.left
        above = x > self.right
        left_value = self.left_values @ coefficients
        right_value = self.right_values @ coefficients
        left_derivative = self.left_derivatives @ coefficients
        right_derivative = self.right_derivatives @ coefficients
        values = torch.where(
            below,
            left_value + (x - self.left) * left_derivative,
            values,
        )
        values = torch.where(
            above,
            right_value + (x - self.right) * right_derivative,
            values,
        )
        derivatives = torch.where(below, left_derivative, derivatives)
        derivatives = torch.where(above, right_derivative, derivatives)
        return values, derivatives

    def _bounded_evaluate_and_derivative(
        self,
        x: Tensor,
        coefficients: Tensor,
    ) -> tuple[Tensor, Tensor]:
        knot_indices, local_values, local_derivatives = self._bounded_local_design(
            x,
            derivative=True,
        )
        assert local_derivatives is not None
        local_coefficients = coefficients[knot_indices]
        values = torch.sum(local_values * local_coefficients, dim=1)
        derivatives = torch.sum(local_derivatives * local_coefficients, dim=1)
        return values, derivatives

    def evaluate(self, x: Tensor, coefficients: Tensor) -> Tensor:
        return self.evaluate_local(self.local_design(x), coefficients)

    def prepare_evaluation(self, x: Tensor) -> Tensor | _LocalBasisEvaluation:
        if self.nbasis <= 4 * (self.degree + 1):
            return self.design(x)
        return self.local_design(x)

    def evaluate_prepared(
        self,
        evaluation: Tensor | _LocalBasisEvaluation,
        coefficients: Tensor,
    ) -> Tensor:
        if isinstance(evaluation, Tensor):
            return evaluation @ coefficients
        return self.evaluate_local(evaluation, coefficients)

    def local_design(self, x: Tensor) -> _LocalBasisEvaluation:
        bounded = x.clamp(min=self.left, max=self.right)
        knot_indices, local_values, _ = self._bounded_local_design(
            bounded,
            derivative=False,
        )
        return _LocalBasisEvaluation(
            x=x,
            knot_indices=knot_indices,
            values=local_values,
            below=x < self.left,
            above=x > self.right,
        )

    def evaluate_local(
        self,
        local_design: _LocalBasisEvaluation,
        coefficients: Tensor,
    ) -> Tensor:
        values = torch.sum(
            local_design.values * coefficients[local_design.knot_indices],
            dim=1,
        )
        left_value = self.left_values @ coefficients
        right_value = self.right_values @ coefficients
        left_derivative = self.left_derivatives @ coefficients
        right_derivative = self.right_derivatives @ coefficients
        values = torch.where(
            local_design.below,
            left_value + (local_design.x - self.left) * left_derivative,
            values,
        )
        values = torch.where(
            local_design.above,
            right_value + (local_design.x - self.right) * right_derivative,
            values,
        )
        return values


@dataclass(frozen=True)
class _ComponentProblem:
    dimension: int
    parents: tuple[int, ...]
    basis_blocks: tuple[Tensor, ...]
    design: Tensor
    gram: Tensor
    nonmonotone_design: Tensor | None
    derivative_basis: Tensor
    penalties: tuple[Tensor, ...]
    block_scales: tuple[Tensor, ...]
    block_sizes: tuple[int, ...]
    monotone_cumulative: Tensor
    monotone_identity: Tensor


@dataclass(frozen=True)
class _ReducedProblem:
    design: Tensor
    derivative_basis: Tensor
    penalty: Tensor
    quadratic: Tensor
    dependency: Tensor | None
    monotone_cumulative: Tensor
    inverse_sample_size: float


@dataclass(frozen=True)
class _ComponentFit:
    log_lambdas: Tensor
    raw_coefficients: Tensor
    effective_dof: Tensor
    aicc: Tensor
    nll: Tensor


class _FittedComponent(nn.Module):
    def __init__(
        self,
        *,
        dimension: int,
        parents: tuple[int, ...],
        block_sizes: tuple[int, ...],
        log_lambdas: Tensor,
        coefficients: Tensor,
        effective_dof: Tensor,
        aicc: Tensor,
        nll: Tensor,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.parents = parents
        self.block_sizes = block_sizes
        self.register_buffer("log_lambdas", log_lambdas)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("effective_dof", effective_dof)
        self.register_buffer("aicc", aicc)
        self.register_buffer("nll", nll)


class _BatchedDiagonal(nn.Module):
    """Fitted independent splines sharing one tensor layout."""

    def __init__(
        self,
        *,
        knots: Tensor,
        coefficients: Tensor,
        log_lambdas: Tensor,
        effective_dof: Tensor,
        aicc: Tensor,
        nll: Tensor,
        left_values: Tensor,
        right_values: Tensor,
        left_derivatives: Tensor,
        right_derivatives: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("knots", knots)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("log_lambdas", log_lambdas)
        self.register_buffer("effective_dof", effective_dof)
        self.register_buffer("aicc", aicc)
        self.register_buffer("nll", nll)
        self.register_buffer("left_values", left_values)
        self.register_buffer("right_values", right_values)
        self.register_buffer("left_derivatives", left_derivatives)
        self.register_buffer("right_derivatives", right_derivatives)


def _batched_bounded_design_and_derivative(
    x: Tensor,
    knots: Tensor,
    degree: int,
) -> tuple[Tensor, Tensor]:
    """Evaluate all diagonal B-spline bases without materializing modules."""
    if x.ndim != 2:
        raise ValueError("batched spline inputs must have shape (N, D)")
    knot_count = knots.shape[1]
    nbasis = knot_count - degree - 1
    values = (
        (x.unsqueeze(-1) >= knots[:, :-1].unsqueeze(0))
        & (x.unsqueeze(-1) < knots[:, 1:].unsqueeze(0))
    ).to(dtype=x.dtype)
    at_right = x >= knots[:, nbasis].unsqueeze(0)
    values[..., -1] = torch.where(
        at_right,
        torch.ones_like(x),
        values[..., -1],
    )
    lower_values: Tensor | None = None
    for level in range(1, degree + 1):
        size = knot_count - level - 1
        left_denominator = (
            knots[:, level : level + size] - knots[:, :size]
        ).unsqueeze(0)
        right_denominator = (
            knots[:, level + 1 : level + 1 + size]
            - knots[:, 1 : 1 + size]
        ).unsqueeze(0)
        left = torch.where(
            left_denominator != 0,
            (x.unsqueeze(-1) - knots[:, :size].unsqueeze(0))
            / torch.where(
                left_denominator != 0,
                left_denominator,
                torch.ones_like(left_denominator),
            )
            * values[..., :size],
            torch.zeros_like(values[..., :size]),
        )
        right = torch.where(
            right_denominator != 0,
            (knots[:, level + 1 : level + 1 + size].unsqueeze(0)
            - x.unsqueeze(-1))
            / torch.where(
                right_denominator != 0,
                right_denominator,
                torch.ones_like(right_denominator),
            )
            * values[..., 1 : size + 1],
            torch.zeros_like(values[..., :size]),
        )
        values = left + right
        if level == degree - 1:
            lower_values = values

    assert lower_values is not None
    left_denominator = knots[:, degree : degree + nbasis] - knots[:, :nbasis]
    right_denominator = (
        knots[:, degree + 1 : degree + 1 + nbasis]
        - knots[:, 1 : 1 + nbasis]
    )
    left_scale = torch.where(
        left_denominator != 0,
        degree / left_denominator,
        torch.zeros_like(left_denominator),
    )
    right_scale = torch.where(
        right_denominator != 0,
        degree / right_denominator,
        torch.zeros_like(right_denominator),
    )
    derivatives = (
        left_scale.unsqueeze(0) * lower_values[..., :nbasis]
        - right_scale.unsqueeze(0) * lower_values[..., 1 : nbasis + 1]
    )
    right_values = torch.zeros_like(values)
    right_values[..., -1] = 1.0
    right_derivatives = torch.zeros_like(derivatives)
    final_slope = degree / (knots[:, nbasis] - knots[:, nbasis - 1])
    right_derivatives[..., -2] = -final_slope.unsqueeze(0)
    right_derivatives[..., -1] = final_slope.unsqueeze(0)
    values = torch.where(at_right.unsqueeze(-1), right_values, values)
    derivatives = torch.where(
        at_right.unsqueeze(-1),
        right_derivatives,
        derivatives,
    )
    return values, derivatives


def _batched_diagonal_evaluate(
    x: Tensor,
    diagonal: _BatchedDiagonal,
    degree: int,
) -> tuple[Tensor, Tensor]:
    left = diagonal.knots[:, degree]
    right = diagonal.knots[:, -degree - 1]
    bounded = torch.maximum(
        torch.minimum(x, right.unsqueeze(0)),
        left.unsqueeze(0),
    )
    design, derivative_design = _batched_bounded_design_and_derivative(
        bounded,
        diagonal.knots,
        degree,
    )
    coefficients = _batched_reparameterize(diagonal.coefficients)
    values = torch.sum(design * coefficients.unsqueeze(0), dim=-1)
    derivatives = torch.sum(
        derivative_design * coefficients.unsqueeze(0),
        dim=-1,
    )
    below = x < left.unsqueeze(0)
    above = x > right.unsqueeze(0)
    values = torch.where(
        below,
        diagonal.left_values.unsqueeze(0)
        + (x - left.unsqueeze(0)) * diagonal.left_derivatives.unsqueeze(0),
        values,
    )
    values = torch.where(
        above,
        diagonal.right_values.unsqueeze(0)
        + (x - right.unsqueeze(0)) * diagonal.right_derivatives.unsqueeze(0),
        values,
    )
    derivatives = torch.where(
        below,
        diagonal.left_derivatives.unsqueeze(0),
        derivatives,
    )
    derivatives = torch.where(
        above,
        diagonal.right_derivatives.unsqueeze(0),
        derivatives,
    )
    return values, derivatives


def _reparameterize(raw_monotone_increments: Tensor) -> Tensor:
    if raw_monotone_increments.ndim != 1:
        raise ValueError("monotone increments must be one-dimensional")
    positive_increments = torch.cat(
        (
            raw_monotone_increments[:1],
            F.softplus(raw_monotone_increments[1:]),
        )
    )
    return torch.cumsum(positive_increments, dim=0)


def _smoothing_matrix(nbasis: int, reference: Tensor) -> Tensor:
    identity = torch.eye(
        nbasis,
        dtype=reference.dtype,
        device=reference.device,
    )
    differences = torch.diff(identity, n=2, dim=1)
    return differences @ differences.T


def _block_scale(design: Tensor) -> Tensor:
    return torch.sqrt(torch.sum(design.square()) / design.shape[0])


def _scaled_penalties(
    problem: _ComponentProblem,
    log_lambdas: Tensor,
) -> tuple[Tensor, ...]:
    if log_lambdas.shape != (len(problem.basis_blocks),):
        raise ValueError("one log smoothing parameter is required per spline block")
    return tuple(
        torch.exp(log_lambdas[index]) * scale.square() * penalty
        for index, (scale, penalty) in enumerate(
            zip(problem.block_scales, problem.penalties, strict=True)
        )
    )


def _regularized_normal_matrix(gram: Tensor, penalty: Tensor) -> Tensor:
    normal_matrix = gram + penalty
    ridge = 1e-6 * torch.maximum(
        normal_matrix.new_ones(()),
        torch.diagonal(normal_matrix).mean(),
    )
    return normal_matrix + ridge * torch.eye(
        normal_matrix.shape[0],
        dtype=normal_matrix.dtype,
        device=normal_matrix.device,
    )


def _prepare_reduced_problem(
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> _ReducedProblem:
    monotone_design = problem.basis_blocks[-1]
    scaled_penalties = _scaled_penalties(problem, log_lambdas)
    monotone_penalty = scaled_penalties[-1]

    if len(problem.basis_blocks) == 1:
        reduced_design = monotone_design
        combined_penalty = monotone_penalty
        dependency = None
    else:
        assert problem.nonmonotone_design is not None
        nonmonotone_design = problem.nonmonotone_design
        nonmonotone_penalty = torch.block_diag(*scaled_penalties[:-1])
        nonmonotone_size = nonmonotone_design.shape[1]
        normal_matrix = _regularized_normal_matrix(
            problem.gram[:nonmonotone_size, :nonmonotone_size],
            nonmonotone_penalty,
        )
        projection = torch.linalg.solve(normal_matrix, nonmonotone_design.T)
        dependency = projection @ monotone_design
        reduced_design = monotone_design - nonmonotone_design @ dependency
        combined_penalty = (
            dependency.T @ nonmonotone_penalty @ dependency + monotone_penalty
        )

    return _ReducedProblem(
        design=reduced_design,
        derivative_basis=problem.derivative_basis,
        penalty=combined_penalty,
        quadratic=(
            reduced_design.T @ reduced_design + combined_penalty
        ) / reduced_design.shape[0],
        dependency=dependency,
        monotone_cumulative=problem.monotone_cumulative,
        inverse_sample_size=1.0 / reduced_design.shape[0],
    )


def _prepared_reduced_objective(
    raw_monotone_increments: Tensor,
    problem: _ReducedProblem,
) -> Tensor:
    monotone_coefficients = _reparameterize(raw_monotone_increments)
    derivative = problem.derivative_basis @ monotone_coefficients
    if torch.any(derivative <= 0):
        return raw_monotone_increments.new_full((), torch.inf)

    return (
        0.5
        * monotone_coefficients
        @ problem.quadratic
        @ monotone_coefficients
        - problem.inverse_sample_size * torch.sum(torch.log(derivative))
    )


def _reparameterization_derivatives(
    raw_monotone_increments: Tensor,
) -> tuple[Tensor, Tensor]:
    sigmoid = torch.sigmoid(raw_monotone_increments[1:])
    first = raw_monotone_increments.new_ones((1,))
    slopes = torch.cat((first, sigmoid))
    curvatures = torch.cat((torch.zeros_like(first), sigmoid * (1.0 - sigmoid)))
    return slopes, curvatures


def _prepared_reduced_derivatives(
    raw_monotone_increments: Tensor,
    problem: _ReducedProblem,
) -> tuple[Tensor, Tensor, Tensor]:
    monotone_coefficients = _reparameterize(raw_monotone_increments)
    derivative = problem.derivative_basis @ monotone_coefficients
    if torch.any(derivative <= 0):
        invalid = raw_monotone_increments.new_full((), torch.inf)
        return (
            invalid,
            torch.full_like(raw_monotone_increments, torch.nan),
            raw_monotone_increments.new_full(
                (raw_monotone_increments.numel(),) * 2,
                torch.nan,
            ),
        )

    reciprocal = derivative.reciprocal()
    coefficient_gradient = (
        problem.quadratic @ monotone_coefficients
        - problem.inverse_sample_size
        * (problem.derivative_basis.T @ reciprocal)
    )
    coefficient_hessian = problem.quadratic + (
        problem.inverse_sample_size
        * problem.derivative_basis.T
        * reciprocal.square()
    ) @ problem.derivative_basis
    slopes, curvatures = _reparameterization_derivatives(
        raw_monotone_increments
    )
    jacobian = problem.monotone_cumulative * slopes.unsqueeze(0)
    reverse_gradient = torch.flip(
        torch.cumsum(torch.flip(coefficient_gradient, dims=(0,)), dim=0),
        dims=(0,),
    )
    gradient = slopes * reverse_gradient
    objective_hessian = (
        jacobian.T @ coefficient_hessian @ jacobian
        + torch.diag(curvatures * reverse_gradient)
    )
    objective = (
        0.5
        * monotone_coefficients
        @ problem.quadratic
        @ monotone_coefficients
        - problem.inverse_sample_size * torch.sum(torch.log(derivative))
    )
    return objective, gradient, objective_hessian


def _prepared_reduced_gradient(
    raw_monotone_increments: Tensor,
    problem: _ReducedProblem,
) -> Tensor:
    monotone_coefficients = _reparameterize(raw_monotone_increments)
    derivative = problem.derivative_basis @ monotone_coefficients
    if torch.any(derivative <= 0):
        return torch.full_like(raw_monotone_increments, torch.nan)

    coefficient_gradient = (
        problem.quadratic @ monotone_coefficients
        - problem.inverse_sample_size
        * (problem.derivative_basis.T @ derivative.reciprocal())
    )
    slopes, _ = _reparameterization_derivatives(raw_monotone_increments)
    reverse_gradient = torch.flip(
        torch.cumsum(torch.flip(coefficient_gradient, dims=(0,)), dim=0),
        dims=(0,),
    )
    return slopes * reverse_gradient


def _solve_monotone(
    initial_raw_increments: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
    *,
    max_iter: int,
    reduced_problem: _ReducedProblem | None = None,
) -> Tensor:
    parameters = log_lambdas.detach()
    raw_increments = initial_raw_increments.detach().clone()
    gradient_tolerance = max(
        1e-8,
        10.0 * torch.finfo(raw_increments.dtype).eps ** 0.5,
    )
    identity = problem.monotone_identity
    if reduced_problem is None:
        reduced_problem = _prepare_reduced_problem(parameters, problem)

    for _ in range(max_iter):
        objective, objective_gradient, objective_hessian = (
            _prepared_reduced_derivatives(raw_increments, reduced_problem)
        )

        if not torch.isfinite(objective) or not torch.all(
            torch.isfinite(objective_gradient)
        ):
            raise RuntimeError(
                f"inner optimization for component {problem.dimension} "
                "produced non-finite values"
            )

        gradient_norm = torch.linalg.vector_norm(
            objective_gradient,
            ord=float("inf"),
        )
        if gradient_norm <= gradient_tolerance:
            return raw_increments

        curvature_scale = torch.maximum(
            raw_increments.new_ones(()),
            torch.diagonal(objective_hessian).abs().mean(),
        )
        damping = 1e-8 * curvature_scale
        direction = None
        for _ in range(12):
            candidate_direction = torch.linalg.solve(
                objective_hessian + damping * identity,
                -objective_gradient,
            )
            directional_derivative = objective_gradient @ candidate_direction
            if (
                torch.all(torch.isfinite(candidate_direction))
                and directional_derivative < 0
            ):
                direction = candidate_direction.detach()
                break
            damping = damping * 10.0
        if direction is None:
            raise RuntimeError(
                f"could not find a descent direction for component {problem.dimension}"
            )

        step = 1.0
        initial_objective = objective.detach()
        slope = (objective_gradient.detach() @ direction).item()
        objective_tolerance = (
            10.0
            * torch.finfo(raw_increments.dtype).eps
            * torch.maximum(
                raw_increments.new_ones(()),
                initial_objective.abs(),
            )
        )
        accepted = False
        for _ in range(25):
            candidate = raw_increments + step * direction
            candidate_objective = _prepared_reduced_objective(
                candidate,
                reduced_problem,
            )
            if torch.isfinite(candidate_objective) and (
                candidate_objective
                <= initial_objective + 1e-4 * step * slope + objective_tolerance
            ):
                raw_increments = candidate.detach()
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError(f"line search failed for component {problem.dimension}")
        if torch.linalg.vector_norm(step * direction) <= gradient_tolerance * (
            1.0 + torch.linalg.vector_norm(raw_increments)
        ):
            break

    _, final_gradient, _ = _prepared_reduced_derivatives(
        raw_increments,
        reduced_problem,
    )
    gradient_norm = torch.linalg.vector_norm(final_gradient, ord=float("inf"))
    if gradient_norm > 20.0 * gradient_tolerance:
        raise RuntimeError(
            f"inner optimization for component {problem.dimension} did not "
            f"converge; gradient norm={gradient_norm.item():.3e}"
        )
    return raw_increments


def _make_implicit_solver(
    problem: _ComponentProblem,
    *,
    max_iter: int,
):
    warm_start: list[Tensor | None] = [None]

    class _ImplicitSolve(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            initial_raw_increments: Tensor,
            log_lambdas: Tensor,
        ) -> Tensor:
            initial = warm_start[0]
            if initial is None:
                initial = initial_raw_increments
            reduced_problem = _prepare_reduced_problem(
                log_lambdas.detach(),
                problem,
            )
            try:
                solution = _solve_monotone(
                    initial,
                    log_lambdas,
                    problem,
                    max_iter=max_iter,
                    reduced_problem=reduced_problem,
                )
            except RuntimeError:
                if warm_start[0] is None:
                    raise
                solution = _solve_monotone(
                    initial_raw_increments,
                    log_lambdas,
                    problem,
                    max_iter=max_iter,
                    reduced_problem=reduced_problem,
                )
            warm_start[0] = solution.detach()

            _, _, objective_hessian = _prepared_reduced_derivatives(
                solution,
                reduced_problem,
            )
            ctx.save_for_backward(
                solution.detach(),
                log_lambdas.detach(),
                objective_hessian.detach(),
            )
            return solution

        @staticmethod
        def backward(
            ctx: Any,
            output_gradient: Tensor,
        ) -> tuple[None, Tensor]:
            solution, log_lambdas, objective_hessian = ctx.saved_tensors
            adjoint = torch.linalg.solve(
                objective_hessian.T,
                output_gradient,
            )

            with torch.enable_grad():
                differentiable_logs = log_lambdas.detach().requires_grad_(True)
                reduced_problem = _prepare_reduced_problem(
                    differentiable_logs,
                    problem,
                )
                optimality = _prepared_reduced_gradient(
                    solution,
                    reduced_problem,
                )
                (log_gradient,) = torch.autograd.grad(
                    optimality,
                    differentiable_logs,
                    grad_outputs=-adjoint,
                )
            return None, log_gradient

    return _ImplicitSolve.apply, warm_start


def _assemble_coefficients(
    raw_monotone_increments: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
    reduced_problem: _ReducedProblem | None = None,
) -> Tensor:
    if len(problem.basis_blocks) == 1:
        return raw_monotone_increments

    if reduced_problem is None:
        reduced_problem = _prepare_reduced_problem(log_lambdas, problem)
    assert reduced_problem.dependency is not None
    nonmonotone_coefficients = (
        -reduced_problem.dependency
        @ _reparameterize(raw_monotone_increments)
    )
    return torch.cat((nonmonotone_coefficients, raw_monotone_increments))


def _materialize_coefficients(
    raw_coefficients: Tensor,
    monotone_size: int,
) -> Tensor:
    return torch.cat(
        (
            raw_coefficients[:-monotone_size],
            _reparameterize(raw_coefficients[-monotone_size:]),
        )
    )


def _nll(
    raw_coefficients: Tensor,
    problem: _ComponentProblem,
) -> Tensor:
    monotone_size = problem.block_sizes[-1]
    coefficients = _materialize_coefficients(
        raw_coefficients,
        monotone_size,
    )
    mapped = problem.design @ coefficients
    derivative = problem.derivative_basis @ coefficients[-monotone_size:]
    return 0.5 * torch.sum(mapped.square()) - torch.sum(torch.log(derivative))


def _penalized_nll(
    raw_coefficients: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> Tensor:
    monotone_size = problem.block_sizes[-1]
    nonmonotone_coefficients = raw_coefficients[:-monotone_size]
    monotone_coefficients = _reparameterize(raw_coefficients[-monotone_size:])
    penalties = _scaled_penalties(problem, log_lambdas)

    penalty = 0.5 * monotone_coefficients @ penalties[-1] @ monotone_coefficients
    if nonmonotone_coefficients.numel():
        nonmonotone_penalty = torch.block_diag(*penalties[:-1])
        penalty = (
            penalty
            + 0.5
            * nonmonotone_coefficients
            @ nonmonotone_penalty
            @ nonmonotone_coefficients
        )
    return _nll(raw_coefficients, problem) + penalty


def _raw_objective_hessian(
    raw_coefficients: Tensor,
    problem: _ComponentProblem,
    penalties: tuple[Tensor, ...] | None = None,
) -> Tensor:
    monotone_size = problem.block_sizes[-1]
    nonmonotone_size = raw_coefficients.numel() - monotone_size
    coefficients = _materialize_coefficients(
        raw_coefficients,
        monotone_size,
    )
    quadratic = problem.gram
    if penalties is not None:
        quadratic = quadratic + torch.block_diag(*penalties)

    derivative = (
        problem.derivative_basis @ coefficients[-monotone_size:]
    )
    reciprocal = derivative.reciprocal()
    coefficient_gradient = quadratic @ coefficients
    coefficient_gradient = torch.cat(
        (
            coefficient_gradient[:nonmonotone_size],
            coefficient_gradient[nonmonotone_size:]
            - problem.derivative_basis.T @ reciprocal,
        )
    )
    derivative_hessian = (
        problem.derivative_basis.T * reciprocal.square()
    ) @ problem.derivative_basis
    coefficient_hessian = quadratic + torch.block_diag(
        raw_coefficients.new_zeros(
            (nonmonotone_size, nonmonotone_size)
        ),
        derivative_hessian,
    )

    slopes, curvatures = _reparameterization_derivatives(
        raw_coefficients[-monotone_size:]
    )
    cumulative = torch.tril(
        torch.ones(
            monotone_size,
            monotone_size,
            dtype=raw_coefficients.dtype,
            device=raw_coefficients.device,
        )
    )
    monotone_jacobian = cumulative * slopes.unsqueeze(0)
    if nonmonotone_size:
        transformation = torch.block_diag(
            torch.eye(
                nonmonotone_size,
                dtype=raw_coefficients.dtype,
                device=raw_coefficients.device,
            ),
            monotone_jacobian,
        )
    else:
        transformation = monotone_jacobian

    monotone_gradient = coefficient_gradient[-monotone_size:]
    reverse_gradient = torch.flip(
        torch.cumsum(torch.flip(monotone_gradient, dims=(0,)), dim=0),
        dims=(0,),
    )
    curvature_correction = torch.cat(
        (
            raw_coefficients.new_zeros((nonmonotone_size,)),
            curvatures * reverse_gradient,
        )
    )
    return (
        transformation.T @ coefficient_hessian @ transformation
        + torch.diag(curvature_correction)
    )


def _information_criterion(
    raw_coefficients: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> tuple[Tensor, Tensor, Tensor]:
    unpenalized_hessian = _raw_objective_hessian(
        raw_coefficients,
        problem,
    )
    penalties = _scaled_penalties(problem, log_lambdas)
    penalized_hessian = _raw_objective_hessian(
        raw_coefficients,
        problem,
        penalties,
    )
    ridge = 1e-8 * torch.maximum(
        raw_coefficients.new_ones(()),
        torch.diagonal(penalized_hessian).abs().mean(),
    )
    regularized = penalized_hessian + ridge * torch.eye(
        raw_coefficients.numel(),
        dtype=raw_coefficients.dtype,
        device=raw_coefficients.device,
    )
    effective_dof = torch.trace(torch.linalg.solve(regularized, unpenalized_hessian))
    nll = _nll(raw_coefficients, problem)
    sample_size = problem.basis_blocks[0].shape[0]
    denominator = torch.clamp(
        sample_size - effective_dof - 1.0,
        min=1e-12,
    )
    correction = effective_dof * (effective_dof + 1.0) / denominator
    aicc_half = nll + effective_dof + correction
    return aicc_half, effective_dof, nll


def _invert_monotone(
    target: Tensor,
    basis: _LinearTailBasis,
    monotone_coefficients: Tensor,
    *,
    iterations: int,
) -> Tensor:
    left_value = basis.left_values @ monotone_coefficients
    right_value = basis.right_values @ monotone_coefficients
    left_slope = basis.left_derivatives @ monotone_coefficients
    right_slope = basis.right_derivatives @ monotone_coefficients
    epsilon = 100.0 * torch.finfo(target.dtype).eps
    if left_slope <= epsilon or right_slope <= epsilon:
        raise RuntimeError("the fitted map has a non-positive tail slope")

    below = target < left_value
    above = target > right_value
    low = torch.full_like(target, basis.left)
    high = torch.full_like(target, basis.right)
    root_tolerance = math.sqrt(torch.finfo(target.dtype).eps)
    required_iterations = max(
        1,
        math.ceil(math.log2(basis.width / root_tolerance)),
    )
    middle = basis.left + (basis.right - basis.left) * (
        (target - left_value) / (right_value - left_value)
    )
    middle = middle.clamp(min=basis.left, max=basis.right)
    for _ in range(min(iterations, required_iterations)):
        values, slopes = basis._bounded_evaluate_and_derivative(
            middle,
            monotone_coefficients,
        )
        residual = values - target
        newton_step = residual / slopes
        if target.device.type == "cpu" and torch.all(
            newton_step.abs()
            <= root_tolerance
            * torch.maximum(
                torch.ones_like(middle),
                middle.abs(),
            )
        ):
            break
        low = torch.where(values < target, middle, low)
        high = torch.where(values >= target, middle, high)
        newton = middle - newton_step
        valid_newton = (
            torch.isfinite(newton)
            & (slopes > epsilon)
            & (newton >= low)
            & (newton <= high)
        )
        middle = torch.where(valid_newton, newton, 0.5 * (low + high))

    left_tail = basis.left + (target - left_value) / left_slope
    right_tail = basis.right + (target - right_value) / right_slope
    result = torch.where(below, left_tail, middle)
    result = torch.where(above, right_tail, result)

    root = result.detach()
    root_value, root_slope = basis.evaluate_and_derivative(
        root,
        monotone_coefficients,
    )
    if torch.any(root_slope <= epsilon):
        raise RuntimeError("the fitted map is not strictly monotone")
    return root + (target - root_value) / root_slope.detach()


def _batched_reparameterize(raw_increments: Tensor) -> Tensor:
    return torch.cumsum(
        torch.cat(
            (
                raw_increments[:, :1],
                F.softplus(raw_increments[:, 1:]),
            ),
            dim=1,
        ),
        dim=1,
    )


def _batched_raw_derivatives(
    raw_increments: Tensor,
    quadratic: Tensor,
    derivative_basis: Tensor,
    cumulative: Tensor,
    *,
    sample_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    coefficients = _batched_reparameterize(raw_increments)
    derivative = torch.sum(
        derivative_basis * coefficients.unsqueeze(0),
        dim=2,
    )
    reciprocal = derivative.reciprocal()
    basis_by_dimension = derivative_basis.permute(1, 0, 2)
    coefficient_gradient = (
        torch.bmm(quadratic, coefficients.unsqueeze(2)).squeeze(2)
        - torch.sum(
            derivative_basis * reciprocal.unsqueeze(2),
            dim=0,
        )
        / sample_size
    )
    coefficient_hessian = quadratic + torch.bmm(
        basis_by_dimension.transpose(1, 2),
        basis_by_dimension
        * (reciprocal.square().T / sample_size).unsqueeze(2),
    )
    sigmoid = torch.sigmoid(raw_increments[:, 1:])
    slopes = torch.cat((torch.ones_like(raw_increments[:, :1]), sigmoid), dim=1)
    curvatures = torch.cat(
        (torch.zeros_like(raw_increments[:, :1]), sigmoid * (1.0 - sigmoid)),
        dim=1,
    )
    reverse_gradient = torch.flip(
        torch.cumsum(torch.flip(coefficient_gradient, dims=(1,)), dim=1),
        dims=(1,),
    )
    jacobian = cumulative.unsqueeze(0) * slopes.unsqueeze(1)
    hessian = torch.matmul(
        torch.matmul(jacobian.transpose(1, 2), coefficient_hessian),
        jacobian,
    ) + torch.diag_embed(curvatures * reverse_gradient)
    return coefficients, slopes * reverse_gradient, hessian


def _batched_diagonal_objective(
    raw_increments: Tensor,
    quadratic: Tensor,
    derivative_basis: Tensor,
    *,
    sample_size: int,
) -> Tensor:
    coefficients = _batched_reparameterize(raw_increments)
    derivative = torch.sum(
        derivative_basis * coefficients.unsqueeze(0),
        dim=2,
    )
    quadratic_coefficients = torch.bmm(
        quadratic,
        coefficients.unsqueeze(2),
    ).squeeze(2)
    return (
        0.5 * torch.sum(coefficients * quadratic_coefficients, dim=1)
        - torch.log(derivative).sum(dim=0) / sample_size
    )


class AdaptiveSplineTransport(nn.Module):
    """Adaptive triangular P-spline transport implemented with PyTorch.

    The class learns a lower-triangular map from samples to a standard-normal
    reference. Spline complexity is selected per additive block by minimizing
    AICc with an implicitly differentiable damped-Newton solve.
    """

    def __init__(
        self,
        k: int | None = None,
        degree: int = 3,
        skip_dimensions: int = 0,
        *,
        inner_max_iter: int = 100,
        outer_max_iter: int = 30,
        inverse_iterations: int = 60,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be at least one")
        if skip_dimensions < 0:
            raise ValueError("skip_dimensions cannot be negative")
        if k is not None and k < 1:
            raise ValueError("k must be positive")

        super().__init__()
        self.k = k
        self.degree = degree
        self.skip_dimensions = skip_dimensions
        self.inner_max_iter = inner_max_iter
        self.outer_max_iter = outer_max_iter
        self.inverse_iterations = inverse_iterations

        self.register_buffer("mean_", None)
        self.register_buffer("scale_", None)
        self.register_buffer("sparsity_", None)
        self.register_buffer("diagonal_", None)
        self.bases = nn.ModuleDict()
        self.components = nn.ModuleDict()
        self.diagonal: _BatchedDiagonal | None = None
        self.training_data_range: dict[int, tuple[Tensor, Tensor]] = {}
        self.n_samples_seen_: int | None = None
        self.n_features_in_: int | None = None
        self.n_inner_knots_: int | None = None

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

    def _basis(self, dimension: int) -> _LinearTailBasis:
        return self.bases[str(dimension)]

    def _component(self, dimension: int) -> _FittedComponent:
        return self.components[str(dimension)]

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "n_samples_seen": self.n_samples_seen_,
            "n_features_in": self.n_features_in_,
            "n_inner_knots": self.n_inner_knots_,
            "degree": self.degree,
            "skip_dimensions": self.skip_dimensions,
            "diagonal": self.diagonal is not None,
            "components": {
                dimension: {
                    "parents": component.parents,
                    "block_sizes": component.block_sizes,
                }
                for dimension, component in self.components.items()
            },
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.n_samples_seen_ = state["n_samples_seen"]
        self.n_features_in_ = state["n_features_in"]
        self.n_inner_knots_ = state["n_inner_knots"]
        self.degree = state.get("degree", self.degree)
        self.skip_dimensions = state.get("skip_dimensions", self.skip_dimensions)

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
        metadata = state_dict.get(f"{prefix}_extra_state")
        if metadata is not None and self.mean_ is None:
            self.degree = metadata.get("degree", self.degree)
            self.skip_dimensions = metadata.get(
                "skip_dimensions",
                self.skip_dimensions,
            )
            self.mean_ = torch.empty_like(state_dict[f"{prefix}mean_"])
            self.scale_ = torch.empty_like(state_dict[f"{prefix}scale_"])
            diagonal = metadata.get("diagonal", False)
            diagonal_key = f"{prefix}diagonal_"
            if diagonal_key in state_dict:
                self.diagonal_ = torch.empty_like(state_dict[diagonal_key])
            else:
                self.diagonal_ = torch.zeros(
                    (),
                    dtype=torch.bool,
                    device=self.mean_.device,
                )
                if isinstance(state_dict, MutableMapping):
                    state_dict[diagonal_key] = self.diagonal_
            if not diagonal:
                self.sparsity_ = torch.empty_like(state_dict[f"{prefix}sparsity_"])

            self.bases = nn.ModuleDict()
            n_features = metadata["n_features_in"]
            if diagonal:
                diagonal_prefix = f"{prefix}diagonal."
                self.diagonal = _BatchedDiagonal(
                    knots=torch.empty_like(state_dict[f"{diagonal_prefix}knots"]),
                    coefficients=torch.empty_like(
                        state_dict[f"{diagonal_prefix}coefficients"]
                    ),
                    log_lambdas=torch.empty_like(
                        state_dict[f"{diagonal_prefix}log_lambdas"]
                    ),
                    effective_dof=torch.empty_like(
                        state_dict[f"{diagonal_prefix}effective_dof"]
                    ),
                    aicc=torch.empty_like(state_dict[f"{diagonal_prefix}aicc"]),
                    nll=torch.empty_like(state_dict[f"{diagonal_prefix}nll"]),
                    left_values=torch.empty_like(
                        state_dict[f"{diagonal_prefix}left_values"]
                    ),
                    right_values=torch.empty_like(
                        state_dict[f"{diagonal_prefix}right_values"]
                    ),
                    left_derivatives=torch.empty_like(
                        state_dict[f"{diagonal_prefix}left_derivatives"]
                    ),
                    right_derivatives=torch.empty_like(
                        state_dict[f"{diagonal_prefix}right_derivatives"]
                    ),
                )
            else:
                for dimension in range(n_features):
                    knots = state_dict[f"{prefix}bases.{dimension}.knots"]
                    self.bases[str(dimension)] = _LinearTailBasis(
                        self.degree,
                        knots.detach().clone(),
                    )

                self.components = nn.ModuleDict()
                for dimension, component_state in metadata["components"].items():
                    component_prefix = f"{prefix}components.{dimension}"
                    self.components[str(dimension)] = _FittedComponent(
                        dimension=int(dimension),
                        parents=tuple(component_state["parents"]),
                        block_sizes=tuple(component_state["block_sizes"]),
                        log_lambdas=torch.empty_like(
                            state_dict[f"{component_prefix}.log_lambdas"]
                        ),
                        coefficients=torch.empty_like(
                            state_dict[f"{component_prefix}.coefficients"]
                        ),
                        effective_dof=torch.empty_like(
                            state_dict[f"{component_prefix}.effective_dof"]
                        ),
                        aicc=torch.empty_like(state_dict[f"{component_prefix}.aicc"]),
                        nll=torch.empty_like(state_dict[f"{component_prefix}.nll"]),
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
    def reparameterize(increments: Tensor) -> Tensor:
        return _reparameterize(increments)

    def _as_tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("fit must be called before applying the transport")

    def _make_basis(self, x: Tensor) -> _LinearTailBasis:
        assert self.n_inner_knots_ is not None
        first, last = torch.quantile(
            x,
            x.new_tensor((0.1, 0.9)),
        ).unbind()
        if not torch.isfinite(first) or not torch.isfinite(last) or first >= last:
            raise ValueError(
                "each input dimension must contain finite, non-constant samples"
            )

        real_knots = torch.linspace(
            first,
            last,
            self.n_inner_knots_ + 2,
            dtype=self.dtype,
            device=self.device,
        )
        knots = torch.cat(
            (
                real_knots[:1].repeat(self.degree),
                real_knots,
                real_knots[-1:].repeat(self.degree),
            )
        )
        return _LinearTailBasis(self.degree, knots)

    def _prepare_sparsity(
        self,
        dimensions: int,
        sparsity: Sparsity,
        *,
        device: torch.device,
    ) -> Tensor:
        rows = dimensions - self.skip_dimensions
        if rows <= 0:
            raise ValueError(
                "skip_dimensions must be smaller than the number of dimensions"
            )

        if sparsity is None:
            matrix = torch.tril(
                torch.ones(dimensions, dimensions, dtype=torch.bool, device=device)
            )[self.skip_dimensions :]
        elif sparsity == "diagonal":
            raise ValueError(
                'sparsity="diagonal" requires optimize_lambdas=False'
            )
        else:
            raw = torch.as_tensor(sparsity, device=device)
            if raw.shape != (rows, dimensions):
                raise ValueError(
                    f"sparsity must have shape {(rows, dimensions)}, got {tuple(raw.shape)}"
                )
            if not torch.all((raw == 0) | (raw == 1)):
                raise ValueError("sparsity entries must be zero or one")
            matrix = raw.bool()

        for dimension in range(self.skip_dimensions, dimensions):
            row = dimension - self.skip_dimensions
            if not matrix[row, dimension]:
                raise ValueError(
                    "every fitted component must depend on its diagonal variable"
                )
            if torch.any(matrix[row, dimension + 1 :]):
                raise ValueError("sparsity must describe a lower-triangular map")
        return matrix

    def _is_diagonal_sparsity(
        self,
        dimensions: int,
        sparsity: Sparsity,
        *,
        device: torch.device,
    ) -> bool:
        if sparsity == "diagonal":
            return True
        if sparsity is None:
            return False
        rows = dimensions - self.skip_dimensions
        if rows <= 0:
            return False
        raw = torch.as_tensor(sparsity, device=device)
        if raw.shape != (rows, dimensions):
            return False
        valid_entries = torch.all((raw == 0) | (raw == 1))
        diagonal = torch.diagonal(raw, offset=self.skip_dimensions)
        return bool(
            valid_entries
            and torch.count_nonzero(raw) == rows
            and torch.all(diagonal == 1)
        )

    def _fit_diagonal(
        self,
        standardized: Tensor,
        lambda_initial: float | Tensor | Mapping[int, Sequence[float] | Tensor],
        beta_initial: Mapping[int, Tensor | Sequence[float]] | None,
    ) -> None:
        """Fit independent diagonal splines in a single batched solve."""
        if isinstance(lambda_initial, Mapping):
            initial_logs = torch.stack(
                [
                    self._initial_log_lambdas(dimension, 1, lambda_initial)[0]
                    for dimension in range(self.skip_dimensions, standardized.shape[1])
                ]
            )
        else:
            initial = self._as_tensor(lambda_initial).flatten()
            if initial.numel() != 1:
                raise ValueError(
                    "diagonal components require a scalar lambda_initial"
                )
            initial_logs = initial.expand(
                standardized.shape[1] - self.skip_dimensions
            )
        initial_logs = initial_logs.clamp(min=-10.0, max=10.0)

        fitted = standardized[:, self.skip_dimensions :]
        sample_size, dimensions = fitted.shape
        quantiles = torch.quantile(
            fitted,
            fitted.new_tensor((0.1, 0.9)),
            dim=0,
        )
        first, last = quantiles.unbind(dim=0)
        if torch.any(~torch.isfinite(first)) or torch.any(~torch.isfinite(last)) or torch.any(first >= last):
            raise ValueError(
                "each input dimension must contain finite, non-constant samples"
            )
        knot_positions = torch.linspace(
            0.0,
            1.0,
            self.n_inner_knots_ + 2,
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)
        real_knots = first.unsqueeze(0) + (last - first).unsqueeze(0) * knot_positions
        knots = torch.cat(
            (
                real_knots[:1].repeat(self.degree, 1),
                real_knots,
                real_knots[-1:].repeat(self.degree, 1),
            ),
            dim=0,
        ).T.contiguous()
        left = knots[:, self.degree]
        right = knots[:, -self.degree - 1]
        bounded = torch.maximum(
            torch.minimum(fitted, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        design, derivative_basis = _batched_bounded_design_and_derivative(
            bounded,
            knots,
            self.degree,
        )
        nbasis = design.shape[-1]
        penalty = _smoothing_matrix(nbasis, design)
        scales = torch.sqrt(design.square().sum(dim=(0, 2)) / sample_size)
        design_by_dimension = design.permute(1, 0, 2)
        gram = torch.bmm(
            design_by_dimension.transpose(1, 2),
            design_by_dimension,
        )
        smoothing = (
            torch.exp(initial_logs) * scales.square()
        ).unsqueeze(1).unsqueeze(2) * penalty.unsqueeze(0)
        quadratic = (gram + smoothing) / sample_size
        cumulative = torch.tril(design.new_ones((nbasis, nbasis)))
        identity = torch.eye(nbasis, dtype=self.dtype, device=self.device)

        if beta_initial is None:
            raw_increments = torch.full(
                (dimensions, nbasis),
                1e-6,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            raw_increments = torch.stack(
                [
                    self._initial_raw_increments(
                        dimension,
                        nbasis,
                        beta_initial,
                    )
                    for dimension in range(
                        self.skip_dimensions,
                        self.n_features_in_,
                    )
                ]
            )

        gradient_tolerance = max(
            1e-8,
            10.0 * torch.finfo(self.dtype).eps ** 0.5,
        )
        for iteration in range(self.inner_max_iter):
            _, gradient, hessian = _batched_raw_derivatives(
                raw_increments,
                quadratic,
                derivative_basis,
                cumulative,
                sample_size=sample_size,
            )
            curvature = torch.maximum(
                torch.ones_like(initial_logs),
                torch.diagonal(hessian, dim1=1, dim2=2).abs().mean(dim=1),
            )
            damping = 1e-8 * curvature
            direction = torch.zeros_like(raw_increments)
            selected = torch.zeros(
                dimensions,
                dtype=torch.bool,
                device=self.device,
            )
            for _ in range(12):
                candidate = torch.linalg.solve(
                    hessian + damping[:, None, None] * identity,
                    -gradient.unsqueeze(2),
                ).squeeze(2)
                descent = (gradient * candidate).sum(dim=1) < 0
                valid = torch.isfinite(candidate).all(dim=1) & descent & ~selected
                direction = torch.where(valid[:, None], candidate, direction)
                selected = selected | valid
                damping = torch.where(selected, damping, damping * 10.0)
                if bool(torch.all(selected)):
                    break
            if not bool(torch.all(selected)):
                failed = torch.count_nonzero(~selected).item()
                raise RuntimeError(
                    f"could not find a descent direction for {failed} "
                    "diagonal components"
                )

            objective = _batched_diagonal_objective(
                raw_increments,
                quadratic,
                derivative_basis,
                sample_size=sample_size,
            )
            tolerance = (
                10.0
                * torch.finfo(self.dtype).eps
                * torch.maximum(torch.ones_like(objective), objective.abs())
            )
            slope = (gradient * direction).sum(dim=1)
            accepted = torch.zeros_like(selected)
            step = torch.ones_like(objective)
            accepted_step = torch.zeros_like(step)
            for _ in range(25):
                candidate = raw_increments + step[:, None] * direction
                candidate_objective = _batched_diagonal_objective(
                    candidate,
                    quadratic,
                    derivative_basis,
                    sample_size=sample_size,
                )
                accept = (
                    torch.isfinite(candidate_objective)
                    & (
                        candidate_objective
                        <= objective + 1e-4 * step * slope + tolerance
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
                failed = torch.count_nonzero(~accepted).item()
                raise RuntimeError(
                    f"line search failed for {failed} diagonal components"
                )
            if (iteration + 1) % 10 == 0:
                converged = accepted & (
                    torch.linalg.vector_norm(
                        accepted_step[:, None] * direction,
                        dim=1,
                    )
                    <= gradient_tolerance
                    * (
                        1.0
                        + torch.linalg.vector_norm(raw_increments, dim=1)
                    )
                )
                if bool(torch.all(converged)):
                    break

        coefficients = _batched_reparameterize(raw_increments).detach()
        unpenalized_quadratic = gram
        _, _, unpenalized_hessian = _batched_raw_derivatives(
            raw_increments,
            unpenalized_quadratic,
            derivative_basis,
            cumulative,
            sample_size=1,
        )
        _, _, penalized_hessian = _batched_raw_derivatives(
            raw_increments,
            gram + smoothing,
            derivative_basis,
            cumulative,
            sample_size=1,
        )
        ridge = 1e-8 * torch.maximum(
            torch.ones_like(initial_logs),
            torch.diagonal(penalized_hessian, dim1=1, dim2=2).abs().mean(dim=1),
        )
        effective_dof = torch.diagonal(
            torch.linalg.solve(
                penalized_hessian + ridge[:, None, None] * identity,
                unpenalized_hessian,
            ),
            dim1=1,
            dim2=2,
        ).sum(dim=1)
        mapped = torch.sum(design * coefficients.unsqueeze(0), dim=2)
        derivatives = torch.sum(
            derivative_basis * coefficients.unsqueeze(0),
            dim=2,
        )
        nll = 0.5 * mapped.square().sum(dim=0) - torch.log(derivatives).sum(dim=0)
        correction = effective_dof * (effective_dof + 1.0) / torch.clamp(
            sample_size - effective_dof - 1.0,
            min=1e-12,
        )
        aicc = 2.0 * (nll + effective_dof + correction)
        left_design, left_derivative_design = _batched_bounded_design_and_derivative(
            left.unsqueeze(0),
            knots,
            self.degree,
        )
        right_design, right_derivative_design = _batched_bounded_design_and_derivative(
            right.unsqueeze(0),
            knots,
            self.degree,
        )
        self.diagonal = _BatchedDiagonal(
            knots=knots,
            coefficients=raw_increments.detach(),
            log_lambdas=initial_logs.detach(),
            effective_dof=effective_dof.detach(),
            aicc=aicc.detach(),
            nll=nll.detach(),
            left_values=torch.sum(left_design[0] * coefficients, dim=1),
            right_values=torch.sum(right_design[0] * coefficients, dim=1),
            left_derivatives=torch.sum(
                left_derivative_design[0] * coefficients,
                dim=1,
            ),
            right_derivatives=torch.sum(
                right_derivative_design[0] * coefficients,
                dim=1,
            ),
        )

    def _initial_log_lambdas(
        self,
        dimension: int,
        count: int,
        initial: float | Tensor | Mapping[int, Sequence[float] | Tensor],
    ) -> Tensor:
        if isinstance(initial, Mapping):
            if dimension not in initial:
                raise ValueError(
                    f"lambda_initial has no entry for component {dimension}"
                )
            values = self._as_tensor(initial[dimension]).flatten()
        else:
            values = self._as_tensor(initial).flatten()
            if values.numel() == 1:
                values = values.repeat(count)
        if values.shape != (count,):
            raise ValueError(
                f"component {dimension} requires {count} initial log smoothing values"
            )
        return values.clamp(min=-10.0, max=10.0)

    def _initial_raw_increments(
        self,
        dimension: int,
        count: int,
        beta_initial: Mapping[int, Tensor | Sequence[float]] | None,
    ) -> Tensor:
        if beta_initial is None or dimension not in beta_initial:
            return torch.full(
                (count,),
                1e-6,
                dtype=self.dtype,
                device=self.device,
            )
        values = self._as_tensor(beta_initial[dimension]).flatten()
        if values.numel() < count:
            raise ValueError(
                f"component {dimension} beta_initial has fewer than {count} values"
            )
        return values[-count:]

    def _fit_component(
        self,
        problem: _ComponentProblem,
        lambda_initial: float | Tensor | Mapping[int, Sequence[float] | Tensor],
        optimize_lambdas: bool,
        beta_initial: Mapping[int, Tensor | Sequence[float]] | None,
    ) -> _ComponentFit:
        count = len(problem.basis_blocks)
        initial_logs = self._initial_log_lambdas(
            problem.dimension,
            count,
            lambda_initial,
        )
        initial_raw_increments = self._initial_raw_increments(
            problem.dimension,
            problem.block_sizes[-1],
            beta_initial,
        )
        implicit_solve, warm_start = _make_implicit_solver(
            problem,
            max_iter=self.inner_max_iter,
        )

        if optimize_lambdas:
            normalized = (initial_logs / 10.0).clamp(min=-0.999999, max=0.999999)
            raw_logs = torch.atanh(normalized).detach().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [raw_logs],
                max_iter=self.outer_max_iter,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                optimizer.zero_grad()
                logs = 10.0 * torch.tanh(raw_logs)
                raw_monotone_increments = implicit_solve(
                    initial_raw_increments,
                    logs,
                )
                raw_coefficients = _assemble_coefficients(
                    raw_monotone_increments,
                    logs,
                    problem,
                )
                objective, _, _ = _information_criterion(
                    raw_coefficients,
                    logs,
                    problem,
                )
                if not torch.isfinite(objective):
                    raise RuntimeError(
                        f"outer optimization for component {problem.dimension} "
                        "produced a non-finite AICc"
                    )
                objective.backward()
                return objective

            optimizer.step(closure)
            log_lambdas = (10.0 * torch.tanh(raw_logs)).detach()
        else:
            log_lambdas = initial_logs.detach()

        reduced_problem = _prepare_reduced_problem(log_lambdas, problem)
        final_initial = warm_start[0]
        if final_initial is None:
            final_initial = initial_raw_increments
        raw_monotone_increments = _solve_monotone(
            final_initial,
            log_lambdas,
            problem,
            max_iter=self.inner_max_iter,
            reduced_problem=reduced_problem,
        )
        raw_coefficients = _assemble_coefficients(
            raw_monotone_increments,
            log_lambdas,
            problem,
            reduced_problem,
        ).detach()
        aicc_half, effective_dof, nll = _information_criterion(
            raw_coefficients,
            log_lambdas,
            problem,
        )
        return _ComponentFit(
            log_lambdas=log_lambdas,
            raw_coefficients=raw_coefficients,
            effective_dof=effective_dof.detach(),
            aicc=(2.0 * aicc_half).detach(),
            nll=nll.detach(),
        )

    def fit(
        self,
        X: Tensor,
        sparsity: Sparsity = None,
        *,
        lambda_initial: float | Tensor | Mapping[int, Sequence[float] | Tensor] = 2.0,
        optimize_lambdas: bool = True,
        beta_initial: Mapping[int, Tensor | Sequence[float]] | None = None,
    ) -> AdaptiveSplineTransport:
        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor")
        if not X.is_floating_point():
            raise TypeError("X must have a floating-point dtype")
        samples = X.detach()
        if samples.ndim != 2:
            raise ValueError("X must be a two-dimensional sample matrix")
        if samples.shape[0] < 3:
            raise ValueError("at least three samples are required")
        if not torch.all(torch.isfinite(samples)):
            raise ValueError("X must contain only finite values")

        n_samples, n_features = samples.shape
        is_diagonal = (
            not optimize_lambdas
            and self._is_diagonal_sparsity(
                n_features,
                sparsity,
                device=samples.device,
            )
        )
        fitted_sparsity = (
            None
            if is_diagonal
            else self._prepare_sparsity(
                n_features,
                sparsity,
                device=samples.device,
            )
        )
        mean = samples.mean(dim=0)
        scale = samples.std(dim=0, correction=0)
        if torch.any(scale <= torch.finfo(samples.dtype).eps):
            raise ValueError("every input dimension must have non-zero variance")

        self.n_samples_seen_ = n_samples
        self.n_features_in_ = n_features
        self.n_inner_knots_ = (
            self.k
            if self.k is not None
            else max(1, int(n_samples ** (1.0 / 3.0) + 0.999999))
        )
        self.sparsity_ = fitted_sparsity
        self.diagonal_ = torch.tensor(
            is_diagonal,
            dtype=torch.bool,
            device=samples.device,
        )
        self.mean_ = mean
        self.scale_ = scale
        standardized = (samples - self.mean_) / self.scale_

        self.bases = nn.ModuleDict()
        self.components = nn.ModuleDict()
        self.diagonal = None
        self.training_data_range.clear()
        if is_diagonal:
            minimum = standardized.min(dim=0).values
            maximum = standardized.max(dim=0).values
            self.training_data_range.update(
                {
                    dimension: (minimum[dimension], maximum[dimension])
                    for dimension in range(self.n_features_in_)
                }
            )
            self._fit_diagonal(
                standardized,
                lambda_initial,
                beta_initial,
            )
            return self

        assert self.sparsity_ is not None
        cached_designs: dict[int, Tensor] = {}
        cached_derivatives: dict[int, Tensor] = {}
        cached_penalties: dict[int, Tensor] = {}
        for dimension in range(self.n_features_in_):
            basis = self._make_basis(standardized[:, dimension])
            design, derivative = basis.design_and_derivative(
                standardized[:, dimension]
            )
            self.bases[str(dimension)] = basis
            cached_designs[dimension] = design
            cached_derivatives[dimension] = derivative
            self.training_data_range[dimension] = (
                standardized[:, dimension].min(),
                standardized[:, dimension].max(),
            )

        for dimension in range(self.skip_dimensions, self.n_features_in_):
            row = dimension - self.skip_dimensions
            parents = tuple(
                parent
                for parent in range(dimension)
                if bool(self.sparsity_[row, parent])
            )
            diagonal_design = cached_designs[dimension]
            derivative = cached_derivatives[dimension]
            blocks = tuple(cached_designs[parent] for parent in parents) + (
                diagonal_design,
            )
            block_sizes = tuple(block.shape[1] for block in blocks)
            for size, block in zip(block_sizes, blocks, strict=True):
                if size not in cached_penalties:
                    cached_penalties[size] = _smoothing_matrix(size, block)
            penalties = tuple(cached_penalties[size] for size in block_sizes)
            block_scales = tuple(_block_scale(block) for block in blocks)
            design = torch.cat(blocks, dim=1)
            monotone_size = block_sizes[-1]
            monotone_cumulative = torch.tril(
                design.new_ones((monotone_size, monotone_size))
            )
            problem = _ComponentProblem(
                dimension=dimension,
                parents=parents,
                basis_blocks=blocks,
                design=design,
                gram=design.T @ design,
                nonmonotone_design=(
                    design[:, :-monotone_size] if parents else None
                ),
                derivative_basis=derivative,
                penalties=penalties,
                block_scales=block_scales,
                block_sizes=block_sizes,
                monotone_cumulative=monotone_cumulative,
                monotone_identity=torch.eye(
                    monotone_size,
                    dtype=design.dtype,
                    device=design.device,
                ),
            )
            fit = self._fit_component(
                problem,
                lambda_initial,
                optimize_lambdas,
                beta_initial,
            )
            self.components[str(dimension)] = _FittedComponent(
                dimension=dimension,
                parents=parents,
                block_sizes=block_sizes,
                log_lambdas=fit.log_lambdas,
                coefficients=fit.raw_coefficients,
                effective_dof=fit.effective_dof,
                aicc=fit.aicc,
                nll=fit.nll,
            )
        return self

    def _standardize(self, X: Tensor) -> Tensor:
        self._require_fitted()
        assert (
            self.mean_ is not None
            and self.scale_ is not None
            and self.n_features_in_ is not None
        )
        self._validate_map_input(X, self.n_features_in_, name="X")
        return (X - self.mean_) / self.scale_

    def _validate_map_input(self, value: Tensor, features: int, *, name: str) -> None:
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

    def _forward_diagonal(self, standardized: Tensor) -> Tensor:
        assert self.diagonal is not None
        return _batched_diagonal_evaluate(
            standardized[:, self.skip_dimensions :],
            self.diagonal,
            self.degree,
        )[0]

    def forward(self, X: Tensor) -> Tensor:
        standardized = self._standardize(X)
        assert self.n_features_in_ is not None
        if self.diagonal is not None:
            return self._forward_diagonal(standardized)
        prepared_evaluations = {
            dimension: self._basis(dimension).prepare_evaluation(
                standardized[:, dimension]
            )
            for dimension in range(self.n_features_in_)
        }
        outputs = []
        for dimension in range(self.skip_dimensions, self.n_features_in_):
            component = self._component(dimension)
            monotone_size = component.block_sizes[-1]
            coefficients = _materialize_coefficients(
                component.coefficients,
                monotone_size,
            )
            coefficient_blocks = torch.split(
                coefficients,
                component.block_sizes,
            )
            block_dimensions = component.parents + (dimension,)
            first_dimension = block_dimensions[0]
            mapped = self._basis(first_dimension).evaluate_prepared(
                prepared_evaluations[first_dimension],
                coefficient_blocks[0],
            )
            for block_dimension, coefficient_block in zip(
                block_dimensions[1:],
                coefficient_blocks[1:],
                strict=True,
            ):
                mapped = mapped + self._basis(block_dimension).evaluate_prepared(
                    prepared_evaluations[block_dimension],
                    coefficient_block,
                )
            outputs.append(mapped)
        return torch.stack(outputs, dim=1)

    def log_abs_det_jacobian(self, X: Tensor) -> Tensor:
        standardized = self._standardize(X)
        assert self.scale_ is not None and self.n_features_in_ is not None
        if self.diagonal is not None:
            _, derivatives = _batched_diagonal_evaluate(
                standardized[:, self.skip_dimensions :],
                self.diagonal,
                self.degree,
            )
        else:
            derivative_columns = []
            for dimension in range(
                self.skip_dimensions,
                self.n_features_in_,
            ):
                component = self._component(dimension)
                monotone_size = component.block_sizes[-1]
                coefficients = _reparameterize(
                    component.coefficients[-monotone_size:]
                )
                _, derivative = self._basis(
                    dimension
                ).evaluate_and_derivative(
                    standardized[:, dimension],
                    coefficients,
                )
                derivative_columns.append(derivative)
            derivatives = torch.stack(derivative_columns, dim=1)
        if torch.any(derivatives <= 0):
            raise RuntimeError("the fitted map has a non-positive derivative")
        scales = self.scale_[self.skip_dimensions :]
        return torch.log(derivatives / scales).sum(dim=1)

    def log_prob(self, X: Tensor) -> Tensor:
        reference = self(X)
        return (
            -0.5
            * (reference.square() + math.log(2.0 * math.pi)).sum(dim=1)
            + self.log_abs_det_jacobian(X)
        )

    def _invert_component(
        self,
        prepared_evaluations: Sequence[Tensor | _LocalBasisEvaluation],
        reference_column: Tensor,
        dimension: int,
    ) -> Tensor:
        component = self._component(dimension)
        monotone_size = component.block_sizes[-1]
        nonmonotone_coefficients = component.coefficients[:-monotone_size]
        monotone_coefficients = _reparameterize(
            component.coefficients[-monotone_size:]
        )

        if component.parents:
            coefficient_blocks = torch.split(
                nonmonotone_coefficients,
                component.block_sizes[:-1],
            )
            offset = torch.zeros_like(reference_column)
            for parent, coefficients in zip(
                component.parents,
                coefficient_blocks,
                strict=True,
            ):
                offset = offset + self._basis(parent).evaluate_prepared(
                    prepared_evaluations[parent],
                    coefficients,
                )
        else:
            offset = torch.zeros_like(reference_column)
        return _invert_monotone(
            reference_column - offset,
            self._basis(dimension),
            monotone_coefficients,
            iterations=self.inverse_iterations,
        )

    def _inverse_diagonal(self, reference: Tensor) -> Tensor:
        assert self.diagonal is not None
        diagonal = self.diagonal
        left = diagonal.knots[:, self.degree]
        right = diagonal.knots[:, -self.degree - 1]
        epsilon = 100.0 * torch.finfo(reference.dtype).eps
        below = reference < diagonal.left_values.unsqueeze(0)
        above = reference > diagonal.right_values.unsqueeze(0)
        low = left.unsqueeze(0).expand_as(reference)
        high = right.unsqueeze(0).expand_as(reference)
        middle = left.unsqueeze(0) + (right - left).unsqueeze(0) * (
            (reference - diagonal.left_values.unsqueeze(0))
            / (diagonal.right_values - diagonal.left_values).unsqueeze(0)
        )
        middle = torch.maximum(
            torch.minimum(middle, right.unsqueeze(0)),
            left.unsqueeze(0),
        )
        required_iterations = 24 if reference.dtype == torch.float32 else 40
        for _ in range(min(self.inverse_iterations, required_iterations)):
            values, slopes = _batched_diagonal_evaluate(
                middle,
                diagonal,
                self.degree,
            )
            residual = values - reference
            low = torch.where(values < reference, middle, low)
            high = torch.where(values >= reference, middle, high)
            newton = middle - residual / slopes
            valid_newton = (
                torch.isfinite(newton)
                & (slopes > epsilon)
                & (newton >= low)
                & (newton <= high)
            )
            middle = torch.where(valid_newton, newton, 0.5 * (low + high))

        left_tail = left.unsqueeze(0) + (
            reference - diagonal.left_values.unsqueeze(0)
        ) / diagonal.left_derivatives.unsqueeze(0)
        right_tail = right.unsqueeze(0) + (
            reference - diagonal.right_values.unsqueeze(0)
        ) / diagonal.right_derivatives.unsqueeze(0)
        result = torch.where(below, left_tail, middle)
        result = torch.where(above, right_tail, result)
        root = result.detach()
        root_value, root_slope = _batched_diagonal_evaluate(
            root,
            diagonal,
            self.degree,
        )
        return root + (reference - root_value) / root_slope.detach()

    def inverse(self, Z: Tensor) -> Tensor:
        self._require_fitted()
        assert (
            self.n_features_in_ is not None
            and self.mean_ is not None
            and self.scale_ is not None
        )
        if self.skip_dimensions:
            raise ValueError(
                "inverse requires skip_dimensions=0; use conditional_inverse otherwise"
            )
        self._validate_map_input(Z, self.n_features_in_, name="Z")
        if self.diagonal is not None:
            standardized = self._inverse_diagonal(Z)
            return standardized * self.scale_ + self.mean_

        standardized_columns: list[Tensor] = []
        prepared_evaluations: list[Tensor | _LocalBasisEvaluation] = []
        for dimension in range(self.n_features_in_):
            column = self._invert_component(
                prepared_evaluations,
                Z[:, dimension],
                dimension,
            )
            standardized_columns.append(column)
            prepared_evaluations.append(
                self._basis(dimension).prepare_evaluation(column)
            )
        standardized = torch.stack(standardized_columns, dim=1)
        return standardized * self.scale_ + self.mean_

    def conditional_inverse(self, X_star: Tensor, Z: Tensor) -> Tensor:
        self._require_fitted()
        assert (
            self.n_features_in_ is not None
            and self.mean_ is not None
            and self.scale_ is not None
        )
        expected_reference = self.n_features_in_ - self.skip_dimensions
        self._validate_map_input(X_star, self.skip_dimensions, name="X_star")
        self._validate_map_input(Z, expected_reference, name="Z")
        if X_star.shape[0] != Z.shape[0]:
            raise ValueError("X_star and Z must contain the same number of samples")
        if self.diagonal is not None:
            standardized = torch.cat(
                (
                    (X_star - self.mean_[: self.skip_dimensions])
                    / self.scale_[: self.skip_dimensions],
                    self._inverse_diagonal(Z),
                ),
                dim=1,
            )
            return standardized * self.scale_ + self.mean_

        standardized_columns = list(
            (
                (X_star - self.mean_[: self.skip_dimensions])
                / self.scale_[: self.skip_dimensions]
            ).unbind(dim=1)
        )
        prepared_evaluations = [
            self._basis(dimension).prepare_evaluation(column)
            for dimension, column in enumerate(standardized_columns)
        ]
        for dimension in range(self.skip_dimensions, self.n_features_in_):
            column = self._invert_component(
                prepared_evaluations,
                Z[:, dimension - self.skip_dimensions],
                dimension,
            )
            standardized_columns.append(column)
            prepared_evaluations.append(
                self._basis(dimension).prepare_evaluation(column)
            )
        standardized = torch.stack(standardized_columns, dim=1)
        return standardized * self.scale_ + self.mean_

    @property
    def log_smoothing_(self) -> dict[int, Tensor]:
        self._require_fitted()
        if self.diagonal is not None:
            return {
                dimension: self.diagonal.log_lambdas[dimension - self.skip_dimensions].unsqueeze(0)
                for dimension in range(self.skip_dimensions, self.n_features_in_)
            }
        return {
            int(dimension): component.log_lambdas
            for dimension, component in self.components.items()
            if component.log_lambdas is not None
        }

    @property
    def coefficients_(self) -> dict[int, Tensor]:
        self._require_fitted()
        if self.diagonal is not None:
            return {
                dimension: self.diagonal.coefficients[
                    dimension - self.skip_dimensions
                ]
                for dimension in range(self.skip_dimensions, self.n_features_in_)
            }
        return {
            int(dimension): component.coefficients
            for dimension, component in self.components.items()
            if component.coefficients is not None
        }

    @property
    def effective_dof_(self) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        values = torch.zeros(
            self.n_features_in_,
            dtype=self.dtype,
            device=self.device,
        )
        if self.diagonal is not None:
            values[self.skip_dimensions :] = self.diagonal.effective_dof
            return values
        for dimension, component in self.components.items():
            assert component.effective_dof is not None
            values[int(dimension)] = component.effective_dof
        return values

    @property
    def aicc_(self) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        values = torch.zeros(
            self.n_features_in_,
            dtype=self.dtype,
            device=self.device,
        )
        if self.diagonal is not None:
            values[self.skip_dimensions :] = self.diagonal.aicc
            return values
        for dimension, component in self.components.items():
            assert component.aicc is not None
            values[int(dimension)] = component.aicc
        return values

    @property
    def nll_(self) -> Tensor:
        self._require_fitted()
        assert self.n_features_in_ is not None
        values = torch.zeros(
            self.n_features_in_,
            dtype=self.dtype,
            device=self.device,
        )
        if self.diagonal is not None:
            values[self.skip_dimensions :] = self.diagonal.nll
            return values
        for dimension, component in self.components.items():
            assert component.nll is not None
            values[int(dimension)] = component.nll
        return values
