from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchopt
from torch import Tensor, nn
from torch.func import grad, hessian
from torchcurves import BSplineBasis


def _identity_map(x: Tensor, _out_min: float, _out_max: float) -> Tensor:
    return x


class _LinearTailBasis(nn.Module):
    def __init__(self, degree: int, knots: Tensor) -> None:
        super().__init__()
        self.degree = degree
        self.register_buffer("knots", knots)
        self.basis = BSplineBasis(
            degree=self.degree,
            knots_config=self.knots,
            input_map=_identity_map,
        )
        self.nbasis = self.knots.numel() - self.degree - 1
        self.register_buffer("left", self.knots[self.degree].clone())
        self.register_buffer("right", self.knots[-self.degree - 1].clone())
        self.register_buffer(
            "_identity",
            torch.eye(
                self.nbasis,
                dtype=self.knots.dtype,
                device=self.knots.device,
            ).unsqueeze(0),
            persistent=False,
        )
        edges = torch.stack((self.left, self.right))
        edge_values, edge_derivatives = self._bounded_design_and_derivative(edges)
        self.register_buffer("left_values", edge_values[0])
        self.register_buffer("right_values", edge_values[1])
        self.register_buffer("left_derivatives", edge_derivatives[0])
        self.register_buffer("right_derivatives", edge_derivatives[1])

    def _bounded_design(self, x: Tensor) -> Tensor:
        values = self.basis(x.reshape(-1, 1), self._identity)
        return values.squeeze(1)

    def _bounded_design_and_derivative(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.inference_mode(False), torch.enable_grad():
            x_grad = x.detach().clone().requires_grad_(True)
            values = self._bounded_design(x_grad)
            derivatives = torch.stack(
                [
                    torch.autograd.grad(
                        values[:, index].sum(),
                        x_grad,
                        retain_graph=index + 1 < self.nbasis,
                    )[0]
                    for index in range(self.nbasis)
                ],
                dim=1,
            )
        return values.detach(), derivatives.detach()

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
        _, derivatives = self._bounded_design_and_derivative(bounded)
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


@dataclass(frozen=True)
class _ComponentProblem:
    dimension: int
    parents: tuple[int, ...]
    basis_blocks: tuple[Tensor, ...]
    derivative_basis: Tensor
    penalties: tuple[Tensor, ...]
    block_sizes: tuple[int, ...]


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
        torch.exp(log_lambdas[index]) * _block_scale(design).square() * penalty
        for index, (design, penalty) in enumerate(
            zip(problem.basis_blocks, problem.penalties, strict=True)
        )
    )


def _regularized_normal_matrix(design: Tensor, penalty: Tensor) -> Tensor:
    normal_matrix = design.T @ design + penalty
    ridge = 1e-6 * torch.maximum(
        normal_matrix.new_ones(()),
        torch.diagonal(normal_matrix).mean(),
    )
    return normal_matrix + ridge * torch.eye(
        normal_matrix.shape[0],
        dtype=normal_matrix.dtype,
        device=normal_matrix.device,
    )


def _reduced_objective(
    raw_monotone_increments: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> Tensor:
    monotone_coefficients = _reparameterize(raw_monotone_increments)
    monotone_design = problem.basis_blocks[-1]
    scaled_penalties = _scaled_penalties(problem, log_lambdas)
    monotone_penalty = scaled_penalties[-1]

    if len(problem.basis_blocks) == 1:
        reduced_design = monotone_design
        combined_penalty = monotone_penalty
    else:
        nonmonotone_design = torch.cat(problem.basis_blocks[:-1], dim=1)
        nonmonotone_penalty = torch.block_diag(*scaled_penalties[:-1])
        normal_matrix = _regularized_normal_matrix(
            nonmonotone_design,
            nonmonotone_penalty,
        )
        projection = torch.linalg.solve(normal_matrix, nonmonotone_design.T)
        dependency = projection @ monotone_design
        reduced_design = monotone_design - nonmonotone_design @ dependency
        combined_penalty = (
            dependency.T @ nonmonotone_penalty @ dependency + monotone_penalty
        )

    derivative = problem.derivative_basis @ monotone_coefficients
    if torch.any(derivative <= 0):
        return raw_monotone_increments.new_full((), torch.inf)

    return (
        0.5 * torch.sum((reduced_design @ monotone_coefficients).square())
        - torch.sum(torch.log(derivative))
        + 0.5 * monotone_coefficients @ combined_penalty @ monotone_coefficients
    )


def _solve_monotone(
    initial_raw_increments: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
    *,
    max_iter: int,
) -> Tensor:
    parameters = log_lambdas.detach()
    raw_increments = initial_raw_increments.detach().clone()
    gradient_tolerance = max(
        1e-8,
        10.0 * torch.finfo(raw_increments.dtype).eps ** 0.5,
    )
    identity = torch.eye(
        raw_increments.numel(),
        dtype=raw_increments.dtype,
        device=raw_increments.device,
    )

    for _ in range(max_iter):
        with torch.enable_grad():
            point = raw_increments.detach().requires_grad_(True)
            objective = _reduced_objective(point, parameters, problem)
            objective_gradient = torch.autograd.grad(objective, point)[0]
            objective_hessian = hessian(
                lambda candidate: _reduced_objective(
                    candidate,
                    parameters,
                    problem,
                )
            )(point)

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
        accepted = False
        for _ in range(25):
            candidate = raw_increments + step * direction
            candidate_objective = _reduced_objective(
                candidate,
                parameters,
                problem,
            )
            if torch.isfinite(candidate_objective) and (
                candidate_objective <= initial_objective + 1e-4 * step * slope
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

    final_gradient = grad(
        lambda candidate: _reduced_objective(
            candidate,
            parameters,
            problem,
        )
    )(raw_increments)
    gradient_norm = torch.linalg.vector_norm(final_gradient, ord=float("inf"))
    if gradient_norm > 10.0 * gradient_tolerance:
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
    def optimality(
        raw_monotone_increments: Tensor,
        log_lambdas: Tensor,
    ) -> Tensor:
        return grad(
            lambda candidate: _reduced_objective(
                candidate,
                log_lambdas,
                problem,
            )
        )(raw_monotone_increments)

    @torchopt.diff.implicit.custom_root(optimality, argnums=1)
    def solve(
        initial_raw_increments: Tensor,
        log_lambdas: Tensor,
    ) -> Tensor:
        return _solve_monotone(
            initial_raw_increments,
            log_lambdas,
            problem,
            max_iter=max_iter,
        )

    return solve


def _assemble_coefficients(
    raw_monotone_increments: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> Tensor:
    if len(problem.basis_blocks) == 1:
        return raw_monotone_increments

    scaled_penalties = _scaled_penalties(problem, log_lambdas)
    nonmonotone_design = torch.cat(problem.basis_blocks[:-1], dim=1)
    monotone_design = problem.basis_blocks[-1]
    nonmonotone_penalty = torch.block_diag(*scaled_penalties[:-1])
    normal_matrix = _regularized_normal_matrix(
        nonmonotone_design,
        nonmonotone_penalty,
    )
    right_hand_side = (
        nonmonotone_design.T
        @ monotone_design
        @ _reparameterize(raw_monotone_increments)
    )
    nonmonotone_coefficients = -torch.linalg.solve(
        normal_matrix,
        right_hand_side,
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
    design = torch.cat(problem.basis_blocks, dim=1)
    mapped = design @ coefficients
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


def _information_criterion(
    raw_coefficients: Tensor,
    log_lambdas: Tensor,
    problem: _ComponentProblem,
) -> tuple[Tensor, Tensor, Tensor]:
    unpenalized_hessian = hessian(lambda candidate: _nll(candidate, problem))(
        raw_coefficients
    )
    penalized_hessian = hessian(
        lambda candidate: _penalized_nll(
            candidate,
            log_lambdas,
            problem,
        )
    )(raw_coefficients)
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
    result = torch.empty_like(target)
    result[below] = basis.left + (target[below] - left_value) / left_slope
    result[above] = basis.right + (target[above] - right_value) / right_slope

    inside = ~(below | above)
    if torch.any(inside):
        inside_target = target[inside]
        low = torch.full_like(inside_target, basis.left)
        high = torch.full_like(inside_target, basis.right)
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            values = basis.design(middle) @ monotone_coefficients
            low = torch.where(values < inside_target, middle, low)
            high = torch.where(values >= inside_target, middle, high)
        result[inside] = 0.5 * (low + high)

    root = result.detach()
    root_value = basis.design(root) @ monotone_coefficients
    root_slope = basis.derivative_design(root) @ monotone_coefficients
    if torch.any(root_slope <= epsilon):
        raise RuntimeError("the fitted map is not strictly monotone")
    return root + (target - root_value) / root_slope.detach()


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
        self.bases = nn.ModuleDict()
        self.components = nn.ModuleDict()
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
            self.sparsity_ = torch.empty_like(state_dict[f"{prefix}sparsity_"])

            self.bases = nn.ModuleDict()
            n_features = metadata["n_features_in"]
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
        first = torch.quantile(x, 0.1)
        last = torch.quantile(x, 0.9)
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
        sparsity: Tensor | Sequence[Sequence[int]] | None,
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
        implicit_solve = _make_implicit_solver(
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

        raw_monotone_increments = _solve_monotone(
            initial_raw_increments,
            log_lambdas,
            problem,
            max_iter=self.inner_max_iter,
        )
        raw_coefficients = _assemble_coefficients(
            raw_monotone_increments,
            log_lambdas,
            problem,
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
        sparsity: Tensor | Sequence[Sequence[int]] | None = None,
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
        fitted_sparsity = self._prepare_sparsity(
            n_features,
            sparsity,
            device=samples.device,
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
        self.mean_ = mean
        self.scale_ = scale
        standardized = (samples - self.mean_) / self.scale_

        self.bases = nn.ModuleDict()
        self.components = nn.ModuleDict()
        self.training_data_range.clear()
        cached_designs: dict[int, Tensor] = {}
        for dimension in range(self.n_features_in_):
            basis = self._make_basis(standardized[:, dimension])
            design, _ = basis.design_and_derivative(standardized[:, dimension])
            self.bases[str(dimension)] = basis
            cached_designs[dimension] = design
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
            diagonal_design, derivative = self._basis(dimension).design_and_derivative(
                standardized[:, dimension]
            )
            blocks = tuple(cached_designs[parent] for parent in parents) + (
                diagonal_design,
            )
            block_sizes = tuple(block.shape[1] for block in blocks)
            penalties = tuple(
                _smoothing_matrix(
                    size,
                    block,
                )
                for size, block in zip(block_sizes, blocks, strict=True)
            )
            problem = _ComponentProblem(
                dimension=dimension,
                parents=parents,
                basis_blocks=blocks,
                derivative_basis=derivative,
                penalties=penalties,
                block_sizes=block_sizes,
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

    def forward(self, X: Tensor) -> Tensor:
        standardized = self._standardize(X)
        assert self.n_features_in_ is not None
        outputs = []
        for dimension in range(self.skip_dimensions, self.n_features_in_):
            component = self._component(dimension)
            monotone_size = component.block_sizes[-1]
            coefficients = _materialize_coefficients(
                component.coefficients,
                monotone_size,
            )
            blocks = tuple(
                self._basis(parent).design(standardized[:, parent])
                for parent in component.parents
            ) + (self._basis(dimension).design(standardized[:, dimension]),)
            outputs.append(torch.cat(blocks, dim=1) @ coefficients)
        return torch.stack(outputs, dim=1)

    def _invert_component(
        self,
        standardized_columns: Sequence[Tensor],
        reference_column: Tensor,
        dimension: int,
    ) -> Tensor:
        component = self._component(dimension)
        monotone_size = component.block_sizes[-1]
        nonmonotone_coefficients = component.coefficients[:-monotone_size]
        monotone_coefficients = _reparameterize(component.coefficients[-monotone_size:])

        if component.parents:
            parent_design = torch.cat(
                tuple(
                    self._basis(parent).design(standardized_columns[parent])
                    for parent in component.parents
                ),
                dim=1,
            )
            offset = parent_design @ nonmonotone_coefficients
        else:
            offset = torch.zeros_like(reference_column)
        return _invert_monotone(
            reference_column - offset,
            self._basis(dimension),
            monotone_coefficients,
            iterations=self.inverse_iterations,
        )

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

        standardized_columns: list[Tensor] = []
        for dimension in range(self.n_features_in_):
            standardized_columns.append(
                self._invert_component(
                    standardized_columns,
                    Z[:, dimension],
                    dimension,
                )
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

        standardized_columns = list(
            (
                (X_star - self.mean_[: self.skip_dimensions])
                / self.scale_[: self.skip_dimensions]
            ).unbind(dim=1)
        )
        for dimension in range(self.skip_dimensions, self.n_features_in_):
            standardized_columns.append(
                self._invert_component(
                    standardized_columns,
                    Z[:, dimension - self.skip_dimensions],
                    dimension,
                )
            )
        standardized = torch.stack(standardized_columns, dim=1)
        return standardized * self.scale_ + self.mean_

    @property
    def log_smoothing_(self) -> dict[int, Tensor]:
        self._require_fitted()
        return {
            int(dimension): component.log_lambdas
            for dimension, component in self.components.items()
            if component.log_lambdas is not None
        }

    @property
    def coefficients_(self) -> dict[int, Tensor]:
        self._require_fitted()
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
        for dimension, component in self.components.items():
            assert component.nll is not None
            values[int(dimension)] = component.nll
        return values
