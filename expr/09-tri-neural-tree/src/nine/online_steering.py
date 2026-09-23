from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from nine.boosted_soft_transport import BoostedSoftTreeTransport

ScoreFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class SteeringSearchConfig:
    rank: int = 8
    iterations: int = 2
    antithetic_pairs: int = 4
    relative_radius: float = 0.08
    proposal_scale: float = 0.5
    proposal_decay: float = 0.7
    learning_rate: float = 0.75
    elite_fraction: float = 0.25
    proposal: str = "gaussian"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.antithetic_pairs < 1:
            raise ValueError("antithetic_pairs must be positive")
        if self.relative_radius <= 0.0:
            raise ValueError("relative_radius must be positive")
        if self.proposal_scale <= 0.0:
            raise ValueError("proposal_scale must be positive")
        if not 0.0 < self.proposal_decay <= 1.0:
            raise ValueError("proposal_decay must be in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.proposal not in {"gaussian", "transport"}:
            raise ValueError("proposal must be 'gaussian' or 'transport'")


@dataclass(frozen=True)
class SteeringIteration:
    iteration: int
    candidate_scores: tuple[float, ...]
    best_score: float
    proposal_scale: float


@dataclass(frozen=True)
class SteeringSearchResult:
    vector: Tensor
    coefficients: Tensor
    best_score: float
    history: tuple[SteeringIteration, ...]


def shrinkage_covariance(
    values: Tensor,
    *,
    shrinkage: float = 0.9,
    minimum_variance: float | None = None,
) -> Tensor:
    """Estimate a finite covariance when samples are scarce."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must have shape (N, D) with N >= 2")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    centered = values - values.mean(dim=0)
    empirical = centered.T @ centered / (values.shape[0] - 1)
    average_variance = empirical.diagonal().mean()
    floor = (
        100.0 * torch.finfo(values.dtype).eps
        if minimum_variance is None
        else minimum_variance
    )
    isotropic_variance = average_variance.clamp_min(floor)
    identity = torch.eye(
        values.shape[1],
        dtype=values.dtype,
        device=values.device,
    )
    return (1.0 - shrinkage) * empirical + shrinkage * isotropic_variance * identity


def contrastive_basis(
    activations: Tensor,
    scores: Tensor,
    *,
    rank: int,
    elite_fraction: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Build an orthonormal basis headed by elite-minus-rejected direction."""
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must have shape (N, D) with N >= 2")
    if scores.shape != (activations.shape[0],):
        raise ValueError("scores must have shape (N,)")
    if rank < 1:
        raise ValueError("rank must be positive")
    if not 0.0 < elite_fraction <= 0.5:
        raise ValueError("elite_fraction must be in (0, 0.5]")
    if not torch.all(torch.isfinite(activations)) or not torch.all(
        torch.isfinite(scores)
    ):
        raise ValueError("activations and scores must be finite")

    group_size = max(1, math.floor(activations.shape[0] * elite_fraction))
    order = torch.argsort(scores, descending=True)
    contrast = (
        activations[order[:group_size]].mean(dim=0)
        - activations[order[-group_size:]].mean(dim=0)
    )
    centered = activations - activations.mean(dim=0)
    _, centered_singular_values, right = torch.linalg.svd(
        centered,
        full_matrices=False,
    )
    activation_tolerance = (
        100.0
        * torch.finfo(activations.dtype).eps
        * centered_singular_values.max().clamp_min(1.0)
    )
    supported = right[centered_singular_values > activation_tolerance]
    contrast_norm = torch.linalg.vector_norm(contrast)
    candidate_rows = []
    if contrast_norm > activation_tolerance:
        candidate_rows.append(contrast.unsqueeze(0))
    if supported.numel():
        candidate_rows.append(supported)
    if not candidate_rows:
        basis = torch.eye(
            activations.shape[1],
            dtype=activations.dtype,
            device=activations.device,
        )[:, :1]
        return basis, contrast

    candidates = torch.cat(candidate_rows, dim=0).T
    basis, singular_values, _ = torch.linalg.svd(candidates, full_matrices=False)
    tolerance = (
        100.0
        * torch.finfo(activations.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    independent = int(torch.count_nonzero(singular_values > tolerance))
    final_rank = min(rank, independent, activations.shape[1])
    if final_rank == 0:
        basis = torch.eye(
            activations.shape[1],
            dtype=activations.dtype,
            device=activations.device,
        )[:, :1]
    else:
        basis = basis[:, :final_rank]
    return basis, contrast


def project_to_ball(values: Tensor, radius: float) -> Tensor:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    scales = (radius / norms.clamp_min(torch.finfo(values.dtype).eps)).clamp(max=1.0)
    return values * scales


def _standardized(values: Tensor) -> Tensor:
    scale = values.std(correction=0)
    if scale <= torch.finfo(values.dtype).eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / scale


def _transport_proposals(
    coefficients: Tensor,
    scores: Tensor,
    *,
    count: int,
    elite_fraction: float,
    radius: float,
) -> Tensor | None:
    minimum_count = max(6, coefficients.shape[1] + 2)
    if coefficients.shape[0] < minimum_count:
        return None
    elite_count = max(
        minimum_count,
        math.ceil(coefficients.shape[0] * elite_fraction),
    )
    elite_count = min(elite_count, coefficients.shape[0])
    elites = coefficients[torch.topk(scores, elite_count).indices]
    if (
        torch.any(
            elites.std(dim=0, correction=0)
            <= 100.0 * torch.finfo(elites.dtype).eps
        )
    ):
        return None
    transport = BoostedSoftTreeTransport(
        max_stages=1,
        max_depth=1,
        num_bins=4,
        learning_rate=1e-2,
        max_epochs=20,
        patience=5,
        validation_fraction=0.0,
        max_parent_distance=None,
        max_fit_iterations=3,
        minimum_improvement_per_sample=0.0,
        fine_tune_epochs=0,
    ).to(device=elites.device, dtype=elites.dtype)
    transport.fit(elites)
    reference = torch.randn(
        count,
        elites.shape[1],
        dtype=elites.dtype,
        device=elites.device,
    )
    return project_to_ball(transport.inverse(reference), radius)


def optimize_steering_vector(
    basis: Tensor,
    reference_state: Tensor,
    evaluate: ScoreFunction,
    *,
    initial_vector: Tensor | None = None,
    config: SteeringSearchConfig | None = None,
    generator: torch.Generator | None = None,
) -> SteeringSearchResult:
    """Optimize one additive vector through low-rank black-box search."""
    config = config or SteeringSearchConfig()
    if basis.ndim != 2 or reference_state.ndim != 1:
        raise ValueError("basis and reference_state must be a matrix and vector")
    if basis.shape[0] != reference_state.numel():
        raise ValueError("basis and reference_state dimensions are incompatible")
    if basis.shape[1] < 1 or basis.shape[1] > config.rank:
        raise ValueError("basis rank must be between 1 and config.rank")
    gram = basis.T @ basis
    identity = torch.eye(
        basis.shape[1],
        dtype=basis.dtype,
        device=basis.device,
    )
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("basis columns must be orthonormal")

    radius = config.relative_radius * float(torch.linalg.vector_norm(reference_state))
    radius = max(radius, 100.0 * torch.finfo(reference_state.dtype).eps)
    if initial_vector is None:
        center = reference_state.new_zeros(basis.shape[1])
    else:
        if initial_vector.shape != reference_state.shape:
            raise ValueError("initial_vector and reference_state must have equal shape")
        center = project_to_ball((basis.T @ initial_vector).unsqueeze(0), radius)[0]

    initial_score = evaluate((basis @ center).unsqueeze(0))
    if initial_score.shape != (1,) or not torch.all(torch.isfinite(initial_score)):
        raise ValueError("evaluate must return one finite score per vector")
    best_coefficients = center.clone()
    best_score = float(initial_score[0])
    history: list[SteeringIteration] = []
    coefficient_archive: list[Tensor] = []
    score_archive: list[Tensor] = []
    scale = config.proposal_scale

    for iteration in range(config.iterations):
        directions = torch.randn(
            config.antithetic_pairs,
            basis.shape[1],
            dtype=basis.dtype,
            device=basis.device,
            generator=generator,
        )
        directions = directions / torch.linalg.vector_norm(
            directions,
            dim=1,
            keepdim=True,
        ).clamp_min(torch.finfo(directions.dtype).eps)
        sigma = scale * radius
        positive = center.unsqueeze(0) + sigma * directions
        negative = center.unsqueeze(0) - sigma * directions
        coefficients = project_to_ball(
            torch.cat((positive, negative), dim=0),
            radius,
        )
        vectors = coefficients @ basis.T
        scores = evaluate(vectors)
        if scores.shape != (coefficients.shape[0],):
            raise ValueError("evaluate must return one score per vector")
        if not torch.all(torch.isfinite(scores)):
            raise RuntimeError("steering evaluation returned a non-finite score")
        coefficient_archive.append(coefficients.detach())
        score_archive.append(scores.detach())

        iteration_best = int(torch.argmax(scores))
        iteration_best_score = float(scores[iteration_best])
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_coefficients = coefficients[iteration_best].clone()

        paired_difference = scores[: config.antithetic_pairs] - scores[
            config.antithetic_pairs :
        ]
        weights = _standardized(paired_difference)
        update = (weights.unsqueeze(1) * directions).mean(dim=0)
        update_norm = torch.linalg.vector_norm(update)
        if update_norm > torch.finfo(update.dtype).eps:
            center = center + (
                config.learning_rate * sigma * update / update_norm
            )
            center = project_to_ball(center.unsqueeze(0), radius)[0]

        if config.proposal == "transport":
            proposals = _transport_proposals(
                torch.cat(coefficient_archive),
                torch.cat(score_archive),
                count=coefficients.shape[0],
                elite_fraction=config.elite_fraction,
                radius=radius,
            )
            if proposals is not None:
                proposal_scores = evaluate(proposals @ basis.T)
                proposal_best = int(torch.argmax(proposal_scores))
                if float(proposal_scores[proposal_best]) > best_score:
                    best_score = float(proposal_scores[proposal_best])
                    best_coefficients = proposals[proposal_best].clone()
                coefficient_archive.append(proposals.detach())
                score_archive.append(proposal_scores.detach())
                elite_count = max(
                    1,
                    math.ceil(proposals.shape[0] * config.elite_fraction),
                )
                center = proposals[torch.topk(proposal_scores, elite_count).indices].mean(
                    dim=0
                )
                center = project_to_ball(center.unsqueeze(0), radius)[0]
                scores = torch.cat((scores, proposal_scores))

        history.append(
            SteeringIteration(
                iteration=iteration,
                candidate_scores=tuple(float(score) for score in scores),
                best_score=best_score,
                proposal_scale=scale,
            )
        )
        scale *= config.proposal_decay

    center_score = evaluate((basis @ center).unsqueeze(0))
    if float(center_score[0]) > best_score:
        best_score = float(center_score[0])
        best_coefficients = center.clone()
    return SteeringSearchResult(
        vector=basis @ best_coefficients,
        coefficients=best_coefficients,
        best_score=best_score,
        history=tuple(history),
    )


class LayerSteeringHook:
    """Capture decoder activations and add one vector to each batch item."""

    def __init__(self) -> None:
        self._vectors: Tensor | None = None
        self._strength = 1.0
        self._captures: list[Tensor] = []

    def set_vectors(self, vectors: Tensor | None, *, strength: float = 1.0) -> None:
        if vectors is not None and vectors.ndim != 2:
            raise ValueError("vectors must have shape (batch, hidden_dimension)")
        self._vectors = vectors
        self._strength = strength

    def reset_capture(self) -> None:
        self._captures = []

    def activation_means(self, active_steps: Tensor | None = None) -> Tensor:
        if not self._captures:
            raise RuntimeError("no activations have been captured")
        captures = torch.stack(self._captures)
        if active_steps is None:
            return captures.mean(dim=0)
        if active_steps.shape != (captures.shape[1],):
            raise ValueError("active_steps must contain one count per batch item")
        steps = torch.arange(captures.shape[0], device=captures.device).unsqueeze(1)
        mask = steps < active_steps.to(device=captures.device).unsqueeze(0)
        weighted = captures * mask.unsqueeze(2)
        return weighted.sum(dim=0) / active_steps.to(captures).clamp_min(1).unsqueeze(1)

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Tensor | tuple[Any, ...],
    ) -> Tensor | tuple[Any, ...]:
        del module, inputs
        hidden = output if isinstance(output, Tensor) else output[0]
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise TypeError("decoder layer output must contain a rank-three tensor")
        last = hidden[:, -1, :]
        detached = last.detach().float()
        self._captures.append(detached)

        if self._vectors is None:
            return output
        if self._vectors.shape != last.shape:
            raise ValueError(
                "steering vectors must match the decoder output batch and width"
            )
        steered = hidden.clone()
        steered[:, -1, :] = (
            last + self._strength * self._vectors.to(last)
        )
        if isinstance(output, Tensor):
            return steered
        return (steered, *output[1:])


__all__ = [
    "LayerSteeringHook",
    "SteeringIteration",
    "SteeringSearchConfig",
    "SteeringSearchResult",
    "contrastive_basis",
    "optimize_steering_vector",
    "project_to_ball",
    "shrinkage_covariance",
]
