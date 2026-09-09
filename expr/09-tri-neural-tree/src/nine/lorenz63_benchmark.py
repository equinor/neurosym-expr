from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from nine.adaptive_pspline import AdaptiveSplineTransport
from nine.boosted_transport import BoostedHardTreeTransport

PAPER_ENSEMBLE_SIZES = (50, 100, 175, 250, 375, 500, 750, 1000)
PAPER_SEEDS = tuple(range(10))
OBSERVATION_PERMUTATIONS = ((0, 1, 2), (1, 0, 2), (2, 1, 0))


@dataclass(frozen=True)
class Lorenz63Result:
    model: str
    seed: int
    ensemble_size: int
    steps: int
    spinup_steps: int
    mean_rmse: float
    mean_crps: float
    duration_seconds: float


def lorenz63_dynamics(values: Tensor) -> Tensor:
    x, y, z = values.unbind(dim=-1)
    return torch.stack(
        (
            10.0 * (y - x),
            x * (28.0 - z) - y,
            x * y - (8.0 / 3.0) * z,
        ),
        dim=-1,
    )


def rk4(values: Tensor, *, step_size: float = 0.05, steps: int = 2) -> Tensor:
    result = values
    for _ in range(steps):
        first = lorenz63_dynamics(result)
        second = lorenz63_dynamics(result + 0.5 * step_size * first)
        third = lorenz63_dynamics(result + 0.5 * step_size * second)
        fourth = lorenz63_dynamics(result + step_size * third)
        result = result + step_size / 6.0 * (
            first + 2.0 * second + 2.0 * third + fourth
        )
    return result


def generate_truth_and_observations(
    *,
    seed: int,
    steps: int,
    spinup_steps: int,
    observation_sd: float = 2.0,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    truth = torch.empty(steps + spinup_steps, 3, dtype=torch.float64)
    truth[0] = torch.randn(3, generator=generator, dtype=torch.float64)
    for index in range(truth.shape[0] - 1):
        truth[index + 1] = rk4(truth[index])
    observations = truth + observation_sd * torch.randn(
        truth.shape,
        generator=generator,
        dtype=truth.dtype,
    )
    return truth, observations


def stochastic_enkf_update(
    ensemble: Tensor,
    observation: Tensor,
    *,
    observation_sd: float,
    generator: torch.Generator,
) -> Tensor:
    centered = ensemble - ensemble.mean(dim=0)
    covariance = centered.T @ centered / (ensemble.shape[0] - 1)
    observation_variance = observation_sd**2
    gain = covariance @ torch.linalg.inv(
        covariance
        + observation_variance
        * torch.eye(3, dtype=ensemble.dtype, device=ensemble.device)
    )
    perturbed = observation.unsqueeze(0) + observation_sd * torch.randn(
        ensemble.shape,
        generator=generator,
        dtype=ensemble.dtype,
        device=ensemble.device,
    )
    return ensemble + (perturbed - ensemble) @ gain.T


def spinup_ensemble(
    observations: Tensor,
    *,
    ensemble_size: int,
    spinup_steps: int,
    observation_sd: float,
    generator: torch.Generator,
) -> Tensor:
    ensemble = torch.randn(
        ensemble_size,
        3,
        generator=generator,
        dtype=observations.dtype,
        device=observations.device,
    )
    for index in range(spinup_steps + 1):
        ensemble = stochastic_enkf_update(
            ensemble,
            observations[index],
            observation_sd=observation_sd,
            generator=generator,
        )
        if index < spinup_steps:
            ensemble = rk4(ensemble)
    return ensemble


def ensemble_crps(ensemble: Tensor, truth: Tensor) -> Tensor:
    first = (ensemble - truth.unsqueeze(0)).abs().mean(dim=0)
    sorted_values = torch.sort(ensemble, dim=0).values
    count = ensemble.shape[0]
    coefficients = (
        2.0
        * torch.arange(1, count + 1, dtype=ensemble.dtype).unsqueeze(1)
        - count
        - 1.0
    )
    pairwise = (coefficients * sorted_values).sum(dim=0) / count**2
    return first - pairwise


def _adaptive_update(
    map_input: Tensor,
    observation: Tensor,
    *,
    optimize_lambdas: bool,
    lambda_initial: float | dict[int, Tensor],
) -> tuple[Tensor, dict[int, Tensor]]:
    sparsity = torch.tensor(
        (
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
        ),
        dtype=torch.bool,
        device=map_input.device,
    )
    model = AdaptiveSplineTransport(
        skip_dimensions=1,
        inner_max_iter=40,
        outer_max_iter=15,
    ).fit(
        map_input,
        sparsity=sparsity,
        lambda_initial=lambda_initial,
        optimize_lambdas=optimize_lambdas,
    )
    reference = model(map_input)
    conditioned = model.conditional_inverse(
        observation.expand(map_input.shape[0], 1),
        reference,
    )
    return conditioned[:, 1:], {
        key: value.detach().clone()
        for key, value in model.log_smoothing_.items()
    }


def _hard_tree_update(map_input: Tensor, observation: Tensor) -> Tensor:
    model = BoostedHardTreeTransport(
        max_stages=4,
        max_leaves=16,
        max_fit_iterations=15,
    ).fit(map_input)
    reference = model(map_input)
    return model.conditional_inverse(
        observation.expand(map_input.shape[0], 1),
        reference[:, 1:],
    )[:, 1:]


def run_filter(
    *,
    model: str,
    seed: int,
    ensemble_size: int,
    steps: int = 1000,
    spinup_steps: int = 1000,
    adaptation_steps: int = 10,
    observation_sd: float = 2.0,
) -> Lorenz63Result:
    truth, observations = generate_truth_and_observations(
        seed=seed,
        steps=steps,
        spinup_steps=spinup_steps,
        observation_sd=observation_sd,
    )
    generator = torch.Generator().manual_seed(seed)
    ensemble = spinup_ensemble(
        observations,
        ensemble_size=ensemble_size,
        spinup_steps=spinup_steps,
        observation_sd=observation_sd,
        generator=generator,
    )
    smoothing_history: dict[int, list[Tensor]] = {}
    fixed_smoothing: dict[int, Tensor] | None = None
    squared_errors = []
    crps_values = []
    start = time.perf_counter()

    for step in range(steps):
        for observed_index, permutation in enumerate(OBSERVATION_PERMUTATIONS):
            simulated_observation = (
                ensemble[:, observed_index]
                + observation_sd
                * torch.randn(
                    ensemble_size,
                    generator=generator,
                    dtype=ensemble.dtype,
                )
            )
            map_input = torch.cat(
                (
                    simulated_observation.unsqueeze(1),
                    ensemble[:, permutation],
                ),
                dim=1,
            )
            actual_observation = observations[
                spinup_steps + step,
                observed_index,
            ].reshape(1)
            if model == "adaptive-p-spline":
                adapting = step < adaptation_steps
                if not adapting and fixed_smoothing is None:
                    fixed_smoothing = {
                        key: torch.median(torch.stack(values), dim=0).values
                        for key, values in smoothing_history.items()
                    }
                conditioned, smoothing = _adaptive_update(
                    map_input,
                    actual_observation,
                    optimize_lambdas=adapting,
                    lambda_initial=(
                        0.0 if adapting else fixed_smoothing or 0.0
                    ),
                )
                if adapting:
                    for key, values in smoothing.items():
                        smoothing_history.setdefault(key, []).append(values)
            elif model == "boosted-hard-tree":
                conditioned = _hard_tree_update(
                    map_input,
                    actual_observation,
                )
            else:
                raise ValueError(f"unknown model: {model}")

            inverse_permutation = torch.argsort(torch.tensor(permutation))
            ensemble = conditioned[:, inverse_permutation]

        target = truth[spinup_steps + step]
        squared_errors.append((ensemble.mean(dim=0) - target).square().mean())
        crps_values.append(ensemble_crps(ensemble, target).mean())
        if step < steps - 1:
            model_sd = 5.0 / ensemble_size
            ensemble = rk4(ensemble) + model_sd * torch.randn(
                ensemble.shape,
                generator=generator,
                dtype=ensemble.dtype,
            )

    return Lorenz63Result(
        model=model,
        seed=seed,
        ensemble_size=ensemble_size,
        steps=steps,
        spinup_steps=spinup_steps,
        mean_rmse=float(torch.sqrt(torch.stack(squared_errors)).mean().item()),
        mean_crps=float(torch.stack(crps_values).mean().item()),
        duration_seconds=time.perf_counter() - start,
    )


def _parse_integer_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="adaptive-p-spline,boosted-hard-tree",
    )
    parser.add_argument(
        "--ensemble-sizes",
        default=",".join(map(str, PAPER_ENSEMBLE_SIZES)),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, PAPER_SEEDS)),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--spinup-steps", type=int, default=1000)
    parser.add_argument("--adaptation-steps", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lorenz63_results.json"),
    )
    arguments = parser.parse_args()
    models = tuple(
        item for item in arguments.models.split(",") if item
    )
    results: list[Lorenz63Result] = []
    if arguments.output.exists():
        payload = json.loads(arguments.output.read_text())
        results = [Lorenz63Result(**row) for row in payload["results"]]
    completed = {
        (row.model, row.seed, row.ensemble_size, row.steps, row.spinup_steps)
        for row in results
    }

    for model in models:
        for seed in _parse_integer_list(arguments.seeds):
            for ensemble_size in _parse_integer_list(arguments.ensemble_sizes):
                key = (
                    model,
                    seed,
                    ensemble_size,
                    arguments.steps,
                    arguments.spinup_steps,
                )
                if key in completed:
                    continue
                result = run_filter(
                    model=model,
                    seed=seed,
                    ensemble_size=ensemble_size,
                    steps=arguments.steps,
                    spinup_steps=arguments.spinup_steps,
                    adaptation_steps=arguments.adaptation_steps,
                )
                results.append(result)
                arguments.output.write_text(
                    json.dumps(
                        {"results": [asdict(row) for row in results]},
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
