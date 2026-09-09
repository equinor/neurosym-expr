from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from nine.adaptive_pspline import AdaptiveSplineTransport
from nine.boosted_soft_transport import BoostedSoftTreeTransport
from nine.boosted_transport import BoostedHardTreeTransport


@dataclass(frozen=True)
class BenchmarkResult:
    experiment: str
    model: str
    seed: int
    training_samples: int
    validation_nll: float
    oracle_nll: float
    excess_nll: float
    reference_mean_rmse: float
    reference_covariance_rmse: float
    fit_seconds: float
    stages: int | None
    leaves: int | None


def _with_seed(seed: int, function: Callable[[], Tensor]) -> Tensor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return function()


def sample_overfitting_mixture(
    count: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    def sample() -> Tensor:
        component = torch.rand(count, dtype=dtype) > 0.25
        means = torch.where(component, 1.0, -1.0)
        return (means + 0.5 * torch.randn(count, dtype=dtype)).unsqueeze(1)

    return _with_seed(seed, sample)


def overfitting_mixture_log_prob(values: Tensor) -> Tensor:
    x = values[:, 0]
    constant = -math.log(0.5) - 0.5 * math.log(2.0 * math.pi)
    left = math.log(0.25) + constant - 0.5 * ((x + 1.0) / 0.5).square()
    right = math.log(0.75) + constant - 0.5 * ((x - 1.0) / 0.5).square()
    return torch.logsumexp(torch.stack((left, right), dim=1), dim=1)


def sample_wavy(
    count: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    def sample() -> Tensor:
        beta = torch.distributions.Beta(
            torch.tensor(2.0, dtype=dtype),
            torch.tensor(2.0, dtype=dtype),
        ).sample((count,))
        first = (2.0 * beta - 1.0) * 3.0
        second = (
            torch.randn(count, dtype=dtype) / 6.0
            + torch.sin(1.2 * first)
        )
        return torch.stack((first / 1.5, second * 1.5), dim=1)

    return _with_seed(seed, sample)


def wavy_log_prob(values: Tensor) -> Tensor:
    first = values[:, 0] * 1.5
    second = values[:, 1] / 1.5 - torch.sin(1.2 * first)
    beta = ((first / 3.0 + 1.0) / 2.0).clamp(1e-12, 1.0 - 1e-12)
    beta_log_prob = math.log(6.0) + torch.log(beta) + torch.log1p(-beta)
    normal_log_prob = (
        -0.5 * (second * 6.0).square()
        - math.log(1.0 / 6.0)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return beta_log_prob + normal_log_prob - math.log(6.0)


def _reference_diagnostics(reference: Tensor) -> tuple[float, float]:
    mean_rmse = torch.sqrt(reference.mean(dim=0).square().mean())
    centered = reference - reference.mean(dim=0)
    covariance = centered.T @ centered / reference.shape[0]
    identity = torch.eye(
        reference.shape[1],
        dtype=reference.dtype,
        device=reference.device,
    )
    covariance_rmse = torch.sqrt((covariance - identity).square().mean())
    return float(mean_rmse.item()), float(covariance_rmse.item())


def _model_factories() -> list[tuple[str, Callable[[], object]]]:
    return [
        (
            "adaptive-p-spline",
            lambda: AdaptiveSplineTransport(
                inner_max_iter=60,
                outer_max_iter=20,
            ),
        ),
        (
            "boosted-hard-tree",
            lambda: BoostedHardTreeTransport(
                max_stages=8,
                max_leaves=32,
                max_fit_iterations=20,
            ),
        ),
        (
            "boosted-soft-tree",
            lambda: BoostedSoftTreeTransport(
                max_stages=4,
                max_depth=2,
                num_bins=8,
                max_epochs=200,
                patience=30,
                max_fit_iterations=20,
            ),
        ),
    ]


def _fit_models(training: Tensor) -> list[tuple[str, object, float]]:
    factories = _model_factories()
    for _, factory in factories:
        factory().fit(training)
    fitted = []
    for name, factory in factories:
        model = factory()
        start = time.perf_counter()
        model.fit(training)
        fitted.append((name, model, time.perf_counter() - start))
    return fitted


def run_benchmark(
    *,
    experiment: str,
    seed: int,
    training_samples: int,
    validation_samples: int,
) -> list[BenchmarkResult]:
    if experiment == "overfitting":
        sampler = sample_overfitting_mixture
        oracle = overfitting_mixture_log_prob
    elif experiment == "wavy":
        sampler = sample_wavy
        oracle = wavy_log_prob
    else:
        raise ValueError(f"unknown experiment: {experiment}")

    training = sampler(training_samples, seed)
    validation = sampler(validation_samples, seed + 10_000)
    oracle_nll = float((-oracle(validation)).mean().item())
    results = []
    for name, model, fit_seconds in _fit_models(training):
        with torch.no_grad():
            log_probability = model.log_prob(validation)
            reference = model(validation)
        validation_nll = float((-log_probability).mean().item())
        mean_rmse, covariance_rmse = _reference_diagnostics(reference)
        results.append(
            BenchmarkResult(
                experiment=experiment,
                model=name,
                seed=seed,
                training_samples=training_samples,
                validation_nll=validation_nll,
                oracle_nll=oracle_nll,
                excess_nll=validation_nll - oracle_nll,
                reference_mean_rmse=mean_rmse,
                reference_covariance_rmse=covariance_rmse,
                fit_seconds=fit_seconds,
                stages=(
                    model.stage_count_
                    if isinstance(
                        model,
                        (BoostedHardTreeTransport, BoostedSoftTreeTransport),
                    )
                    else None
                ),
                leaves=(
                    model.leaf_count_
                    if isinstance(
                        model,
                        (BoostedHardTreeTransport, BoostedSoftTreeTransport),
                    )
                    else None
                ),
            )
        )
    return results


def summarize(
    results: list[BenchmarkResult],
) -> list[dict[str, float | str | None]]:
    groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.experiment, result.model), []).append(result)
    summary = []
    for (experiment, model), rows in sorted(groups.items()):
        stages = [row.stages for row in rows if row.stages is not None]
        leaves = [row.leaves for row in rows if row.leaves is not None]
        summary.append(
            {
                "experiment": experiment,
                "model": model,
                "runs": len(rows),
                "validation_nll_mean": sum(row.validation_nll for row in rows)
                / len(rows),
                "excess_nll_mean": sum(row.excess_nll for row in rows)
                / len(rows),
                "reference_mean_rmse_mean": sum(
                    row.reference_mean_rmse for row in rows
                )
                / len(rows),
                "reference_covariance_rmse_mean": sum(
                    row.reference_covariance_rmse for row in rows
                )
                / len(rows),
                "fit_seconds_mean": sum(row.fit_seconds for row in rows)
                / len(rows),
                "stages_mean": (
                    sum(stages) / len(stages) if stages else None
                ),
                "leaves_mean": (
                    sum(leaves) / len(leaves) if leaves else None
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
    )
    arguments = parser.parse_args()
    results = []
    for seed in range(arguments.repetitions):
        results.extend(
            run_benchmark(
                experiment="overfitting",
                seed=seed,
                training_samples=25,
                validation_samples=arguments.validation_samples,
            )
        )
        results.extend(
            run_benchmark(
                experiment="wavy",
                seed=seed,
                training_samples=30,
                validation_samples=arguments.validation_samples,
            )
        )
    payload = {
        "source": (
            "MaxRamgraber/Adaptive-P-Spline-Triangular-Measure-Transport"
        ),
        "results": [asdict(result) for result in results],
        "summary": summarize(results),
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
