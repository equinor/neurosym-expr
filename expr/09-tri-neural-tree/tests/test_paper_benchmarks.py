import torch

from nine.paper_benchmarks import (
    overfitting_mixture_log_prob,
    sample_overfitting_mixture,
    sample_wavy,
    wavy_log_prob,
)


def test_paper_benchmark_generators_are_reproducible_and_finite() -> None:
    mixture = sample_overfitting_mixture(32, 7)
    wavy = sample_wavy(32, 7)

    assert torch.equal(mixture, sample_overfitting_mixture(32, 7))
    assert torch.equal(wavy, sample_wavy(32, 7))
    assert mixture.shape == (32, 1)
    assert wavy.shape == (32, 2)
    assert torch.all(torch.isfinite(overfitting_mixture_log_prob(mixture)))
    assert torch.all(torch.isfinite(wavy_log_prob(wavy)))
