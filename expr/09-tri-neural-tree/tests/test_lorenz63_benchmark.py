import torch

from nine.lorenz63_benchmark import (
    ensemble_crps,
    generate_truth_and_observations,
    lorenz63_dynamics,
    rk4,
)


def test_lorenz_dynamics_rk4_and_generation() -> None:
    state = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    derivative = lorenz63_dynamics(state)
    advanced = rk4(state)
    truth, observations = generate_truth_and_observations(
        seed=4,
        steps=5,
        spinup_steps=3,
    )

    assert derivative.shape == state.shape
    assert advanced.shape == state.shape
    assert torch.all(torch.isfinite(advanced))
    assert truth.shape == observations.shape == (8, 3)
    assert torch.equal(
        truth,
        generate_truth_and_observations(
            seed=4,
            steps=5,
            spinup_steps=3,
        )[0],
    )


def test_ensemble_crps_is_zero_for_perfect_degenerate_ensemble() -> None:
    truth = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ensemble = truth.expand(8, -1)
    assert torch.equal(ensemble_crps(ensemble, truth), torch.zeros_like(truth))
