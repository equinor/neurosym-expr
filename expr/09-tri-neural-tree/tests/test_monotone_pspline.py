import torch

from nine import BatchedDiagonalSplineTransport


def test_diagonal_spline_is_monotone_and_invertible_with_linear_tails() -> None:
    generator = torch.Generator().manual_seed(11)
    samples = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    transport = BatchedDiagonalSplineTransport(max_fit_iterations=20).fit(samples)
    reference = transport(samples)
    reconstructed = transport.inverse(reference)

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)
    assert transport.left_derivatives_ is not None
    assert transport.right_derivatives_ is not None
    assert transport.log_smoothing_ is not None
    assert torch.all(transport.left_derivatives_ > 0)
    assert torch.all(transport.right_derivatives_ > 0)
    assert set(transport.log_smoothing_.tolist()) <= {-2.0, 2.0, 6.0}

    inputs = samples[:4].detach().requires_grad_(True)
    outputs = transport(inputs)
    (gradient,) = torch.autograd.grad(outputs.sum(), inputs)
    assert torch.all(gradient > 0)
