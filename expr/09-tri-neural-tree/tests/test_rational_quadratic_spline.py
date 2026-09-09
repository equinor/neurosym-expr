import pytest
import torch

from nine.rational_quadratic_spline import (
    identity_parameters,
    rational_quadratic_spline,
)


def test_identity_parameters_produce_exact_identity_with_zero_logdet() -> None:
    inputs = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64).reshape(101, 1)
    widths, heights, derivatives = identity_parameters(
        8, (1,), dtype=inputs.dtype
    )

    outputs, logabsdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives, tail_bound=3.0
    )

    assert torch.allclose(outputs, inputs, atol=2e-15, rtol=0.0)
    assert torch.allclose(logabsdet, torch.zeros_like(inputs), atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("dtype, tolerance", [(torch.float32, 2e-5), (torch.float64, 1e-10)])
def test_forward_inverse_round_trip_for_per_sample_parameters(
    dtype: torch.dtype, tolerance: float
) -> None:
    generator = torch.Generator().manual_seed(19)
    inputs = torch.empty(7, 3, dtype=dtype).uniform_(-4.0, 4.0, generator=generator)
    widths = torch.randn(7, 3, 6, dtype=dtype, generator=generator)
    heights = torch.randn(7, 3, 6, dtype=dtype, generator=generator)
    derivatives = torch.randn(7, 3, 5, dtype=dtype, generator=generator)

    outputs, forward_logdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives, tail_bound=2.5
    )
    reconstructed, inverse_logdet = rational_quadratic_spline(
        outputs, widths, heights, derivatives, inverse=True, tail_bound=2.5
    )

    assert reconstructed.dtype == dtype
    assert torch.allclose(reconstructed, inputs, atol=tolerance, rtol=tolerance)
    assert torch.allclose(
        inverse_logdet, -forward_logdet, atol=4 * tolerance, rtol=4 * tolerance
    )


def test_batched_component_parameters_broadcast_over_samples() -> None:
    inputs = torch.tensor(
        [[-1.5, -0.5, 0.5], [1.5, 0.25, -0.25]], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(5)
    widths = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    heights = torch.randn(1, 3, 5, dtype=torch.float64, generator=generator)
    derivatives = torch.randn(3, 4, dtype=torch.float64, generator=generator)

    outputs, logabsdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives
    )

    assert outputs.shape == inputs.shape
    assert logabsdet.shape == inputs.shape
    for row in range(inputs.shape[0]):
        expected, expected_logdet = rational_quadratic_spline(
            inputs[row], widths, heights[0], derivatives
        )
        assert torch.allclose(outputs[row], expected)
        assert torch.allclose(logabsdet[row], expected_logdet)


def test_logabsdet_matches_autograd_and_tails_are_identity() -> None:
    inputs = torch.tensor(
        [-4.0, -2.0, -0.4, 0.7, 2.0, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    widths = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
    heights = torch.tensor([-0.3, 0.5, -0.1, 0.8], dtype=torch.float64)
    derivatives = torch.tensor([0.1, -0.6, 0.9], dtype=torch.float64)

    outputs, logabsdet = rational_quadratic_spline(
        inputs, widths, heights, derivatives, tail_bound=2.0
    )
    (gradient,) = torch.autograd.grad(outputs.sum(), inputs)

    assert torch.all(gradient > 0)
    assert torch.allclose(logabsdet, torch.log(gradient), atol=1e-11, rtol=1e-11)
    assert torch.equal(outputs[[0, -1]], inputs.detach()[[0, -1]])
    assert torch.equal(logabsdet[[0, -1]], torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize(
    "parameter_shapes, message",
    [
        (((2, 4), (2, 5), (2, 3)), "same final dimension"),
        (((2, 4), (2, 4), (2, 4)), "num_bins - 1"),
        (((3, 4), (3, 4), (3, 3)), "broadcastable"),
    ],
)
def test_rejects_invalid_parameter_shapes(
    parameter_shapes: tuple[tuple[int, ...], ...], message: str
) -> None:
    inputs = torch.zeros(2)
    parameters = [torch.zeros(shape) for shape in parameter_shapes]

    with pytest.raises(ValueError, match=message):
        rational_quadratic_spline(inputs, *parameters)


def test_rejects_invalid_dtype_values_and_minimum_sizes() -> None:
    inputs = torch.zeros(2, dtype=torch.float64)
    widths = torch.zeros(2, 4, dtype=torch.float64)
    heights = torch.zeros_like(widths)
    derivatives = torch.zeros(2, 3, dtype=torch.float64)

    with pytest.raises(TypeError, match="same dtype"):
        rational_quadratic_spline(inputs, widths.float(), heights, derivatives)
    with pytest.raises(ValueError, match="finite"):
        rational_quadratic_spline(inputs, widths.fill_(float("nan")), heights, derivatives)
    with pytest.raises(ValueError, match=r"num_bins \* min_bin_width"):
        rational_quadratic_spline(
            inputs, widths.zero_(), heights, derivatives, tail_bound=1.0, min_bin_width=0.5
        )
