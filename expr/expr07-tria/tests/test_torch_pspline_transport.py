import torch

from expr07_tria import AdaptiveSplineTransport
from expr07_tria.torch_pspline_transport import _LinearTailBasis


def correlated_samples(count: int = 36) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    x0 = torch.randn(count, generator=generator, dtype=torch.float64)
    x1 = 0.4 * x0.square() + 0.25 * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    x2 = torch.sin(x1) + 0.2 * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.stack((x0, x1, x2), dim=1)


def test_analytical_basis_derivative_matches_finite_difference() -> None:
    points = torch.tensor(
        [-1.5, -0.83, -0.37, 0.11, 0.62, 0.91, 1.5],
        dtype=torch.float64,
    )
    step = 1e-6

    for degree in (1, 2, 3):
        real_knots = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64)
        knots = torch.cat(
            (
                real_knots[:1].repeat(degree),
                real_knots,
                real_knots[-1:].repeat(degree),
            )
        )
        basis = _LinearTailBasis(degree, knots)
        derivative = basis.derivative_design(points)
        finite_difference = (
            basis.design(points + step) - basis.design(points - step)
        ) / (2.0 * step)

        assert torch.allclose(
            derivative,
            finite_difference,
            atol=2e-8,
            rtol=2e-7,
        )


def test_local_spline_evaluation_matches_dense_design() -> None:
    points = torch.linspace(-1.5, 1.5, 31, dtype=torch.float64)
    generator = torch.Generator().manual_seed(11)

    for degree in (1, 2, 3):
        real_knots = torch.linspace(-1.0, 1.0, 6, dtype=torch.float64)
        knots = torch.cat(
            (
                real_knots[:1].repeat(degree),
                real_knots,
                real_knots[-1:].repeat(degree),
            )
        )
        basis = _LinearTailBasis(degree, knots)
        coefficients = torch.randn(
            basis.nbasis,
            generator=generator,
            dtype=torch.float64,
        )

        values, derivatives = basis.evaluate_and_derivative(
            points,
            coefficients,
        )
        cached_values = basis.evaluate_local(
            basis.local_design(points),
            coefficients,
        )

        assert torch.allclose(values, basis.design(points) @ coefficients)
        assert torch.allclose(cached_values, values)
        assert torch.allclose(
            derivatives,
            basis.derivative_design(points) @ coefficients,
        )


def test_forward_inverse_round_trip() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)

    transport.fit(samples, optimize_lambdas=False, lambda_initial=2.0)
    reference = transport.forward(samples)
    reconstructed = transport.inverse(reference)

    assert reference.shape == samples.shape
    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)
    assert torch.all(torch.isfinite(transport.aicc_))
    assert torch.all(transport.effective_dof_ > 0)
    assert torch.equal(transport(samples), reference)


def test_inverse_has_nonzero_interior_gradients() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)
    reference = transport.forward(samples[:4]).detach().requires_grad_(True)

    reconstructed = transport.inverse(reference)
    inverse_gradient = torch.autograd.grad(reconstructed.sum(), reference)[0]

    assert torch.all(torch.isfinite(inverse_gradient))
    assert torch.all(inverse_gradient > 0)


def test_conditional_inverse_has_gradients_for_condition_and_reference() -> None:
    samples = correlated_samples()
    transport = AdaptiveSplineTransport(
        k=3,
        skip_dimensions=1,
        inner_max_iter=60,
    )
    transport.fit(samples, optimize_lambdas=False)
    condition = samples[:4, :1].detach().requires_grad_(True)
    reference = transport.forward(samples[:4]).detach().requires_grad_(True)

    reconstructed = transport.conditional_inverse(condition, reference)
    condition_gradient, reference_gradient = torch.autograd.grad(
        reconstructed.sum(),
        (condition, reference),
    )

    assert torch.all(torch.isfinite(condition_gradient))
    assert torch.all(torch.isfinite(reference_gradient))


def test_fitted_diagonal_splines_are_monotone_with_linear_tails() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)

    for dimension_string, component in transport.components.items():
        dimension = int(dimension_string)
        assert component.coefficients is not None
        monotone_size = component.block_sizes[-1]
        coefficients = transport.reparameterize(
            component.coefficients[-monotone_size:]
        )
        basis = transport.bases[dimension_string]
        grid = torch.linspace(
            basis.left - 2.0,
            basis.right + 2.0,
            300,
            dtype=torch.float64,
        )
        values = basis.design(grid) @ coefficients
        differences = torch.diff(values)

        assert torch.all(differences > 0)

        low_points = torch.stack((basis.left - 2.0, basis.left - 1.0))
        high_points = torch.stack((basis.right + 1.0, basis.right + 2.0))
        low_slope = torch.diff(basis.design(low_points) @ coefficients)
        high_slope = torch.diff(basis.design(high_points) @ coefficients)

        assert low_slope > 0
        assert high_slope > 0


def test_conditional_inverse_preserves_reference_coordinates() -> None:
    samples = correlated_samples()
    transport = AdaptiveSplineTransport(
        k=3,
        skip_dimensions=1,
        inner_max_iter=60,
    )
    transport.fit(samples, optimize_lambdas=False)

    reference = transport.forward(samples)
    reconstructed = transport.conditional_inverse(samples[:, :1], reference)

    assert reference.shape == (samples.shape[0], 2)
    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)


def test_aicc_optimization_uses_differentiable_inner_solution() -> None:
    samples = correlated_samples(24)[:, :1]
    transport = AdaptiveSplineTransport(
        k=2,
        inner_max_iter=60,
        outer_max_iter=2,
    )

    transport.fit(samples, optimize_lambdas=True)

    log_lambda = transport.log_smoothing_[0]
    assert log_lambda.shape == (1,)
    assert torch.all(torch.isfinite(log_lambda))
    assert torch.all(log_lambda >= -10.0)
    assert torch.all(log_lambda <= 10.0)
    assert torch.all(torch.isfinite(transport.aicc_))


def test_float32_fit_uses_precision_appropriate_tolerances() -> None:
    samples = correlated_samples().to(torch.float32)[:, :2]
    transport = AdaptiveSplineTransport(
        k=3,
        inner_max_iter=100,
    )

    transport.fit(samples, lambda_initial=2.0, optimize_lambdas=False)
    reconstructed = transport.inverse(transport.forward(samples))

    assert torch.allclose(reconstructed, samples, atol=2e-5, rtol=2e-5)


def test_float32_fit_scales_to_production_sample_count() -> None:
    samples = torch.randn(
        64,
        11_008,
        generator=torch.Generator().manual_seed(19),
        dtype=torch.float32,
    )
    transport = AdaptiveSplineTransport(inner_max_iter=100)

    transport.fit(
        samples,
        sparsity=torch.eye(11_008, dtype=torch.bool),
        optimize_lambdas=False,
    )
    reconstructed = transport.inverse(transport.forward(samples))

    assert torch.allclose(reconstructed, samples, atol=2e-5, rtol=2e-5)


def test_batched_diagonal_state_gradients_and_dtype_semantics() -> None:
    samples = torch.randn(
        32,
        4,
        generator=torch.Generator().manual_seed(23),
        dtype=torch.float64,
    )
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(
        samples,
        sparsity=torch.eye(4, dtype=torch.bool),
        optimize_lambdas=False,
    )
    assert transport.diagonal is not None
    assert transport.sparsity_ is None
    assert "diagonal.coefficients" in transport.state_dict()
    assert not any(key.startswith("bases.") for key in transport.state_dict())

    reference = transport(samples).detach().requires_grad_(True)
    reconstructed = transport.inverse(reference)
    (gradient,) = torch.autograd.grad(reconstructed.sum(), reference)
    assert torch.all(torch.isfinite(gradient))
    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)

    restored = AdaptiveSplineTransport(k=3)
    restored.load_state_dict(transport.state_dict())
    assert restored.diagonal is not None
    assert torch.equal(restored(samples), transport(samples))

    transport = transport.to(dtype=torch.float32)
    result = transport(samples.to(torch.float32))
    assert result.dtype == torch.float32
    assert transport.coefficients_[0].dtype == torch.float32


def test_invalid_nontriangular_sparsity_is_rejected() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=2)
    sparsity = torch.tensor([[1, 1], [1, 1]])

    try:
        transport.fit(samples, sparsity=sparsity, optimize_lambdas=False)
    except ValueError as error:
        assert "lower-triangular" in str(error)
    else:
        raise AssertionError("nontriangular sparsity should be rejected")


def test_fitted_state_uses_module_device_and_dtype_semantics() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)

    state = transport.state_dict()
    assert "mean_" in state
    assert "scale_" in state
    assert "bases.0.knots" in state
    assert "components.0.coefficients" in state

    transport = transport.to(dtype=torch.float32)
    converted_samples = samples.to(torch.float32)
    result = transport(converted_samples)

    assert result.dtype == torch.float32
    assert transport.mean_.dtype == torch.float32
    assert transport.coefficients_[0].dtype == torch.float32


def test_state_dict_restores_a_fitted_transport() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)
    expected = transport(samples)

    restored = AdaptiveSplineTransport(k=3)
    restored.load_state_dict(transport.state_dict())

    assert torch.equal(restored(samples), expected)
    assert torch.equal(restored.aicc_, transport.aicc_)


def test_state_dict_restores_legacy_torchcurves_basis_state() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)
    state = transport.state_dict()
    for dimension in range(samples.shape[1]):
        knots = state[f"bases.{dimension}.knots"]
        state[f"bases.{dimension}.basis.knots"] = knots.clone()

    restored = AdaptiveSplineTransport(k=3)
    restored.load_state_dict(state)

    assert torch.equal(restored(samples), transport(samples))


def test_state_dict_restores_transport_nested_in_module() -> None:
    samples = correlated_samples()[:, :2]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)
    container = torch.nn.ModuleDict({"transport": transport})

    restored = torch.nn.ModuleDict(
        {"transport": AdaptiveSplineTransport(k=3)}
    )
    restored.load_state_dict(container.state_dict())

    restored_transport = restored["transport"]
    assert torch.equal(restored_transport(samples), transport(samples))


def test_state_dict_restores_fitted_architecture() -> None:
    samples = correlated_samples()
    transport = AdaptiveSplineTransport(
        k=3,
        degree=2,
        skip_dimensions=1,
        inner_max_iter=60,
    )
    transport.fit(samples, optimize_lambdas=False)

    restored = AdaptiveSplineTransport(k=3)
    restored.load_state_dict(transport.state_dict())

    assert restored.degree == 2
    assert restored.skip_dimensions == 1
    assert torch.equal(restored(samples), transport(samples))


def test_map_operations_reject_implicit_tensor_conversion() -> None:
    samples = correlated_samples()[:, :1]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)

    try:
        transport(samples.to(torch.float32))
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("forward should not silently convert input dtype")


def test_inverse_supports_inference_mode() -> None:
    samples = correlated_samples()[:, :1]
    transport = AdaptiveSplineTransport(k=3, inner_max_iter=60)
    transport.fit(samples, optimize_lambdas=False)

    with torch.inference_mode():
        reference = transport(samples)
        reconstructed = transport.inverse(reference)

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)
