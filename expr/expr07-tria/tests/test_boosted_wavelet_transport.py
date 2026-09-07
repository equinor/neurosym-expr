import torch

from expr07_tria import BoostedWaveletSplineTransport


def dependent_samples(count: int = 128) -> torch.Tensor:
    generator = torch.Generator().manual_seed(43)
    x0 = torch.randn(count, generator=generator, dtype=torch.float64)
    x1 = 0.8 * x0.square() + 0.15 * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    x2 = -0.7 * x0 + 0.2 * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    x3 = 0.5 * (x0 - x1) + 0.2 * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.stack((x0, x1, x2, x3), dim=1)


def fitted_transport() -> tuple[torch.Tensor, BoostedWaveletSplineTransport]:
    samples = dependent_samples()
    transport = BoostedWaveletSplineTransport(
        max_wavelet_level=2,
        max_parent_distance=4,
        max_learners_per_component=2,
        candidate_count=8,
        learning_rate=1.0,
        max_fit_iterations=20,
    ).fit(samples)
    return samples, transport


def test_boosting_adds_only_causal_learners_and_reduces_training_loss() -> None:
    samples, transport = fitted_transport()
    assert transport.learner_count_ > 0
    assert transport.learner_starts_ is not None
    assert transport.learner_widths_ is not None
    assert transport.learner_outputs_ is not None
    assert torch.all(
        transport.learner_starts_ + transport.learner_widths_
        <= transport.learner_outputs_
    )
    assert torch.all(transport.learner_outputs_ > 0)

    diagonal = transport.diagonal(samples)
    boosted = transport(samples)
    assert torch.sum(boosted.square()) < torch.sum(diagonal.square())
    assert torch.equal(boosted[:, 0], diagonal[:, 0])


def test_boosted_transport_jacobian_is_lower_triangular() -> None:
    samples, transport = fitted_transport()
    inputs = samples[:3].detach().requires_grad_(True)
    outputs = transport(inputs)

    for component in range(outputs.shape[1] - 1):
        (gradient,) = torch.autograd.grad(
            outputs[:, component].sum(),
            inputs,
            retain_graph=True,
        )
        assert torch.count_nonzero(gradient[:, component + 1 :]) == 0


def test_boosted_transport_round_trip_and_log_jacobian() -> None:
    samples, transport = fitted_transport()
    reference = transport(samples)
    reconstructed = transport.inverse(reference)
    random_reference = torch.randn_like(reference)
    random_round_trip = transport(transport.inverse(random_reference))
    log_jacobian = transport.log_abs_det_jacobian(samples)

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)
    assert torch.allclose(
        random_round_trip,
        random_reference,
        atol=1e-9,
        rtol=1e-9,
    )
    assert log_jacobian.shape == (samples.shape[0],)
    assert torch.all(torch.isfinite(log_jacobian))


def test_boosted_transport_inverse_preserves_gradients() -> None:
    samples, transport = fitted_transport()
    reference = transport(samples[:4]).detach().requires_grad_(True)

    reconstructed = transport.inverse(reference)
    (gradient,) = torch.autograd.grad(reconstructed.sum(), reference)

    assert torch.all(torch.isfinite(gradient))
    assert torch.linalg.vector_norm(gradient) > 0


def test_boosted_transport_conditional_inverse_and_state_dict() -> None:
    samples, transport = fitted_transport()
    reference = transport(samples)
    reconstructed = transport.conditional_inverse(
        samples[:, :1],
        reference[:, 1:],
    )
    restored = BoostedWaveletSplineTransport()
    restored.load_state_dict(transport.state_dict())
    legacy_state = dict(transport.state_dict())
    del legacy_state["learner_component_offsets_"]
    legacy_restored = BoostedWaveletSplineTransport()
    legacy_restored.load_state_dict(legacy_state)

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)
    assert torch.equal(restored(samples), reference)
    assert restored.learner_count_ == transport.learner_count_
    assert transport.learner_component_offsets_ is not None
    assert torch.equal(
        torch.diff(transport.learner_component_offsets_),
        torch.bincount(
            transport.learner_outputs_,
            minlength=samples.shape[1],
        ),
    )
    assert torch.equal(legacy_restored(samples), reference)

    converted = transport.to(dtype=torch.float32)
    converted_samples = samples.to(torch.float32)
    converted_reference = converted(converted_samples)
    assert converted.learner_component_offsets_.dtype == torch.long
    assert torch.allclose(
        converted.inverse(converted_reference),
        converted_samples,
        atol=1e-5,
        rtol=1e-5,
    )


def test_boosted_transport_inverse_supports_all_prefixes_and_gradients() -> None:
    samples, transport = fitted_transport()
    reference = transport(samples)

    for prefix_count in range(samples.shape[1]):
        reconstructed = transport.conditional_inverse(
            samples[:, :prefix_count],
            reference[:, prefix_count:],
        )
        assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)

    condition = samples[:4, :2].detach().requires_grad_(True)
    suffix = reference[:4, 2:].detach().requires_grad_(True)
    reconstructed = transport.conditional_inverse(condition, suffix)
    condition_gradient, suffix_gradient = torch.autograd.grad(
        reconstructed.sum(),
        (condition, suffix),
    )

    assert torch.all(torch.isfinite(condition_gradient))
    assert torch.all(torch.isfinite(suffix_gradient))
    assert torch.linalg.vector_norm(condition_gradient) > 0
    assert torch.linalg.vector_norm(suffix_gradient) > 0


def test_boosted_transport_empty_learners_and_dtype_transfer() -> None:
    samples = dependent_samples()
    transport = BoostedWaveletSplineTransport(
        max_learners_per_component=0,
        max_fit_iterations=20,
    ).fit(samples)
    reference = transport(samples)

    assert transport.learner_count_ == 0
    assert transport.learner_component_offsets_ is not None
    assert torch.equal(
        transport.learner_component_offsets_,
        torch.zeros(samples.shape[1] + 1, dtype=torch.long),
    )
    assert torch.equal(reference, transport.diagonal(samples))
    for prefix_count in range(samples.shape[1]):
        reconstructed = transport.conditional_inverse(
            samples[:, :prefix_count],
            reference[:, prefix_count:],
        )
        assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)

    transport = transport.to(dtype=torch.float32)
    converted_samples = samples.to(torch.float32)
    converted_reference = transport(converted_samples)
    assert transport.learner_component_offsets_.dtype == torch.long
    assert torch.allclose(
        transport.inverse(converted_reference),
        converted_samples,
        atol=1e-5,
        rtol=1e-5,
    )


def test_block_transport_batches_dependencies_and_inverse() -> None:
    samples = dependent_samples()
    transport = BoostedWaveletSplineTransport(
        block_size=2,
        max_wavelet_level=2,
        max_parent_distance=4,
        max_learners_per_component=2,
        candidate_count=8,
        learning_rate=1.0,
        max_fit_iterations=20,
    ).fit(samples)
    assert transport.learner_outputs_ is not None
    assert transport.learner_starts_ is not None
    assert transport.learner_widths_ is not None
    dependency_boundaries = (
        torch.div(
            transport.learner_outputs_,
            transport.block_size,
            rounding_mode="floor",
        )
        * transport.block_size
    )
    assert transport.learner_count_ > 0
    assert torch.all(
        transport.learner_starts_ + transport.learner_widths_
        <= dependency_boundaries
    )

    reference = transport(samples)
    for prefix_count in range(samples.shape[1]):
        condition = samples[:, :prefix_count]
        reconstructed = transport.conditional_inverse(
            condition,
            reference[:, prefix_count:],
        )
        assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)

    inputs = samples[:4].detach().requires_grad_(True)
    outputs = transport(inputs)
    (gradient,) = torch.autograd.grad(outputs[:, 3].sum(), inputs)
    assert torch.count_nonzero(gradient[:, 2]) == 0
    assert torch.count_nonzero(gradient[:, 3]) > 0

    restored = BoostedWaveletSplineTransport()
    restored.load_state_dict(transport.state_dict())
    assert restored.block_size == 2
    assert torch.equal(restored(samples), reference)


def test_leading_component_exceedance_sampling() -> None:
    _, transport = fitted_transport()
    threshold = 0.5
    samples = transport.sample_exceedance(
        component=0,
        threshold=threshold,
        sample_count=256,
        generator=torch.Generator().manual_seed(47),
    )

    assert samples.shape == (256, 4)
    assert torch.all(samples[:, 0] > threshold)

    try:
        transport.sample_exceedance(
            component=1,
            threshold=threshold,
            sample_count=1,
        )
    except NotImplementedError:
        pass
    else:
        raise AssertionError("later-component exceedance must be rejected")


def test_inverse_retains_full_support_of_crossing_wavelets() -> None:
    generator = torch.Generator().manual_seed(91)
    x0 = torch.randn(128, generator=generator, dtype=torch.float64)
    x1 = torch.randn(128, generator=generator, dtype=torch.float64)
    samples = torch.stack(
        (
            x0,
            x1,
            x0
            - x1
            + 0.05
            * torch.randn(128, generator=generator, dtype=torch.float64),
            torch.randn(128, generator=generator, dtype=torch.float64),
        ),
        dim=1,
    )
    transport = BoostedWaveletSplineTransport(
        max_parent_distance=1,
        max_learners_per_component=2,
        candidate_count=8,
        learning_rate=1.0,
        max_fit_iterations=10,
    ).fit(samples)
    reference = transport(samples)

    assert transport.learner_starts_ is not None
    assert transport.learner_widths_ is not None
    assert transport.learner_outputs_ is not None
    crossing = (
        (transport.learner_starts_ == 0)
        & (transport.learner_widths_ == 2)
        & (transport.learner_outputs_ == 2)
    )
    assert torch.any(crossing)
    assert torch.allclose(
        transport.inverse(reference),
        samples,
        atol=1e-9,
        rtol=1e-9,
    )
