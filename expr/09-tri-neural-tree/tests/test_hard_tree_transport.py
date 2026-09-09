from __future__ import annotations

import pytest
import torch

from nine import BoostedHardTreeTransport, HardTreeSpline


def piecewise_scale_samples(count: int = 96) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    x0 = torch.randn(count, generator=generator, dtype=torch.float64)
    noise = torch.randn(count, generator=generator, dtype=torch.float64)
    scale = torch.where(x0 <= 0, 0.15, 2.0)
    return torch.stack((x0, scale * noise), dim=1)


@pytest.fixture(scope="module")
def fitted_transport() -> tuple[torch.Tensor, BoostedHardTreeTransport]:
    samples = piecewise_scale_samples()
    transport = BoostedHardTreeTransport(
        max_stages=1,
        max_leaves=3,
        max_fit_iterations=10,
    ).fit(samples)
    return samples, transport


def test_model_selects_causal_hard_tree(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    _, transport = fitted_transport

    assert transport.stage_count_ == 1
    assert transport.leaf_count_ >= 2
    tree = transport.stages[0].components[1]
    assert not tree.is_identity
    assert torch.all(tree.feature_starts_ + tree.feature_widths_ <= 1)
    assert transport.stages[0].components[0].is_identity


def test_forward_inverse_logdet_and_triangular_jacobian(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    samples, transport = fitted_transport
    reference = transport(samples)

    assert torch.allclose(
        transport.inverse(reference),
        samples,
        atol=1e-9,
        rtol=1e-9,
    )
    assert torch.all(torch.isfinite(transport.log_prob(samples)))

    inputs = samples[:3].detach().requires_grad_(True)
    outputs = transport(inputs)
    jacobians = [
        torch.autograd.grad(
            outputs[row, component],
            inputs,
            retain_graph=True,
        )[0][row]
        for row in range(inputs.shape[0])
        for component in range(inputs.shape[1])
    ]
    for row in range(inputs.shape[0]):
        first, second = jacobians[2 * row : 2 * row + 2]
        assert first[0] > 0
        assert first[1] == 0
        assert second[1] > 0

    full_logdet = torch.stack(
        [
            torch.log(jacobians[2 * row][0])
            + torch.log(jacobians[2 * row + 1][1])
            for row in range(inputs.shape[0])
        ]
    )
    assert torch.allclose(
        transport.log_abs_det_jacobian(inputs),
        full_logdet,
        atol=1e-9,
        rtol=1e-9,
    )


def test_conditional_inverse_preserves_prefix(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    samples, transport = fitted_transport
    reference = transport(samples)
    reconstructed = transport.conditional_inverse(
        samples[:, :1],
        reference[:, 1:],
    )

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)


def test_state_dict_and_dtype_round_trip(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    samples, transport = fitted_transport
    reference = transport(samples)
    restored = BoostedHardTreeTransport()
    restored.load_state_dict(transport.state_dict())

    assert torch.equal(restored(samples), reference)
    converted = restored.to(dtype=torch.float32)
    converted_samples = samples.to(torch.float32)
    converted_reference = converted(converted_samples)
    assert torch.allclose(
        converted.inverse(converted_reference),
        converted_samples,
        atol=1e-5,
        rtol=1e-5,
    )

    converted_before_load = BoostedHardTreeTransport().to(dtype=torch.float32)
    converted_before_load.load_state_dict(transport.state_dict())
    assert converted_before_load.diagonal.mean_.dtype == torch.float32
    assert converted_before_load.stages[0].conditioning_mean_.dtype == torch.float32
    assert converted_before_load(converted_samples).dtype == torch.float32


def test_stage_empty_prefix_and_tree_refit(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    samples, transport = fitted_transport
    diagonal_reference = transport.diagonal(samples)
    stage = transport.stages[0]
    stage_reference = stage(diagonal_reference)

    assert torch.allclose(
        stage.conditional_inverse(
            diagonal_reference.new_empty((samples.shape[0], 0)),
            stage_reference,
        ),
        diagonal_reference,
        atol=1e-9,
        rtol=1e-9,
    )

    tree = HardTreeSpline(max_leaves=3).fit(
        diagonal_reference[:, :1],
        diagonal_reference[:, 1:],
    )
    assert not tree.is_identity
    tree.fit(
        samples.new_empty((samples.shape[0], 0)),
        diagonal_reference[:, 1:],
    )
    assert tree.is_identity
    assert torch.equal(
        tree(
            samples.new_empty((samples.shape[0], 0)),
            diagonal_reference[:, 1:],
        ),
        diagonal_reference[:, 1:],
    )


def test_independent_gaussian_data_rejects_tree_stage() -> None:
    samples = torch.randn(
        48,
        2,
        generator=torch.Generator().manual_seed(19),
        dtype=torch.float64,
    )
    transport = BoostedHardTreeTransport(
        max_stages=2,
        max_leaves=3,
        max_fit_iterations=10,
    ).fit(samples)

    assert transport.stage_count_ == 0
    assert transport.stopping_reason_ == "candidate_stage_is_identity"


def test_tree_split_ranking_is_stable_under_large_target_offset() -> None:
    generator = torch.Generator().manual_seed(41)
    conditioning = torch.randn(96, 4, generator=generator)
    noise = torch.randn(96, generator=generator)
    scale = torch.where(conditioning[:, 2] < 0, 0.1, 3.0)
    targets = (10_000.0 + scale * noise).unsqueeze(1)

    tree = HardTreeSpline(max_leaves=2, max_fit_iterations=5).fit(
        conditioning,
        targets,
        candidate_starts=torch.arange(conditioning.shape[1]),
        candidate_widths=torch.ones(conditioning.shape[1], dtype=torch.long),
    )

    assert not tree.is_identity
    assert tree.feature_starts_[tree.node_features_[0]] == 2


def test_loading_smaller_checkpoint_removes_stale_stages(
    fitted_transport: tuple[torch.Tensor, BoostedHardTreeTransport],
) -> None:
    samples, fitted = fitted_transport
    empty = BoostedHardTreeTransport(max_stages=0).fit(samples)
    restored = BoostedHardTreeTransport()
    restored.load_state_dict(fitted.state_dict())
    assert restored.stage_count_ == 1

    restored.load_state_dict(empty.state_dict())
    assert restored.stage_count_ == 0
    assert torch.equal(restored(samples), empty(samples))


def test_large_transport_stores_only_screened_nonidentity_components() -> None:
    sample_count = 50
    dimension_count = 513
    generator = torch.Generator().manual_seed(5)
    samples = torch.randn(
        sample_count,
        dimension_count,
        generator=generator,
    )
    scale = torch.where(samples[:, 0] < 0, 0.03, 4.0)
    samples[:, -1] = scale * torch.randn(sample_count, generator=generator)

    transport = BoostedHardTreeTransport(
        max_stages=1,
        max_leaves=3,
        max_fit_iterations=5,
    ).fit(samples)

    assert transport.stage_count_ == 1
    stage = transport.stages[0]
    assert torch.equal(
        stage.component_indices_,
        torch.tensor([dimension_count - 1]),
    )
    assert len(stage.components) == 1
    reference = transport(samples)
    assert torch.allclose(
        transport.inverse(reference),
        samples,
        atol=2e-5,
        rtol=2e-5,
    )

    restored = BoostedHardTreeTransport()
    restored.load_state_dict(transport.state_dict())
    assert torch.equal(restored(samples), reference)


@pytest.mark.parametrize("max_parent_distance", [2, 4, 8, 16])
def test_sparse_parent_window_selects_only_local_features(
    max_parent_distance: int,
) -> None:
    sample_count = 50
    dimension_count = 513
    generator = torch.Generator().manual_seed(31)
    samples = torch.randn(
        sample_count,
        dimension_count,
        generator=generator,
    )
    scale = torch.where(samples[:, -2] < 0, 0.03, 4.0)
    samples[:, -1] = scale * torch.randn(sample_count, generator=generator)

    transport = BoostedHardTreeTransport(
        max_stages=1,
        max_leaves=3,
        max_fit_iterations=5,
        max_parent_distance=max_parent_distance,
    ).fit(samples)

    assert transport.stage_count_ == 1
    stage = transport.stages[0]
    assert stage.max_parent_distance == max_parent_distance
    for component_index, tree in zip(
        stage.component_indices_.tolist(),
        stage.components,
        strict=True,
    ):
        assert torch.all(
            tree.feature_starts_ >= component_index - max_parent_distance
        )
        assert torch.all(tree.feature_starts_ < component_index)

    restored = BoostedHardTreeTransport()
    restored.load_state_dict(transport.state_dict())
    assert restored.max_parent_distance == max_parent_distance
    assert restored.stages[0].max_parent_distance == max_parent_distance


def test_sparse_parent_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_parent_distance must be positive"):
        BoostedHardTreeTransport(max_parent_distance=0)
