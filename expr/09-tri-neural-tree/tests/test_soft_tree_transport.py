from __future__ import annotations

import copy
from collections import OrderedDict

import pytest
import torch

from nine import (
    BoostedSoftTreeTransport,
    SoftTreeRationalQuadraticSpline,
    SoftTreeTransportStage,
)


def piecewise_scale_samples(count: int = 96) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    first = torch.randn(count, generator=generator, dtype=torch.float64)
    scale = torch.where(first <= 0, 0.15, 2.0)
    second = scale * torch.randn(count, generator=generator, dtype=torch.float64)
    return torch.stack((first, second), dim=1)


def multi_component_samples(count: int = 96) -> torch.Tensor:
    generator = torch.Generator().manual_seed(23)
    first = torch.randn(count, generator=generator, dtype=torch.float64)
    second_scale = torch.where(first <= 0, 0.15, 2.0)
    second = second_scale * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    third_scale = torch.where(second <= 0, 0.2, 1.8)
    third = third_scale * torch.randn(
        count,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.stack((first, second, third), dim=1)


@pytest.fixture(scope="module")
def fitted_transport() -> tuple[torch.Tensor, BoostedSoftTreeTransport]:
    samples = piecewise_scale_samples()
    transport = BoostedSoftTreeTransport(
        max_stages=1,
        max_depth=1,
        num_bins=6,
        max_epochs=80,
        patience=15,
    ).fit(samples)
    return samples, transport


def test_soft_tree_probabilities_and_gradients() -> None:
    samples = piecewise_scale_samples()
    conditioning = samples[:, :1]
    targets = samples[:, 1:]
    tree = SoftTreeRationalQuadraticSpline(
        max_depth=1,
        num_bins=6,
        max_epochs=40,
        patience=10,
        validation_fraction=0.0,
    ).fit(conditioning, targets)

    probabilities = tree.leaf_probabilities(conditioning)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(samples.shape[0], dtype=samples.dtype),
    )
    mapped, log_derivative = tree.forward_with_log_derivative(
        conditioning,
        targets,
    )
    loss = (0.5 * mapped.square() - log_derivative).mean()
    loss.backward()
    for parameter in tree.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_forward_inverse_logdet_and_triangular_jacobian(
    fitted_transport: tuple[torch.Tensor, BoostedSoftTreeTransport],
) -> None:
    samples, transport = fitted_transport
    assert transport.stage_count_ == 1
    assert transport.leaf_count_ == 2
    reference = transport(samples)
    combined_reference, combined_logdet = transport.forward_with_logdet(samples)
    assert torch.equal(combined_reference, reference)
    assert torch.equal(
        combined_logdet,
        transport.log_abs_det_jacobian(samples),
    )
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

    expected = torch.stack(
        [
            torch.log(jacobians[2 * row][0])
            + torch.log(jacobians[2 * row + 1][1])
            for row in range(inputs.shape[0])
        ]
    )
    assert torch.allclose(
        transport.log_abs_det_jacobian(inputs),
        expected,
        atol=1e-9,
        rtol=1e-9,
    )


def test_conditional_inverse_preserves_prefix(
    fitted_transport: tuple[torch.Tensor, BoostedSoftTreeTransport],
) -> None:
    samples, transport = fitted_transport
    reference = transport(samples)
    reconstructed = transport.conditional_inverse(
        samples[:, :1],
        reference[:, 1:],
    )

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)


def test_state_dict_and_dtype_round_trip(
    fitted_transport: tuple[torch.Tensor, BoostedSoftTreeTransport],
) -> None:
    samples, transport = fitted_transport
    expected = transport(samples)
    restored = BoostedSoftTreeTransport()
    restored.load_state_dict(transport.state_dict())

    assert torch.equal(restored(samples), expected)
    converted = restored.to(dtype=torch.float32)
    converted_samples = samples.float()
    converted_reference = converted(converted_samples)
    assert torch.allclose(
        converted.inverse(converted_reference),
        converted_samples,
        atol=2e-5,
        rtol=2e-5,
    )

    converted_before_load = BoostedSoftTreeTransport().to(dtype=torch.float32)
    converted_before_load.load_state_dict(transport.state_dict())
    assert converted_before_load(converted_samples).dtype == torch.float32


def test_one_dimensional_stage_is_identity() -> None:
    values = torch.randn(
        32,
        1,
        generator=torch.Generator().manual_seed(8),
        dtype=torch.float64,
    )
    stage = SoftTreeTransportStage(max_epochs=10).fit(values)

    assert stage.is_identity
    assert stage.stopping_reason_ == "candidate_stage_is_identity"
    assert torch.equal(stage(values), values)


def test_tree_without_conditioning_is_identity() -> None:
    targets = torch.randn(
        32,
        1,
        generator=torch.Generator().manual_seed(13),
        dtype=torch.float64,
    )
    tree = SoftTreeRationalQuadraticSpline(max_epochs=10).fit(
        targets[:, :0],
        targets,
    )

    assert tree.is_identity
    assert tree.stopping_reason_ == "no_causal_features"


def test_independent_gaussian_data_rejects_soft_stage() -> None:
    samples = torch.randn(
        48,
        2,
        generator=torch.Generator().manual_seed(19),
        dtype=torch.float64,
    )
    transport = BoostedSoftTreeTransport(
        max_stages=1,
        max_depth=1,
        num_bins=6,
        max_epochs=80,
        patience=15,
    ).fit(samples)

    assert transport.stage_count_ == 0
    assert transport.stopping_reason_ == "no_global_criterion_improvement"


def test_parent_window_restricts_tree_features() -> None:
    values = piecewise_scale_samples()
    stage = SoftTreeTransportStage(
        max_depth=1,
        num_bins=6,
        max_epochs=40,
        patience=10,
        max_parent_distance=1,
    ).fit(values)

    assert not stage.is_identity
    tree = stage.components[0]
    component = int(stage.component_indices_[0].item())
    assert torch.all(tree.feature_starts_ >= component - 1)
    assert torch.all(tree.feature_starts_ + tree.feature_widths_ <= component)


def test_stage_batches_multiple_components() -> None:
    values = multi_component_samples()
    stage = SoftTreeTransportStage(
        max_depth=1,
        num_bins=6,
        max_epochs=60,
        patience=15,
        validation_fraction=0.0,
    ).fit(values)

    assert len(stage.components) == 2
    reference, logdet = stage.forward_with_logdet(values)
    assert torch.allclose(stage.inverse(reference), values, atol=1e-9, rtol=1e-9)
    assert torch.equal(stage(values), reference)

    standardized = (
        values - stage.conditioning_mean_
    ) / stage.conditioning_scale_
    expected = values.clone()
    expected_logdet = values.new_zeros(values.shape[0])
    for component, tree in zip(
        stage._component_indices,
        stage.components,
        strict=True,
    ):
        mapped, component_logdet = tree.forward_with_log_derivative(
            standardized[:, :component],
            values[:, component : component + 1],
        )
        expected[:, component : component + 1] = mapped
        expected_logdet += component_logdet[:, 0]

    assert torch.allclose(reference, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(logdet, expected_logdet, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("prefix_size", [0, 1, 2, 3])
def test_stage_prefix_forward_matches_full_forward(
    prefix_size: int,
) -> None:
    values = multi_component_samples()
    stage = SoftTreeTransportStage(
        max_depth=1,
        num_bins=6,
        max_epochs=60,
        patience=15,
        validation_fraction=0.0,
    ).fit(values)

    assert torch.equal(
        stage.forward_prefix(values[:, :prefix_size]),
        stage(values)[:, :prefix_size],
    )


def test_stage_uses_contiguous_batched_parameters() -> None:
    values = multi_component_samples()
    stage = SoftTreeTransportStage(
        max_depth=1,
        num_bins=6,
        max_epochs=60,
        patience=15,
        validation_fraction=0.0,
    ).fit(values)

    assert stage.node_thresholds.shape[:2] == (2, 1)
    assert stage.leaf_width_logits.shape[:2] == (2, 2)
    assert stage.feature_starts_.shape == (2, 1)


@pytest.mark.parametrize("prefix_size", [0, 1, 2])
def test_conditional_inverse_across_multiple_stages(
    fitted_transport: tuple[torch.Tensor, BoostedSoftTreeTransport],
    prefix_size: int,
) -> None:
    samples, fitted = fitted_transport
    transport = copy.deepcopy(fitted)
    transport.stages.append(copy.deepcopy(transport.stages[0]))
    reference = transport(samples)

    reconstructed = transport.conditional_inverse(
        samples[:, :prefix_size],
        reference[:, prefix_size:],
    )

    assert torch.allclose(reconstructed, samples, atol=1e-9, rtol=1e-9)


def test_legacy_component_state_dict_loads(
    fitted_transport: tuple[torch.Tensor, BoostedSoftTreeTransport],
) -> None:
    samples, fitted = fitted_transport
    state = OrderedDict(fitted.state_dict())
    prefix = "stages.0."
    extra = dict(state[f"{prefix}_extra_state"])
    extra.pop("representation_version")
    extra.pop("component_training_nll")
    extra.pop("component_validation_nll")
    state[f"{prefix}_extra_state"] = extra
    stage = fitted.stages[0]
    parameter_names = (
        "node_thresholds",
        "node_raw_temperatures",
        "leaf_width_logits",
        "leaf_height_logits",
        "leaf_derivative_logits",
    )
    parameters = {
        name: state.pop(f"{prefix}{name}")
        for name in parameter_names
    }
    for index, component in enumerate(stage.components):
        for name, parameter in parameters.items():
            state[f"{prefix}components.{index}.{name}"] = parameter[index]
        state[f"{prefix}components.{index}.feature_starts_"] = (
            stage.feature_starts_[index]
        )
        state[f"{prefix}components.{index}.feature_widths_"] = (
            stage.feature_widths_[index]
        )
        state[f"{prefix}components.{index}.node_features_"] = (
            component.node_features_
        )
        state[f"{prefix}components.{index}._extra_state"] = {
            "condition_dimension": component.condition_dimension_,
            "training_nll": component.training_nll_,
            "validation_nll": component.validation_nll_,
        }
    state.pop(f"{prefix}feature_starts_")
    state.pop(f"{prefix}feature_widths_")
    state.pop(f"{prefix}component_indices_")

    restored = BoostedSoftTreeTransport()
    restored.load_state_dict(state)

    assert torch.equal(restored(samples), fitted(samples))


def test_legacy_identity_stage_state_dict_loads() -> None:
    values = torch.randn(
        32,
        1,
        generator=torch.Generator().manual_seed(31),
        dtype=torch.float64,
    )
    stage = SoftTreeTransportStage(max_epochs=10).fit(values)
    state = OrderedDict(stage.state_dict())
    extra = dict(state["_extra_state"])
    extra.pop("representation_version")
    extra.pop("component_training_nll")
    extra.pop("component_validation_nll")
    state["_extra_state"] = extra
    for name in (
        "node_thresholds",
        "node_raw_temperatures",
        "leaf_width_logits",
        "leaf_height_logits",
        "leaf_derivative_logits",
        "feature_starts_",
        "feature_widths_",
    ):
        state.pop(name)

    restored = SoftTreeTransportStage()
    restored.load_state_dict(state)

    assert restored.is_identity
    assert torch.equal(restored(values), values)


def test_joint_fine_tuning_preserves_selection_score() -> None:
    values = piecewise_scale_samples()
    transport = BoostedSoftTreeTransport(
        max_stages=1,
        max_depth=1,
        num_bins=6,
        max_epochs=60,
        patience=15,
        fine_tune_epochs=10,
    ).fit(values)

    assert transport.stage_count_ == 1
    assert transport.fine_tune_improvement_ >= 0.0
