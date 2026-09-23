import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from nine.online_steering import (
    LayerSteeringHook,
    SteeringSearchConfig,
    contrastive_basis,
    optimize_steering_vector,
    project_to_ball,
    shrinkage_covariance,
)
from nine.online_steering_cli import (
    OnlineSteeringExperiment,
    generated_entropies,
    generated_lengths,
    resolve_layer,
    token_entropy,
)


def test_shrinkage_covariance_is_positive_with_few_samples() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 3.0, 5.0],
            [3.0, 0.0, 3.0, 6.0],
        ]
    )

    covariance = shrinkage_covariance(values)

    assert covariance.shape == (4, 4)
    assert torch.linalg.eigvalsh(covariance).min() > 0.0


def test_contrastive_basis_is_low_rank_and_orthonormal() -> None:
    activations = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [-2.0, 0.0, 1.0, 0.0],
            [-3.0, 0.0, 0.0, 1.0],
        ]
    )
    scores = torch.tensor([2.0, 1.0, -1.0, -2.0])

    basis, contrast = contrastive_basis(
        activations,
        scores,
        rank=2,
    )

    assert basis.shape == (4, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-6)
    assert contrast[0] > 0.0
    assert torch.linalg.vector_norm(basis.T @ contrast) > 0.0


def test_contrastive_basis_does_not_use_unsupported_null_directions() -> None:
    activations = torch.ones(4, 6)
    scores = torch.zeros(4)

    basis, contrast = contrastive_basis(activations, scores, rank=4)

    assert basis.shape == (6, 1)
    assert torch.equal(basis[:, 0], torch.eye(6)[:, 0])
    assert torch.count_nonzero(contrast) == 0


def test_project_to_ball_preserves_small_and_clips_large_rows() -> None:
    values = torch.tensor([[0.3, 0.4], [3.0, 4.0]])

    projected = project_to_ball(values, 1.0)

    assert torch.allclose(projected[0], values[0])
    assert torch.allclose(
        torch.linalg.vector_norm(projected, dim=1),
        torch.tensor([0.5, 1.0]),
    )


def test_layer_hook_changes_only_last_position_and_tracks_originals() -> None:
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(4))
    original_weight = layer.weight.detach().clone()
    hook = LayerSteeringHook()
    handle = layer.register_forward_hook(hook)
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    vectors = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]
    )
    hook.set_vectors(vectors, strength=0.5)

    try:
        result = layer(inputs)
    finally:
        handle.remove()

    assert torch.equal(result[:, :-1], inputs[:, :-1])
    assert torch.equal(result[:, -1], inputs[:, -1] + 0.5 * vectors)
    assert torch.equal(hook.activation_means(), inputs[:, -1])
    assert torch.equal(layer.weight, original_weight)


def test_zero_steering_exactly_preserves_layer_output() -> None:
    layer = nn.Linear(3, 3)
    inputs = torch.randn(2, 4, 3)
    expected = layer(inputs)
    hook = LayerSteeringHook()
    hook.set_vectors(torch.zeros(2, 3))
    handle = layer.register_forward_hook(hook)

    try:
        actual = layer(inputs)
    finally:
        handle.remove()

    assert torch.equal(actual, expected)


def test_antithetic_search_improves_a_known_objective() -> None:
    basis = torch.tensor([[1.0], [0.0]])
    reference = torch.tensor([10.0, 0.0])
    target = torch.tensor([0.7, 0.0])

    def evaluate(vectors: torch.Tensor) -> torch.Tensor:
        return -(vectors - target).square().sum(dim=1)

    initial_score = float(evaluate(torch.zeros(1, 2))[0])
    result = optimize_steering_vector(
        basis,
        reference,
        evaluate,
        config=SteeringSearchConfig(
            rank=1,
            iterations=3,
            antithetic_pairs=2,
            relative_radius=0.1,
            proposal_scale=0.5,
            proposal_decay=0.7,
            learning_rate=0.75,
        ),
        generator=torch.Generator().manual_seed(7),
    )

    assert result.best_score > initial_score
    assert result.vector[0] > 0.0
    assert result.vector[1] == 0.0
    assert torch.linalg.vector_norm(result.vector) <= 1.0
    assert len(result.history) == 3


def test_transport_proposal_runs_in_coefficient_space() -> None:
    basis = torch.tensor([[1.0], [0.0]])
    reference = torch.tensor([10.0, 0.0])
    target = torch.tensor([0.6, 0.0])

    result = optimize_steering_vector(
        basis,
        reference,
        lambda vectors: -(vectors - target).square().sum(dim=1),
        config=SteeringSearchConfig(
            rank=1,
            iterations=1,
            antithetic_pairs=4,
            relative_radius=0.1,
            proposal="transport",
        ),
        generator=torch.Generator().manual_seed(9),
    )

    assert len(result.history[0].candidate_scores) == 16
    assert result.vector.shape == reference.shape


def test_entropy_ignores_zero_probability_tokens() -> None:
    entropy = token_entropy(torch.tensor([[1.0, 0.0, -torch.inf]]))

    assert torch.isfinite(entropy).all()
    assert entropy[0] > 0.0


def test_generation_statistics_ignore_steps_after_eos() -> None:
    generated = torch.tensor([[1, 9, 0], [1, 2, 3]])
    lengths = generated_lengths(generated, eos_token_id=9)
    scores = (
        torch.tensor([[4.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [4.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [4.0, 0.0]]),
    )

    entropies = generated_entropies(scores, lengths)

    assert lengths.tolist() == [2, 3]
    assert torch.allclose(
        entropies[0],
        torch.stack([token_entropy(score[:1])[0] for score in scores[:2]]).mean(),
    )


def test_huggingface_generation_supports_batched_layer_steering() -> None:
    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def batch_decode(
            token_ids: torch.Tensor,
            skip_special_tokens: bool,
        ) -> list[str]:
            del skip_special_tokens
            return [" ".join(map(str, row.tolist())) for row in token_ids]

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()
    _, layer = resolve_layer(model, 0)
    hook = LayerSteeringHook()
    handle = layer.register_forward_hook(hook)
    experiment = OnlineSteeringExperiment(
        model,
        Tokenizer(),
        {
            "input_ids": torch.tensor([[1, 4, 5]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        hook,
        question="test",
        max_new_tokens=3,
        score_mode="entropy",
        expected_answer=None,
        kl_weight=0.1,
        incomplete_penalty=5.0,
        temperature=1.0,
        top_p=1.0,
    )

    try:
        answers, entropies, activations, completed = experiment.generate(
            2,
            vectors=torch.zeros(2, 16),
            sample=False,
        )
    finally:
        handle.remove()

    assert len(answers) == 2
    assert entropies.shape == (2,)
    assert torch.all(torch.isfinite(entropies))
    assert activations.shape == (2, 16)
    assert completed.shape == (2,)


def test_incomplete_answers_are_penalized() -> None:
    class Tokenizer:
        eos_token_id = 2

    experiment = object.__new__(OnlineSteeringExperiment)
    experiment.score_mode = "exact"
    experiment.expected_answer = "80"
    experiment.incomplete_penalty = 5.0

    scores = experiment.answer_scores(
        ["Final answer: 80", "Final answer: 80"],
        torch.zeros(2),
        torch.tensor([True, False]),
    )

    assert torch.equal(scores, torch.tensor([1.0, -4.0]))
