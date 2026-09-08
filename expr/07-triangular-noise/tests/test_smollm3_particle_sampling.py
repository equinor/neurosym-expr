import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

MODULE_PATH = Path(__file__).parents[1] / "smollm3_particle_sampling.py"
SPEC = importlib.util.spec_from_file_location("smollm3_particle_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

effective_sample_size = MODULE.effective_sample_size
proposal_log_probs = MODULE.proposal_log_probs
sample_particles = MODULE.sample_particles


class ToyLanguageModel(nn.Module):
    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, 3),
            -4.0,
            device=input_ids.device,
        )
        logits[:, -1, 1] = 4.0
        return SimpleNamespace(logits=logits)


def test_proposal_log_probs_applies_temperature_and_top_p() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0]])

    cold = proposal_log_probs(logits, temperature=0.5)
    warm = proposal_log_probs(logits, temperature=2.0)
    truncated = proposal_log_probs(logits, top_p=0.6)

    assert cold.exp()[0, 0] > warm.exp()[0, 0]
    assert torch.isclose(truncated.exp().sum(), torch.tensor(1.0))
    assert torch.isneginf(truncated[0, 1:]).all()


def test_effective_sample_size_detects_particle_collapse() -> None:
    uniform = effective_sample_size(torch.zeros(4))
    collapsed = effective_sample_size(torch.tensor([0.0, -20.0, -20.0, -20.0]))

    assert uniform == 4.0
    assert collapsed < 1.01


def test_particle_sampler_returns_a_natural_token_trajectory() -> None:
    result = sample_particles(
        ToyLanguageModel(),
        torch.tensor([[0, 0]]),
        eos_token_id=2,
        particle_count=8,
        max_new_tokens=3,
        generator=torch.Generator().manual_seed(7),
    )

    assert result.token_ids.tolist() == [1, 1, 1]
    assert result.resampling_steps == ()
    assert result.model_log_probability <= 0.0
    assert result.target_score == result.model_log_probability
