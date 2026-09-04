import importlib.util
from pathlib import Path

import torch
from rich.console import Console
from torch import nn

MODULE_PATH = Path(__file__).parents[1] / "smollm3_hidden_steering.py"
SPEC = importlib.util.spec_from_file_location("smollm3_hidden_steering", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

Transport = MODULE.Transport
GenerationDisplay = MODULE.GenerationDisplay
FinalOutputStreamer = MODULE.FinalOutputStreamer
candidate_score = MODULE.candidate_score
token_entropy = MODULE.token_entropy


def test_candidate_score_prefers_low_entropy_and_high_density() -> None:
    information_gain = torch.tensor([0.1, 0.5, 0.9])
    log_density = torch.tensor([0.0, 0.0, 0.0])
    assert torch.argmin(candidate_score(information_gain, log_density, 1.0)) == 0

    information_gain = torch.tensor([0.5, 0.5, 0.5])
    log_density = torch.tensor([-2.0, -1.0, 0.0])
    assert torch.argmin(candidate_score(information_gain, log_density, 1.0)) == 2


def test_token_entropy_is_lower_for_confident_logits() -> None:
    logits = torch.tensor([[8.0, -8.0], [0.0, 0.0]])

    entropy = token_entropy(logits)

    assert entropy[0] < entropy[1]


def test_transport_hook_preserves_shape_and_last_state_norm() -> None:
    torch.manual_seed(5)
    lm_head = nn.Linear(4, 7, bias=False)
    hidden_states = torch.randn(1, 3, 4)
    streamed_choices = []
    intervention = Transport(
        sample_count=16,
        iterations=1,
        elite_fraction=1.0,
        max_fit_iterations=1,
        token_choice_callback=lambda *choice: streamed_choices.append(choice),
    )

    (steered_states,) = intervention(lm_head, (hidden_states,))

    assert steered_states.shape == hidden_states.shape
    assert torch.equal(steered_states[:, :-1], hidden_states[:, :-1])
    assert intervention.last_steered_hidden_state is not None
    assert torch.allclose(
        torch.linalg.vector_norm(intervention.last_steered_hidden_state, dim=1),
        torch.linalg.vector_norm(hidden_states[:, -1, :], dim=1),
    )
    assert len(intervention.token_choices) == 1
    original_token_ids, transport_token_ids = intervention.token_choices[0]
    assert torch.equal(
        original_token_ids,
        torch.argmax(lm_head(hidden_states[:, -1, :]), dim=-1),
    )
    assert torch.equal(
        transport_token_ids,
        torch.argmax(lm_head(intervention.last_steered_hidden_state), dim=-1),
    )
    assert len(streamed_choices) == 1
    assert streamed_choices[0][0] == 1
    assert torch.equal(streamed_choices[0][1], original_token_ids)
    assert torch.equal(streamed_choices[0][2], transport_token_ids)


def test_generation_display_streams_transport_diagnostics() -> None:
    class Tokenizer:
        @staticmethod
        def decode(token_ids: list[int]) -> str:
            return f"token-{token_ids[0]}"

    console = Console(record=True, force_terminal=True, color_system="standard")
    display = GenerationDisplay(Tokenizer(), console, token_number_width=2)

    display.add_token_choices(1, torch.tensor([3]), torch.tensor([5]))
    display.add_token_choices(2, torch.tensor([7]), torch.tensor([7]))

    rendered = console.export_text()
    assert (
        "[ 1] 'token-5' 5 | "
        "'token-3' 3 CHANGED"
    ) in rendered
    assert (
        "[ 2] 'token-7' 7 | "
        "'token-7' 7"
    ) in rendered
    assert rendered.count("CHANGED") == 1


def test_final_output_streamer_colors_changed_tokens_red() -> None:
    class Tokenizer:
        @staticmethod
        def decode(
            token_ids: list[int],
            skip_special_tokens: bool = False,
        ) -> str:
            return "".join(f"token-{token_id}" for token_id in token_ids)

    console = Console(record=True, force_terminal=True, color_system="standard")
    display = GenerationDisplay(Tokenizer(), console)
    streamer = FinalOutputStreamer(display)
    display.add_token_choices(1, torch.tensor([3]), torch.tensor([5]))
    display.add_token_choices(2, torch.tensor([7]), torch.tensor([7]))

    streamer.put(torch.tensor([[11, 12, 13]]))
    streamer.put(torch.tensor([5]))
    streamer.put(torch.tensor([7]))
    streamer.end()

    rendered = console.export_text(clear=False)
    styled_rendered = console.export_text(styles=True)
    assert rendered.endswith("\ntoken-3|token-5token-7\n")
    assert "\x1b[1;31mtoken-5\x1b[0m" in styled_rendered
    assert "\x1b[90mtoken-3\x1b[0m" in styled_rendered
    assert "\x1b[1;31mtoken-7\x1b[0m" not in styled_rendered
