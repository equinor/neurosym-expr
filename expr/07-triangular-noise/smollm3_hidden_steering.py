import argparse
import math
from collections import deque
from collections.abc import Callable

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.text import Text
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from expr07_tria import BoostedWaveletSplineTransport


MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
TokenChoiceCallback = Callable[[int, Tensor, Tensor], None]


class GenerationDisplay:
    """Print scrolling diagnostics for each transport token choice."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        console: Console | None = None,
        token_number_width: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.console = console or Console()
        self.token_number_width = token_number_width
        self.pending_original_ids: deque[int | None] = deque()

    def add_token_choices(
        self,
        step: int,
        original_ids: Tensor,
        transport_ids: Tensor,
    ) -> None:
        for batch_index, (original_id, transport_id) in enumerate(
            zip(original_ids.tolist(), transport_ids.tolist(), strict=True)
        ):
            batch = f" batch={batch_index}" if len(original_ids) > 1 else ""
            original_text = self.tokenizer.decode([original_id])
            transport_text = self.tokenizer.decode([transport_id])
            changed = original_id != transport_id
            self.pending_original_ids.append(original_id if changed else None)
            line = Text()
            line.append(
                f"[{step:>{self.token_number_width}}{batch}] "
                f"{transport_text!r} "
                f"{transport_id}",
                style="bold yellow" if changed else None,
            )
            line.append(f" | {original_text!r} {original_id}")
            if changed:
                line.append(" CHANGED", style="bold white on red")
            self.console.print(line, soft_wrap=True)

    def take_original_token_id(self) -> int | None:
        if not self.pending_original_ids:
            raise RuntimeError(
                "received a generated token without transport diagnostics"
            )
        return self.pending_original_ids.popleft()


class FinalOutputStreamer(BaseStreamer):
    """Collect generated tokens and color changed ones in the final output."""

    def __init__(self, display: GenerationDisplay) -> None:
        self.display = display
        self.next_tokens_are_prompt = True
        self.generated_tokens: list[tuple[int, int | None]] = []

    def put(self, value: Tensor) -> None:
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        for token_id in value.detach().reshape(-1).tolist():
            self.generated_tokens.append(
                (token_id, self.display.take_original_token_id())
            )

    def end(self) -> None:
        if self.display.pending_original_ids:
            raise RuntimeError("generation ended before all diagnostics were consumed")

        output = Text()
        for token_id, original_id in self.generated_tokens:
            token_text = self.display.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )
            if original_id is not None:
                original_text = self.display.tokenizer.decode(
                    [original_id],
                    skip_special_tokens=True,
                )
                output.append(original_text, style="bright_black")
                output.append("|")
            output.append(
                token_text, style="bold red" if original_id is not None else None
            )
        self.display.console.print()
        self.display.console.print(output)


def token_entropy(logits: Tensor) -> Tensor:
    """Return next-token entropy in nats for each candidate state."""
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1)


def standardized(values: Tensor) -> Tensor:
    scale = values.std(correction=0).clamp_min(torch.finfo(values.dtype).eps)
    return (values - values.mean()) / scale


def candidate_score(
    information_gain: Tensor,
    log_density: Tensor,
    density_weight: float,
) -> Tensor:
    """Rank low-information, high-density candidates; lower is better."""
    return standardized(information_gain) - density_weight * standardized(log_density)


def transport_log_prob(
    transport: BoostedWaveletSplineTransport,
    states: Tensor,
) -> Tensor:
    """Evaluate density under a transport to a standard Gaussian reference."""
    reference = transport(states)
    log_reference_density = -0.5 * (reference.square() + math.log(2.0 * math.pi)).sum(
        dim=1
    )
    return log_reference_density + transport.log_abs_det_jacobian(states)


def match_norm(samples: Tensor, target: Tensor) -> Tensor:
    target_norm = torch.linalg.vector_norm(target)
    sample_norms = torch.linalg.vector_norm(samples, dim=1, keepdim=True)
    return samples * (
        target_norm / sample_norms.clamp_min(torch.finfo(samples.dtype).eps)
    )


class Transport:
    """Capture and modify the final hidden state immediately before lm_head."""

    def __init__(
        self,
        *,
        sample_count: int = 256,
        iterations: int = 2,
        elite_fraction: float = 0.25,
        noise_scale: float = 0.05,
        density_weight: float = 1.0,
        max_fit_iterations: int = 10,
        max_wavelet_level: int = 6,
        max_parent_distance: int = 256,
        max_learners_per_component: int = 4,
        candidate_count: int = 8,
        block_size: int = 256,
        token_choice_callback: TokenChoiceCallback | None = None,
    ) -> None:
        elite_count = math.floor(sample_count * elite_fraction)
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if elite_count < 2:
            raise ValueError(
                "sample_count * elite_fraction must select at least 2 states"
            )
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")
        if density_weight < 0.0:
            raise ValueError("density_weight cannot be negative")
        if max_wavelet_level < 0:
            raise ValueError("max_wavelet_level cannot be negative")
        if max_parent_distance < 1:
            raise ValueError("max_parent_distance must be positive")
        if max_learners_per_component < 0:
            raise ValueError("max_learners_per_component cannot be negative")
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if block_size < 1:
            raise ValueError("block_size must be positive")

        self.sample_count = sample_count
        self.iterations = iterations
        self.elite_count = elite_count
        self.noise_scale = noise_scale
        self.density_weight = density_weight
        self.max_fit_iterations = max_fit_iterations
        self.max_wavelet_level = max_wavelet_level
        self.max_parent_distance = max_parent_distance
        self.max_learners_per_component = max_learners_per_component
        self.candidate_count = candidate_count
        self.block_size = block_size
        self.token_choice_callback = token_choice_callback
        self.last_hidden_state: Tensor | None = None
        self.last_steered_hidden_state: Tensor | None = None
        self.token_choices: list[tuple[Tensor, Tensor]] = []

    @staticmethod
    def _logits(module: nn.Module, states: Tensor) -> Tensor:
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise TypeError("the output embedding must expose a matrix weight")
        return F.linear(states.to(weight.dtype), weight, bias)

    def _information_gain(self, module: nn.Module, states: Tensor) -> Tensor:
        return token_entropy(self._logits(module, states))

    def _initial_samples(self, state: Tensor) -> Tensor:
        coordinate_scale = (
            state.square().mean().sqrt().clamp_min(torch.finfo(state.dtype).eps)
        )
        samples = state.unsqueeze(0) + (
            torch.randn(
                self.sample_count,
                state.numel(),
                device=state.device,
                dtype=state.dtype,
            )
            * coordinate_scale
            * self.noise_scale
        )
        samples = match_norm(samples, state)
        # samples[0] = state
        return samples

    def _steer_state(self, module: nn.Module, state: Tensor) -> Tensor:
        fit_dtype = (
            torch.float32
            if state.dtype in (torch.float16, torch.bfloat16)
            else state.dtype
        )
        original = state.to(fit_dtype)
        candidates = self._initial_samples(original)
        transport = BoostedWaveletSplineTransport(
            max_wavelet_level=self.max_wavelet_level,
            max_parent_distance=self.max_parent_distance,
            max_learners_per_component=self.max_learners_per_component,
            candidate_count=self.candidate_count,
            block_size=self.block_size,
            max_fit_iterations=self.max_fit_iterations,
        )

        for _ in range(self.iterations):
            transport.fit(candidates)
            score = candidate_score(
                self._information_gain(module, candidates),
                transport_log_prob(transport, candidates),
                self.density_weight,
            )
            elite_indices = torch.topk(
                score,
                self.elite_count,
                largest=False,
            ).indices
            elites = candidates[elite_indices]
            transport.fit(elites)
            reference = torch.randn_like(candidates)
            candidates = match_norm(transport.inverse(reference), original)
            # candidates[0] = original

        transport.fit(candidates)
        final_score = candidate_score(
            self._information_gain(module, candidates),
            transport_log_prob(transport, candidates),
            self.density_weight,
        )
        return candidates[torch.argmin(final_score)].to(state.dtype)

    def __call__(
        self, module: nn.Module, inputs: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        hidden_states = inputs[0]
        self.last_hidden_state = hidden_states[:, -1, :].detach().clone()
        state = hidden_states.clone()
        steered = torch.stack(
            [
                self._steer_state(module, batch_state)
                for batch_state in self.last_hidden_state
            ]
        )
        state[:, -1, :] = steered
        self.last_steered_hidden_state = steered.detach().clone()
        original_token_ids = torch.argmax(
            self._logits(module, self.last_hidden_state), dim=-1
        )
        transport_token_ids = torch.argmax(self._logits(module, steered), dim=-1)
        token_choice = (
            original_token_ids.detach().cpu(),
            transport_token_ids.detach().cpu(),
        )
        self.token_choices.append(token_choice)
        if self.token_choice_callback is not None:
            self.token_choice_callback(
                len(self.token_choices),
                *token_choice,
            )

        return (state, *inputs[1:])


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate with SmolLM3 while multiplying its final hidden state."
    )
    parser.add_argument("prompt")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--max-wavelet-level", type=int, default=6)
    parser.add_argument("--max-parent-distance", type=int, default=256)
    parser.add_argument("--max-learners-per-component", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="disable SmolLM3 extended thinking",
    )

    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.no_reasoning,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    display = GenerationDisplay(
        tokenizer,
        token_number_width=len(str(args.max_new_tokens)),
    )
    intervention = Transport(
        sample_count=args.samples,
        iterations=args.iterations,
        elite_fraction=args.elite_fraction,
        noise_scale=args.noise_scale,
        density_weight=args.density_weight,
        max_fit_iterations=args.fit_iterations,
        max_wavelet_level=args.max_wavelet_level,
        max_parent_distance=args.max_parent_distance,
        max_learners_per_component=args.max_learners_per_component,
        candidate_count=args.candidate_count,
        block_size=args.block_size,
        token_choice_callback=display.add_token_choices,
    )
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("the model does not expose an output embedding (lm_head)")

    handle = lm_head.register_forward_pre_hook(intervention)
    streamer = FinalOutputStreamer(display)
    try:
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
