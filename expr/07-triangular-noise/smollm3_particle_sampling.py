import argparse
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
ProgressCallback = Callable[[int, float, bool], None]


class ParticleResult(NamedTuple):
    token_ids: Tensor
    model_log_probability: float
    target_score: float
    resampling_steps: tuple[int, ...]


def token_entropy_from_log_probs(log_probs: Tensor) -> Tensor:
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum(dim=-1)


def proposal_log_probs(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Construct the token proposal used to extend each particle."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    proposal_logits = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            proposal_logits,
            dim=-1,
            descending=True,
        )
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative_probabilities - sorted_probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        proposal_logits = torch.full_like(proposal_logits, -torch.inf)
        proposal_logits.scatter_(1, sorted_indices, sorted_logits)

    return F.log_softmax(proposal_logits, dim=-1)


def effective_sample_size(log_weights: Tensor) -> float:
    weights = F.softmax(log_weights, dim=0)
    return float(weights.square().sum().reciprocal())


def sample_particles(
    model: nn.Module,
    prompt_ids: Tensor,
    *,
    eos_token_id: int,
    particle_count: int = 32,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    entropy_weight: float = 0.0,
    resample_threshold: float = 0.5,
    select_best: bool = False,
    generator: torch.Generator | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ParticleResult:
    """Sample valid token trajectories with sequential importance resampling.

    The unnormalized target is the model's sequence probability multiplied by
    exp(-entropy_weight * cumulative next-token entropy). With no entropy
    penalty, temperature 1, and top-p 1, this reduces to ancestral sampling.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids must have shape (1, sequence_length)")
    if particle_count < 2:
        raise ValueError("particle_count must be at least 2")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if entropy_weight < 0.0:
        raise ValueError("entropy_weight cannot be negative")
    if not 0.0 < resample_threshold <= 1.0:
        raise ValueError("resample_threshold must be in (0, 1]")

    sequences = prompt_ids.repeat(particle_count, 1)
    prompt_length = prompt_ids.shape[1]
    finished = torch.zeros(
        particle_count,
        dtype=torch.bool,
        device=prompt_ids.device,
    )
    log_weights = torch.zeros(
        particle_count,
        dtype=torch.float32,
        device=prompt_ids.device,
    )
    model_log_probabilities = torch.zeros_like(log_weights)
    target_scores = torch.zeros_like(log_weights)
    resampling_steps: list[int] = []

    for step in range(1, max_new_tokens + 1):
        outputs = model(input_ids=sequences)
        logits = outputs.logits[:, -1, :].float()
        model_log_probs = F.log_softmax(logits, dim=-1)
        proposal = proposal_log_probs(
            logits,
            temperature=temperature,
            top_p=top_p,
        )
        sampled_tokens = torch.multinomial(
            proposal.exp(),
            num_samples=1,
            generator=generator,
        ).squeeze(1)
        sampled_tokens = torch.where(
            finished,
            sampled_tokens.new_full((), eos_token_id),
            sampled_tokens,
        )

        active = ~finished
        chosen_model_log_probs = model_log_probs.gather(
            1, sampled_tokens.unsqueeze(1)
        ).squeeze(1)
        chosen_proposal_log_probs = proposal.gather(
            1, sampled_tokens.unsqueeze(1)
        ).squeeze(1)
        entropies = token_entropy_from_log_probs(model_log_probs)
        importance_increment = (
            chosen_model_log_probs
            - chosen_proposal_log_probs
            - entropy_weight * entropies
        )
        target_increment = chosen_model_log_probs - entropy_weight * entropies
        log_weights = log_weights + torch.where(
            active,
            importance_increment,
            torch.zeros_like(importance_increment),
        )
        model_log_probabilities = model_log_probabilities + torch.where(
            active,
            chosen_model_log_probs,
            torch.zeros_like(chosen_model_log_probs),
        )
        target_scores = target_scores + torch.where(
            active,
            target_increment,
            torch.zeros_like(target_increment),
        )
        sequences = torch.cat((sequences, sampled_tokens.unsqueeze(1)), dim=1)
        finished = finished | (sampled_tokens == eos_token_id)

        sample_size = effective_sample_size(log_weights)
        resampled = (
            sample_size < resample_threshold * particle_count
            and not bool(torch.all(finished))
        )
        if progress_callback is not None:
            progress_callback(step, sample_size, resampled)
        if resampled:
            ancestor_indices = torch.multinomial(
                F.softmax(log_weights, dim=0),
                particle_count,
                replacement=True,
                generator=generator,
            )
            sequences = sequences[ancestor_indices]
            finished = finished[ancestor_indices]
            model_log_probabilities = model_log_probabilities[ancestor_indices]
            target_scores = target_scores[ancestor_indices]
            log_weights.zero_()
            resampling_steps.append(step)

        if bool(torch.all(finished)):
            break

    if select_best:
        selected = int(torch.argmax(target_scores))
    else:
        selected = int(
            torch.multinomial(
                F.softmax(log_weights, dim=0),
                num_samples=1,
                generator=generator,
            )
        )
    generated = sequences[selected, prompt_length:]
    eos_positions = torch.nonzero(
        generated == eos_token_id,
        as_tuple=False,
    )
    if eos_positions.numel():
        generated = generated[: int(eos_positions[0]) + 1]

    return ParticleResult(
        token_ids=generated.detach().cpu(),
        model_log_probability=float(model_log_probabilities[selected]),
        target_score=float(target_scores[selected]),
        resampling_steps=tuple(resampling_steps),
    )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate from SmolLM3 with a population of valid token trajectories."
        )
    )
    parser.add_argument("prompt")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help="favor trajectories whose next-token distributions stay confident",
    )
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument(
        "--best",
        action="store_true",
        help="return the highest-scoring surviving particle instead of sampling one",
    )
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
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    console = Console()

    def show_progress(step: int, sample_size: float, resampled: bool) -> None:
        action = " resampled" if resampled else ""
        console.print(
            f"[{step:>{len(str(args.max_new_tokens))}}] "
            f"ESS={sample_size:.1f}/{args.particles}{action}",
            highlight=False,
        )

    with torch.inference_mode():
        result = sample_particles(
            model,
            prompt_ids,
            eos_token_id=tokenizer.eos_token_id,
            particle_count=args.particles,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            entropy_weight=args.entropy_weight,
            resample_threshold=args.resample_threshold,
            select_best=args.best,
            progress_callback=show_progress,
        )

    console.print()
    console.print(
        tokenizer.decode(result.token_ids, skip_special_tokens=True),
        soft_wrap=True,
    )
    console.print(
        f"\nmodel log p={result.model_log_probability:.3f} "
        f"target score={result.target_score:.3f} "
        f"resampled={len(result.resampling_steps)} times",
        style="bright_black",
    )


if __name__ == "__main__":
    main()
