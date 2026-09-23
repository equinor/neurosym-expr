from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nine.online_steering import (
    LayerSteeringHook,
    SteeringSearchConfig,
    contrastive_basis,
    optimize_steering_vector,
)

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("the model does not expose decoder layers as model.layers")
    return layers


def resolve_layer(model: nn.Module, index: int) -> tuple[int, nn.Module]:
    layers = decoder_layers(model)
    resolved = index if index >= 0 else len(layers) + index
    if not 0 <= resolved < len(layers):
        raise ValueError(
            f"layer index {index} is outside the model's {len(layers)} layers"
        )
    return resolved, layers[resolved]


def repeat_inputs(inputs: Mapping[str, Tensor], count: int) -> dict[str, Tensor]:
    if count < 1:
        raise ValueError("count must be positive")
    repeated = {}
    for name, value in inputs.items():
        if not isinstance(value, Tensor) or value.shape[0] != 1:
            raise ValueError("each model input must be a tensor with batch size one")
        repeated[name] = value.repeat((count,) + (1,) * (value.ndim - 1))
    return repeated


def token_entropy(logits: Tensor) -> Tensor:
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    terms = torch.where(
        probabilities > 0.0,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=-1)


def generated_lengths(generated: Tensor, eos_token_id: int | None) -> Tensor:
    lengths = torch.full(
        (generated.shape[0],),
        generated.shape[1],
        dtype=torch.long,
        device=generated.device,
    )
    if eos_token_id is None:
        return lengths
    eos = generated == eos_token_id
    has_eos = eos.any(dim=1)
    first_eos = eos.to(torch.long).argmax(dim=1) + 1
    return torch.where(has_eos, first_eos, lengths)


def generated_entropies(
    scores: tuple[Tensor, ...],
    active_steps: Tensor,
) -> Tensor:
    if not scores:
        return torch.zeros(active_steps.shape[0])
    entropies = torch.stack([token_entropy(score) for score in scores])
    steps = torch.arange(entropies.shape[0], device=entropies.device).unsqueeze(1)
    mask = steps < active_steps.to(device=entropies.device).unsqueeze(0)
    return (
        (entropies * mask).sum(dim=0)
        / active_steps.to(entropies).clamp_min(1)
    )


def normalize_answer(answer: str) -> str:
    return " ".join(answer.casefold().split())


def exact_answer_scores(answers: Sequence[str], expected: str) -> Tensor:
    expected_normalized = normalize_answer(expected)
    numeric_expected = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", expected_normalized)
    scores = []
    for answer in answers:
        normalized = normalize_answer(answer)
        if numeric_expected is None:
            matched = expected_normalized in normalized
        else:
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)
            matched = bool(numbers) and numbers[-1] == expected_normalized
        scores.append(float(matched))
    return torch.tensor(scores, dtype=torch.float32)


def self_verifier_scores(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answers: Sequence[str],
    *,
    device: torch.device,
) -> Tensor:
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Question:\n"
                        f"{question}\n\n"
                        "Candidate answer:\n"
                        f"{answer}\n\n"
                        "Is the candidate answer correct, relevant, and internally "
                        "consistent? A truncated or unfinished answer is incorrect. "
                        "Respond with only Yes or No."
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for answer in answers
    ]

    def continuation_scores(label: str) -> Tensor:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        if not label_ids:
            raise RuntimeError(f"the tokenizer cannot encode verifier label {label!r}")
        encoded = tokenizer(
            [prompt + label for prompt in prompts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
        token_log_probabilities = F.log_softmax(
            logits[:, -len(label_ids) - 1 : -1],
            dim=-1,
        )
        labels = encoded.input_ids[:, -len(label_ids) :]
        return token_log_probabilities.gather(
            2,
            labels.unsqueeze(2),
        ).squeeze(2).sum(dim=1)

    return (continuation_scores("Yes") - continuation_scores("No")).cpu()


def categorical_kl(candidate_logits: Tensor, baseline_logits: Tensor) -> Tensor:
    candidate_log_probabilities = F.log_softmax(candidate_logits.float(), dim=-1)
    baseline_log_probabilities = F.log_softmax(baseline_logits.float(), dim=-1)
    candidate_probabilities = candidate_log_probabilities.exp()
    return (
        candidate_probabilities
        * (candidate_log_probabilities - baseline_log_probabilities)
    ).sum(dim=-1)


class OnlineSteeringExperiment:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        inputs: Mapping[str, Tensor],
        hook: LayerSteeringHook,
        *,
        question: str,
        max_new_tokens: int,
        score_mode: str,
        expected_answer: str | None,
        kl_weight: float,
        incomplete_penalty: float,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.hook = hook
        self.question = question
        self.max_new_tokens = max_new_tokens
        self.score_mode = score_mode
        self.expected_answer = expected_answer
        self.kl_weight = kl_weight
        self.incomplete_penalty = incomplete_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.evaluations = 0
        self.generated_tokens = 0
        self.last_answers: list[str] = []
        self.last_kl = torch.empty(0)
        self._baseline_logits = self._next_token_logits(None)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _next_token_logits(self, vectors: Tensor | None) -> Tensor:
        count = 1 if vectors is None else vectors.shape[0]
        self.hook.set_vectors(vectors)
        with torch.inference_mode():
            logits = self.model(**repeat_inputs(self.inputs, count)).logits[:, -1, :]
        self.hook.set_vectors(None)
        return logits.detach()

    def generate(
        self,
        count: int,
        *,
        vectors: Tensor | None,
        sample: bool,
    ) -> tuple[list[str], Tensor, Tensor, Tensor]:
        if vectors is not None and vectors.shape[0] != count:
            raise ValueError("one steering vector is required per generated answer")
        self.hook.reset_capture()
        self.hook.set_vectors(vectors)
        generation_options: dict[str, Any] = {}
        if sample:
            generation_options.update(
                temperature=self.temperature,
                top_p=self.top_p,
            )
        with torch.inference_mode():
            outputs = self.model.generate(
                **repeat_inputs(self.inputs, count),
                max_new_tokens=self.max_new_tokens,
                do_sample=sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=self.score_mode == "entropy",
                **generation_options,
            )
        self.hook.set_vectors(None)
        prompt_length = self.inputs["input_ids"].shape[1]
        generated = outputs.sequences[:, prompt_length:]
        answers = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        lengths = generated_lengths(generated, self.tokenizer.eos_token_id)
        completed = (
            torch.zeros(count, dtype=torch.bool, device=generated.device)
            if self.tokenizer.eos_token_id is None
            else (generated == self.tokenizer.eos_token_id).any(dim=1)
        )
        entropies = generated_entropies(outputs.scores, lengths)
        activations = self.hook.activation_means(lengths)
        self.generated_tokens += int(lengths.sum())
        return answers, entropies.cpu(), activations, completed.cpu()

    def answer_scores(
        self,
        answers: Sequence[str],
        entropies: Tensor,
        completed: Tensor,
    ) -> Tensor:
        if self.score_mode == "exact":
            assert self.expected_answer is not None
            scores = exact_answer_scores(answers, self.expected_answer)
        elif self.score_mode == "self-verifier":
            scores = self_verifier_scores(
                self.model,
                self.tokenizer,
                self.question,
                answers,
                device=self.device,
            )
        elif self.score_mode == "entropy":
            scores = -entropies.float()
        else:
            raise RuntimeError(f"unknown score mode: {self.score_mode}")
        return scores - self.incomplete_penalty * (~completed).to(scores.dtype)

    def explore(self, count: int) -> tuple[list[str], Tensor, Tensor, Tensor]:
        answers, entropies, activations, completed = self.generate(
            count,
            vectors=None,
            sample=True,
        )
        return (
            answers,
            self.answer_scores(answers, entropies, completed),
            activations,
            completed,
        )

    def evaluate(self, vectors: Tensor) -> Tensor:
        vectors = vectors.to(device=self.device)
        answers, entropies, _, completed = self.generate(
            vectors.shape[0],
            vectors=vectors,
            sample=False,
        )
        answer_scores = self.answer_scores(
            answers,
            entropies,
            completed,
        ).to(vectors.device)
        steered_logits = self._next_token_logits(vectors)
        baseline = self._baseline_logits.expand_as(steered_logits)
        divergence = categorical_kl(steered_logits, baseline)
        scores = answer_scores - self.kl_weight * divergence
        self.evaluations += vectors.shape[0]
        self.last_answers = answers
        self.last_kl = divergence.detach().cpu()
        return scores


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"{question}\n\nAnswer directly and concisely.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn one ephemeral activation-steering vector per question."
    )
    parser.add_argument("question")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--layer", type=int, default=-8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--exploration-samples", type=int, default=12)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--relative-radius", type=float, default=0.08)
    parser.add_argument("--proposal-scale", type=float, default=0.5)
    parser.add_argument(
        "--proposal",
        choices=("gaussian", "transport"),
        default="gaussian",
    )
    parser.add_argument(
        "--score",
        choices=("self-verifier", "exact", "entropy"),
        default="self-verifier",
    )
    parser.add_argument("--expected-answer")
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--incomplete-penalty", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vector-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-equal-compute-baseline", action="store_true")
    args = parser.parse_args()

    if args.exploration_samples < 4:
        parser.error("--exploration-samples must be at least 4")
    if args.score == "exact" and args.expected_answer is None:
        parser.error("--expected-answer is required with --score exact")
    if args.kl_weight < 0.0:
        parser.error("--kl-weight cannot be negative")
    if args.incomplete_penalty < 0.0:
        parser.error("--incomplete-penalty cannot be negative")
    if not 0.0 < args.temperature:
        parser.error("--temperature must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        model.requires_grad_(False)

    layer_index, layer = resolve_layer(model, args.layer)
    hook = LayerSteeringHook()
    handle = layer.register_forward_hook(hook)
    console = Console()
    started = time.perf_counter()
    try:
        prompt = build_prompt(tokenizer, args.question)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        experiment = OnlineSteeringExperiment(
            model,
            tokenizer,
            inputs,
            hook,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            score_mode=args.score,
            expected_answer=args.expected_answer,
            kl_weight=args.kl_weight,
            incomplete_penalty=args.incomplete_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        (
            exploration_answers,
            exploration_scores,
            activations,
            exploration_completed,
        ) = experiment.explore(args.exploration_samples)
        basis, contrast = contrastive_basis(
            activations.float(),
            exploration_scores.to(activations.device),
            rank=args.rank,
        )
        reference_state = activations.mean(dim=0).float()
        contrast_norm = torch.linalg.vector_norm(contrast)
        initial_vector = (
            contrast
            if contrast_norm > torch.finfo(contrast.dtype).eps
            else torch.zeros_like(contrast)
        )
        config = SteeringSearchConfig(
            rank=args.rank,
            iterations=args.iterations,
            antithetic_pairs=args.pairs,
            relative_radius=args.relative_radius,
            proposal_scale=args.proposal_scale,
            proposal=args.proposal,
        )
        search = optimize_steering_vector(
            basis,
            reference_state,
            experiment.evaluate,
            initial_vector=initial_vector,
            config=config,
        )

        baseline_index = int(torch.argmax(exploration_scores))
        baseline_answer = exploration_answers[baseline_index]
        equal_compute_answer: str | None = None
        equal_compute_score: float | None = None
        equal_compute_completed: bool | None = None
        if not args.skip_equal_compute_baseline:
            control_answers = []
            control_score_batches = []
            control_completed_batches = []
            remaining = max(1, experiment.evaluations)
            control_batch_size = 2 * args.pairs
            while remaining:
                count = min(remaining, control_batch_size)
                answers, scores, _, completed = experiment.explore(count)
                control_answers.extend(answers)
                control_score_batches.append(scores)
                control_completed_batches.append(completed)
                remaining -= count
            control_scores = torch.cat(control_score_batches)
            control_completed = torch.cat(control_completed_batches)
            control_index = int(torch.argmax(control_scores))
            equal_compute_answer = control_answers[control_index]
            equal_compute_score = float(control_scores[control_index])
            equal_compute_completed = bool(control_completed[control_index])
        final_answers, final_entropies, _, final_completed = experiment.generate(
            1,
            vectors=search.vector.to(args.device).unsqueeze(0),
            sample=False,
        )
        final_answer = final_answers[0]
        final_score = float(
            experiment.answer_scores(
                final_answers,
                final_entropies,
                final_completed,
            )[0]
        )
    finally:
        handle.remove()

    elapsed = time.perf_counter() - started
    console.print("\n[bold]Best exploration answer[/bold]")
    console.print(baseline_answer)
    if equal_compute_answer is not None:
        console.print("\n[bold]Equal-compute unsteered answer[/bold]")
        console.print(equal_compute_answer)
    console.print("\n[bold]Steered answer[/bold]")
    console.print(final_answer)
    console.print(
        f"\nlayer={layer_index} rank={basis.shape[1]} "
        f"vector_norm={torch.linalg.vector_norm(search.vector):.4f} "
        f"baseline_score={exploration_scores[baseline_index]:.4f} "
        f"search_score={search.best_score:.4f} "
        f"final_score={final_score:.4f} "
        f"final_completed={bool(final_completed[0])} "
        f"evaluations={experiment.evaluations} "
        f"generated_tokens={experiment.generated_tokens} "
        f"seconds={elapsed:.2f}",
        style="bright_black",
    )

    record: dict[str, Any] = {
        "question": args.question,
        "model": args.model,
        "layer": layer_index,
        "seed": args.seed,
        "score_mode": args.score,
        "expected_answer": args.expected_answer,
        "exploration_scores": [float(score) for score in exploration_scores],
        "best_exploration_answer": baseline_answer,
        "best_exploration_score": float(exploration_scores[baseline_index]),
        "best_exploration_completed": bool(
            exploration_completed[baseline_index]
        ),
        "equal_compute_answer": equal_compute_answer,
        "equal_compute_score": equal_compute_score,
        "equal_compute_completed": equal_compute_completed,
        "steered_answer": final_answer,
        "steered_score": final_score,
        "steered_completed": bool(final_completed[0]),
        "search_best_score": search.best_score,
        "rank": basis.shape[1],
        "steering_vector_norm": float(torch.linalg.vector_norm(search.vector)),
        "evaluations": experiment.evaluations,
        "generated_tokens": experiment.generated_tokens,
        "elapsed_seconds": elapsed,
        "search_history": [
            {
                "iteration": row.iteration,
                "candidate_scores": row.candidate_scores,
                "best_score": row.best_score,
                "proposal_scale": row.proposal_scale,
            }
            for row in search.history
        ],
    }
    if args.output is not None:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.vector_output is not None:
        torch.save(
            {
                "model": args.model,
                "layer": layer_index,
                "strength": 1.0,
                "vector": search.vector.detach().cpu(),
            },
            args.vector_output,
        )


if __name__ == "__main__":
    main()
