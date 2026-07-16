from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from save_reduced_model import ensure_wordnet, write_vocabulary
from wordnet_vocab import (
    ReducedTokenizer,
    build_reduction,
    collect_token_ids,
    reduce_diffusion_gemma_vocabulary,
    save_reduction,
)


MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--vocabulary-output", type=Path, default=Path("reduced_vocabulary.json"))
    parser.add_argument("--vocabulary-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_wordnet()

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    token_ids = collect_token_ids(tokenizer)
    reduction = build_reduction(tokenizer, token_ids)
    write_vocabulary(args.vocabulary_output, tokenizer, reduction)

    print(f"Vocabulary: {len(tokenizer):,} -> {reduction.size:,} tokens")
    if args.vocabulary_only:
        return

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
    )
    reduce_diffusion_gemma_vocabulary(model, reduction)
    if model.model.encoder.vision_tower is not None or model.model.encoder.embed_vision is not None:
        raise RuntimeError("Vision modules were not removed from the reduced model.")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    save_reduction(args.output_dir, reduction)
    print(f"Saved reduced model and tokenizer to {args.output_dir}")

    processor.tokenizer = ReducedTokenizer(tokenizer, reduction)

    if args.prompt is None:
        return
    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(processor.tokenizer.decode(output.sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
