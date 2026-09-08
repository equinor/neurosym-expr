from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor

from text_only_diffusion_gemma import TextOnlyDiffusionGemmaForBlockDiffusion
from wordnet_vocab import ReducedTokenizer, load_reduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reduction = load_reduction(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    if len(processor.tokenizer) != len(reduction.original_to_reduced):
        raise ValueError(
            "The saved tokenizer vocabulary does not match the reduction manifest."
        )
    processor.tokenizer = ReducedTokenizer(processor.tokenizer, reduction)
    model = TextOnlyDiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_dir,
        dtype="auto",
        device_map="auto",
    )

    print(f"Loaded {args.model_dir} with {len(processor.tokenizer):,} tokens")
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
