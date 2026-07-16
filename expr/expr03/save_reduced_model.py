from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nltk
from transformers import AutoProcessor

from text_only_diffusion_gemma import TextOnlyDiffusionGemmaForBlockDiffusion
from wordnet_vocab import (
    build_reduction,
    collect_token_ids,
    reduce_diffusion_gemma_vocabulary,
    save_reduction,
)


MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce DiffusionGemma and save it locally without running inference."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--vocabulary-output",
        type=Path,
        default=Path("reduced_vocabulary.json"),
    )
    return parser.parse_args()


def ensure_wordnet() -> None:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("NLTK could not download the WordNet corpus.")


def write_vocabulary(path: Path, tokenizer: Any, reduction: Any) -> None:
    data = {
        "model_id": tokenizer.name_or_path,
        "original_vocabulary_size": len(tokenizer),
        "reduced_vocabulary_size": reduction.size,
        "original_token_ids": list(reduction.original_token_ids),
        "tokens": tokenizer.convert_ids_to_tokens(list(reduction.original_token_ids)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")


def save_reduced_model(
    model_id: str,
    output_dir: Path,
    vocabulary_output: Path,
) -> None:
    ensure_wordnet()

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    reduction = build_reduction(tokenizer, collect_token_ids(tokenizer))
    write_vocabulary(vocabulary_output, tokenizer, reduction)
    print(f"Vocabulary: {len(tokenizer):,} -> {reduction.size:,} tokens")

    model = TextOnlyDiffusionGemmaForBlockDiffusion.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )
    reduce_diffusion_gemma_vocabulary(model, reduction)
    if model.model.encoder.vision_tower is not None or model.model.encoder.embed_vision is not None:
        raise RuntimeError("Vision modules were not removed from the reduced model.")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    save_reduction(output_dir, reduction)
    print(f"Saved reduced model and tokenizer to {output_dir}")


def main() -> None:
    args = parse_args()
    save_reduced_model(args.model_id, args.output_dir, args.vocabulary_output)


if __name__ == "__main__":
    main()
