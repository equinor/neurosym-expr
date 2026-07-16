from __future__ import annotations

import copy
import json
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from nltk.corpus import wordnet


REDUCTION_FILENAME = "vocabulary_reduction.json"


@dataclass(frozen=True)
class VocabularyReduction:
    original_token_ids: tuple[int, ...]
    original_to_reduced: tuple[int, ...]
    unknown_token_id: int

    @property
    def size(self) -> int:
        return len(self.original_token_ids)


def save_reduction(output_dir: Path, reduction: VocabularyReduction) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": 1,
        "original_vocabulary_size": len(reduction.original_to_reduced),
        "original_token_ids": list(reduction.original_token_ids),
        "unknown_token_id": reduction.unknown_token_id,
    }
    (output_dir / REDUCTION_FILENAME).write_text(
        json.dumps(data, ensure_ascii=True),
        encoding="utf-8",
    )


def load_reduction(model_dir: Path) -> VocabularyReduction:
    path = model_dir / REDUCTION_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported vocabulary reduction format in {path}.")

    original_ids = tuple(data["original_token_ids"])
    original_vocabulary_size = data["original_vocabulary_size"]
    unknown_token_id = data["unknown_token_id"]
    if not original_ids or original_ids != tuple(sorted(set(original_ids))):
        raise ValueError(f"Invalid retained token IDs in {path}.")
    if original_ids[-1] >= original_vocabulary_size:
        raise ValueError(f"Retained token ID is outside the original vocabulary in {path}.")
    if not 0 <= unknown_token_id < len(original_ids):
        raise ValueError(f"Invalid unknown token ID in {path}.")

    original_to_reduced = [unknown_token_id] * original_vocabulary_size
    for reduced_id, original_id in enumerate(original_ids):
        original_to_reduced[original_id] = reduced_id
    return VocabularyReduction(
        original_token_ids=original_ids,
        original_to_reduced=tuple(original_to_reduced),
        unknown_token_id=unknown_token_id,
    )


def english_wordnet_words() -> Iterator[str]:
    """Yield unique WordNet English lemmas in their natural, space-separated form."""
    lemmas = {lemma.replace("_", " ") for lemma in wordnet.all_lemma_names(lang="eng")}
    yield from sorted(lemmas)


def _batches(values: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _is_number_or_symbol_token(token: str) -> bool:
    if token.startswith("<0x") and token.endswith(">"):
        return True

    # SentencePiece and byte-level BPE use these as whitespace/newline markers,
    # not as part of the token's semantic content.
    content = token.lstrip("▁ĠĊ")
    return any(
        character.isdigit() or unicodedata.category(character)[0] in {"P", "S"}
        for character in content
    )


def _non_text_special_token_ids(tokenizer: Any) -> set[int]:
    token_ids: set[int] = set()
    modality_names = ("image", "video")

    for name in (
        "image_token_id",
        "boi_token_id",
        "eoi_token_id",
        "video_token_id",
        "bov_token_id",
        "eov_token_id",
    ):
        token_id = getattr(tokenizer, name, None)
        if isinstance(token_id, int):
            token_ids.add(token_id)

    for token_id in tokenizer.all_special_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, str) and any(modality in token.lower() for modality in modality_names):
            token_ids.add(token_id)

    return token_ids


def _text_special_token_ids(tokenizer: Any) -> set[int]:
    return set(tokenizer.all_special_ids) - _non_text_special_token_ids(tokenizer)


def collect_token_ids(
    tokenizer: Any,
    words: Iterable[str] | None = None,
    *,
    batch_size: int = 2048,
) -> set[int]:
    """Collect tokens for WordNet words in both initial and in-sentence contexts."""
    non_text_token_ids = _non_text_special_token_ids(tokenizer)
    token_ids = _text_special_token_ids(tokenizer)
    vocabulary_size = len(tokenizer)

    for token_id in range(vocabulary_size):
        if token_id in non_text_token_ids:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is not None and _is_number_or_symbol_token(token):
            token_ids.add(token_id)

    source_words = english_wordnet_words() if words is None else words
    for batch in _batches(source_words, batch_size):
        contextualized = batch + [f" {word}" for word in batch]
        encoded = tokenizer(
            contextualized,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        token_ids.update(token_id for sequence in encoded for token_id in sequence)

    return token_ids


def build_reduction(tokenizer: Any, token_ids: Iterable[int]) -> VocabularyReduction:
    original_ids = sorted(
        (set(token_ids) - _non_text_special_token_ids(tokenizer))
        | _text_special_token_ids(tokenizer)
    )
    unknown_original_id = tokenizer.unk_token_id
    if unknown_original_id is None:
        raise ValueError("The tokenizer must define unk_token_id.")
    if unknown_original_id not in original_ids:
        original_ids.append(unknown_original_id)
        original_ids.sort()

    unknown_reduced_id = original_ids.index(unknown_original_id)
    old_to_new = [unknown_reduced_id] * len(tokenizer)
    for new_id, old_id in enumerate(original_ids):
        old_to_new[old_id] = new_id

    return VocabularyReduction(
        original_token_ids=tuple(original_ids),
        original_to_reduced=tuple(old_to_new),
        unknown_token_id=unknown_reduced_id,
    )


class ReducedTokenizer:
    """Tokenizer adapter translating between original and reduced token IDs."""

    def __init__(self, tokenizer: Any, reduction: VocabularyReduction):
        self._tokenizer = tokenizer
        self.reduction = reduction

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __len__(self) -> int:
        return self.reduction.size

    @property
    def vocab_size(self) -> int:
        return self.reduction.size

    def _to_reduced_id(self, token_id: int | None) -> int | None:
        if token_id is None:
            return None
        return self.reduction.original_to_reduced[token_id]

    def _to_original_id(self, token_id: int) -> int:
        return self.reduction.original_token_ids[token_id]

    @property
    def all_special_ids(self) -> list[int]:
        retained_ids = set(self.reduction.original_token_ids)
        return [
            self._to_reduced_id(token_id)
            for token_id in self._tokenizer.all_special_ids
            if token_id in retained_ids
        ]

    @property
    def bos_token_id(self) -> int | None:
        return self._to_reduced_id(self._tokenizer.bos_token_id)

    @property
    def eos_token_id(self) -> int | None:
        return self._to_reduced_id(self._tokenizer.eos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        return self._to_reduced_id(self._tokenizer.pad_token_id)

    @property
    def unk_token_id(self) -> int:
        return self.reduction.unknown_token_id

    @property
    def mask_token_id(self) -> int | None:
        return self._to_reduced_id(self._tokenizer.mask_token_id)

    def _map_to_reduced(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            mapping = torch.tensor(
                self.reduction.original_to_reduced,
                dtype=torch.long,
                device=value.device,
            )
            return mapping[value]
        if isinstance(value, list):
            if not value or isinstance(value[0], int):
                return [self.reduction.original_to_reduced[token_id] for token_id in value]
            return [self._map_to_reduced(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._map_to_reduced(item) for item in value)
        if isinstance(value, int):
            return self.reduction.original_to_reduced[value]
        if hasattr(value, "keys") and "input_ids" in value:
            value["input_ids"] = self._map_to_reduced(value["input_ids"])
        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._map_to_reduced(self._tokenizer(*args, **kwargs))

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        result = self._tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", True):
            return self._map_to_reduced(result)
        return result

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return self._map_to_reduced(self._tokenizer.encode(*args, **kwargs))

    def decode(self, token_ids: Any, *args: Any, **kwargs: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        original_ids = [self._to_original_id(token_id) for token_id in token_ids]
        return self._tokenizer.decode(original_ids, *args, **kwargs)

    def batch_decode(self, sequences: Any, *args: Any, **kwargs: Any) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        original = [
            [self._to_original_id(token_id) for token_id in sequence]
            for sequence in sequences
        ]
        return self._tokenizer.batch_decode(original, *args, **kwargs)

    def convert_ids_to_tokens(self, ids: int | Sequence[int], *args: Any, **kwargs: Any) -> Any:
        if isinstance(ids, int):
            ids = self._to_original_id(ids)
        else:
            ids = [self._to_original_id(token_id) for token_id in ids]
        return self._tokenizer.convert_ids_to_tokens(ids, *args, **kwargs)

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> Any:
        original = self._tokenizer.convert_tokens_to_ids(tokens)
        return self._map_to_reduced(original)

    def get_vocab(self) -> dict[str, int]:
        return {
            self._tokenizer.convert_ids_to_tokens(old_id): new_id
            for new_id, old_id in enumerate(self.reduction.original_token_ids)
        }


def _reduced_embedding(
    module: torch.nn.Module,
    token_ids: torch.Tensor,
    reduction: VocabularyReduction,
) -> torch.nn.Module:
    reduced = copy.copy(module)
    reduced._parameters = module._parameters.copy()
    reduced._buffers = module._buffers.copy()
    reduced._modules = module._modules.copy()
    weight = module.weight.detach().index_select(0, token_ids.to(module.weight.device))
    reduced.weight = torch.nn.Parameter(weight, requires_grad=module.weight.requires_grad)
    reduced.num_embeddings = token_ids.numel()
    if reduced.padding_idx is not None:
        reduced.padding_idx = reduction.original_to_reduced[reduced.padding_idx]
    return reduced


def _remap_id(value: int | list[int] | None, reduction: VocabularyReduction) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_remap_id(item, reduction) for item in value]
    return reduction.original_to_reduced[value]


def reduce_diffusion_gemma_vocabulary(model: Any, reduction: VocabularyReduction) -> None:
    """Reduce DiffusionGemma to its retained text vocabulary in place."""
    token_ids = torch.tensor(reduction.original_token_ids, dtype=torch.long)
    encoder_embeddings = model.model.encoder.language_model.embed_tokens
    decoder_embeddings = model.model.decoder.embed_tokens
    reduced_decoder = _reduced_embedding(decoder_embeddings, token_ids, reduction)
    reduced_encoder = copy.copy(encoder_embeddings)
    reduced_encoder._parameters = encoder_embeddings._parameters.copy()
    reduced_encoder._buffers = encoder_embeddings._buffers.copy()
    reduced_encoder._modules = encoder_embeddings._modules.copy()
    reduced_encoder.weight = reduced_decoder.weight
    reduced_encoder.num_embeddings = reduction.size
    if reduced_encoder.padding_idx is not None:
        reduced_encoder.padding_idx = reduction.original_to_reduced[reduced_encoder.padding_idx]

    model.model.encoder.language_model.embed_tokens = reduced_encoder
    model.model.decoder.embed_tokens = reduced_decoder

    old_head = model.lm_head
    reduced_head = copy.copy(old_head)
    reduced_head._parameters = old_head._parameters.copy()
    reduced_head._buffers = old_head._buffers.copy()
    reduced_head._modules = old_head._modules.copy()
    reduced_head.out_features = reduction.size
    if model.config.tie_word_embeddings:
        reduced_head.weight = reduced_decoder.weight
    else:
        selected = old_head.weight.detach().index_select(0, token_ids.to(old_head.weight.device))
        reduced_head.weight = torch.nn.Parameter(selected, requires_grad=old_head.weight.requires_grad)
    if old_head.bias is not None:
        bias = old_head.bias.detach().index_select(0, token_ids.to(old_head.bias.device))
        reduced_head.bias = torch.nn.Parameter(bias, requires_grad=old_head.bias.requires_grad)
    model.lm_head = reduced_head

    model.config.text_config.vocab_size = reduction.size
    for config in (model.config.text_config, model.generation_config):
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            setattr(config, name, _remap_id(getattr(config, name, None), reduction))

    encoder = model.model.encoder
    if hasattr(encoder, "vision_tower"):
        encoder.vision_tower = None
    if hasattr(encoder, "embed_vision"):
        encoder.embed_vision = None
    encoder.input_modalities = ("text",)

    model.config.vision_config = None
    model.config.boi_token_id = None
    model.config.eoi_token_id = None
    # The Transformers encoder compares every input ID to this value even when
    # no vision tower is configured, so use an impossible ID rather than None.
    model.config.image_token_id = -1
