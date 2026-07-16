import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import DiffusionGemmaConfig, DiffusionGemmaTextConfig

from text_only_diffusion_gemma import TextOnlyDiffusionGemmaForBlockDiffusion
from wordnet_vocab import (
    ReducedTokenizer,
    VocabularyReduction,
    build_reduction,
    collect_token_ids,
    load_reduction,
    reduce_diffusion_gemma_vocabulary,
    save_reduction,
)


class FakeTokenizer:
    tokens = [
        "<pad>",
        "<eos>",
        "<unk>",
        "cat",
        "▁cat",
        "dog",
        "▁dog",
        "12",
        "+",
        "é",
        "<|image|>",
        "<|video|>",
    ]
    all_special_ids = [0, 1, 2, 10, 11]
    bos_token_id = None
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2
    mask_token_id = None
    image_token_id = 10

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, ids, *args, **kwargs):
        if isinstance(ids, int):
            return self.tokens[ids]
        return [self.tokens[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.tokens.index(tokens) if tokens in self.tokens else self.unk_token_id
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def __call__(self, values, **kwargs):
        sequences = []
        for value in values:
            prefix = "▁" if value.startswith(" ") else ""
            token = prefix + value.strip()
            sequences.append([self.convert_tokens_to_ids(token)])
        return {"input_ids": sequences}

    def encode(self, value, **kwargs):
        return self([value], **kwargs)["input_ids"][0]

    def decode(self, ids, **kwargs):
        return "".join(self.tokens[token_id].replace("▁", " ") for token_id in ids)

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(sequence, **kwargs) for sequence in sequences]


class WordNetVocabularyTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_collects_contextual_words_numbers_symbols_and_special_tokens(self):
        token_ids = collect_token_ids(self.tokenizer, ["cat"])

        self.assertEqual(token_ids, {0, 1, 2, 3, 4, 7, 8})

    def test_reduced_tokenizer_maps_in_both_directions(self):
        reduction = build_reduction(self.tokenizer, {0, 1, 2, 3, 4, 7, 8})
        tokenizer = ReducedTokenizer(self.tokenizer, reduction)

        self.assertEqual(tokenizer.encode("dog"), [tokenizer.unk_token_id])
        self.assertEqual(tokenizer.encode(" cat"), [4])
        self.assertEqual(tokenizer.decode(torch.tensor([3, 4, 5, 6])), "cat cat12+")
        self.assertEqual(tokenizer.all_special_ids, [0, 1, 2])

    def test_reduction_manifest_round_trip(self):
        reduction = build_reduction(self.tokenizer, {0, 1, 2, 3, 4, 7, 8})

        with TemporaryDirectory() as directory:
            save_reduction(Path(directory), reduction)
            loaded = load_reduction(Path(directory))

        self.assertEqual(loaded, reduction)

    def test_trimmed_model_round_trip_does_not_restore_vision_modules(self):
        text_config = DiffusionGemmaTextConfig(
            vocab_size=10,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=4,
            global_head_dim=4,
            num_experts=1,
            top_k_experts=1,
            moe_intermediate_size=16,
        )
        config = DiffusionGemmaConfig(text_config=text_config, vision_config=None)
        model = TextOnlyDiffusionGemmaForBlockDiffusion(config)
        reduction = VocabularyReduction(
            original_token_ids=(0, 1, 2, 7, 8, 9),
            original_to_reduced=(0, 1, 2, 2, 2, 2, 2, 3, 4, 5),
            unknown_token_id=2,
        )
        reduce_diffusion_gemma_vocabulary(model, reduction)

        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = TextOnlyDiffusionGemmaForBlockDiffusion.from_pretrained(
                directory,
                low_cpu_mem_usage=True,
                offload_state_dict=True,
            )

        self.assertEqual(loaded.model.decoder.embed_tokens.weight.shape, (6, 8))
        self.assertIsNone(loaded.model.encoder.vision_tower)
        self.assertIsNone(loaded.model.encoder.embed_vision)

    def test_reduces_all_model_vocabulary_components(self):
        embedding = torch.nn.Embedding(10, 4, padding_idx=0)
        encoder_embedding = torch.nn.Embedding(10, 4, padding_idx=0)
        encoder_embedding.weight = embedding.weight
        model = SimpleNamespace(
            model=SimpleNamespace(
                encoder=SimpleNamespace(
                    language_model=SimpleNamespace(embed_tokens=encoder_embedding),
                    vision_tower=torch.nn.Linear(4, 4),
                    embed_vision=torch.nn.Linear(4, 4),
                ),
                decoder=SimpleNamespace(embed_tokens=embedding),
            ),
            lm_head=torch.nn.Linear(4, 10, bias=False),
            config=SimpleNamespace(
                tie_word_embeddings=True,
                text_config=SimpleNamespace(
                    vocab_size=10,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                ),
                vision_config=SimpleNamespace(),
                boi_token_id=7,
                eoi_token_id=8,
                image_token_id=9,
            ),
            generation_config=SimpleNamespace(
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            ),
        )
        model.lm_head.weight = embedding.weight
        reduction = VocabularyReduction(
            original_token_ids=(0, 1, 2, 7, 8, 9),
            original_to_reduced=(0, 1, 2, 2, 2, 2, 2, 3, 4, 5),
            unknown_token_id=2,
        )

        reduce_diffusion_gemma_vocabulary(model, reduction)

        encoder = model.model.encoder.language_model.embed_tokens
        decoder = model.model.decoder.embed_tokens
        self.assertEqual(encoder.weight.shape, (6, 4))
        self.assertEqual(encoder.weight.data_ptr(), decoder.weight.data_ptr())
        self.assertEqual(decoder.weight.data_ptr(), model.lm_head.weight.data_ptr())
        self.assertEqual(model.config.text_config.vocab_size, 6)
        self.assertIsNone(model.model.encoder.vision_tower)
        self.assertIsNone(model.model.encoder.embed_vision)
        self.assertEqual(model.model.encoder.input_modalities, ("text",))
        self.assertIsNone(model.config.vision_config)
        self.assertIsNone(model.config.boi_token_id)
        self.assertIsNone(model.config.eoi_token_id)
        self.assertEqual(model.config.image_token_id, -1)


if __name__ == "__main__":
    unittest.main()
