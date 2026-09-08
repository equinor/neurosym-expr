from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from save_reduced_model import save_reduced_model


class SaveReducedModelTest(TestCase):
    @patch("save_reduced_model.save_reduction")
    @patch("save_reduced_model.reduce_diffusion_gemma_vocabulary")
    @patch("save_reduced_model.collect_token_ids")
    @patch("save_reduced_model.build_reduction")
    @patch("save_reduced_model.TextOnlyDiffusionGemmaForBlockDiffusion.from_pretrained")
    @patch("save_reduced_model.AutoProcessor.from_pretrained")
    @patch("save_reduced_model.ensure_wordnet")
    def test_saves_artifacts_without_running_inference(
        self,
        ensure_wordnet,
        load_processor,
        load_model,
        build_reduction,
        collect_token_ids,
        reduce_vocabulary,
        save_reduction,
    ):
        tokenizer = MagicMock()
        tokenizer.__len__.return_value = 10
        tokenizer.name_or_path = "example/model"
        tokenizer.convert_ids_to_tokens.return_value = ["a", "b"]
        processor = SimpleNamespace(tokenizer=tokenizer, save_pretrained=MagicMock())
        load_processor.return_value = processor

        reduction = SimpleNamespace(size=2, original_token_ids=(1, 2))
        build_reduction.return_value = reduction
        model = MagicMock()
        model.model.encoder.vision_tower = None
        model.model.encoder.embed_vision = None
        load_model.return_value = model

        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "model"
            vocabulary_output = Path(directory) / "vocabulary.json"
            save_reduced_model("example/model", output_dir, vocabulary_output)

            self.assertTrue(vocabulary_output.is_file())

        ensure_wordnet.assert_called_once_with()
        collect_token_ids.assert_called_once_with(tokenizer)
        build_reduction.assert_called_once_with(tokenizer, collect_token_ids.return_value)
        load_model.assert_called_once_with(
            "example/model",
            dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_state_dict=True,
        )
        reduce_vocabulary.assert_called_once_with(model, reduction)
        model.save_pretrained.assert_called_once_with(output_dir)
        processor.save_pretrained.assert_called_once_with(output_dir)
        save_reduction.assert_called_once_with(output_dir, reduction)
        model.generate.assert_not_called()
        model.forward.assert_not_called()
