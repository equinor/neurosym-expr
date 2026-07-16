from __future__ import annotations

from torch import nn
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderModel,
    DiffusionGemmaEncoderModel,
    DiffusionGemmaEncoderTextModel,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaModel,
    DiffusionGemmaPreTrainedModel,
)


class TextOnlyDiffusionGemmaEncoderModel(DiffusionGemmaEncoderModel):
    def __init__(self, config):
        DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = DiffusionGemmaEncoderTextModel(config=config.text_config)
        self.vision_tower = None
        self.embed_vision = None
        self.input_modalities = ("text",)
        self.post_init()


class TextOnlyDiffusionGemmaModel(DiffusionGemmaModel):
    def __init__(self, config):
        DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.encoder = TextOnlyDiffusionGemmaEncoderModel(config)
        self.decoder = DiffusionGemmaDecoderModel(config)
        self.post_init()


class TextOnlyDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    def __init__(self, config):
        DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.model = TextOnlyDiffusionGemmaModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.final_logit_softcapping = config.text_config.final_logit_softcapping
        self.post_init()
