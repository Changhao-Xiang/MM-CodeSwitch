from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip import CLIPVisionConfig
from transformers.models.llama import LlamaConfig
from transformers.models.siglip import SiglipVisionConfig


class LlavaNextLlamaConfig(PretrainedConfig):
    model_type = "llava_next_llama"
    is_composition = True

    def __init__(self, vision_config: Optional[dict] = None, llm_config: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        if vision_config is not None:
            # Load vision config
            if vision_config["model_type"] == "clip_vision_model":
                self.vision_config = CLIPVisionConfig(**vision_config)
            elif vision_config["model_type"] == "siglip_vision_model":
                self.vision_config = SiglipVisionConfig(**vision_config)
            else:
                raise ValueError(f'Unsupported vision model: {vision_config["model_type"]}')

        if llm_config is not None:
            # Load llm config
            if llm_config["architectures"][0] == "LlamaForCausalLM":
                self.llm_config = LlamaConfig(**llm_config)
            else:
                raise ValueError(f'Unsupported architecture: {llm_config["architectures"][0]}')
