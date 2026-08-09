from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


class LlavaQwen2NativeResConfig(PretrainedConfig):
    model_type = "llava_qwen2_native_res"
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[dict] = None,
        llm_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_config is not None:
            # Load vision config
            if vision_config["model_type"] == "qwen2_5_vl":
                self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
            else:
                raise ValueError(f'Unsupported vision model: {vision_config["model_type"]}')

        if llm_config is not None:
            # Load llm config
            if llm_config["architectures"][0] == "Qwen2ForCausalLM":
                self.llm_config = Qwen2Config(**llm_config)
            else:
                raise ValueError(f'Unsupported architecture: {llm_config["architectures"][0]}')
