from typing import Literal, Optional

import torch
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLPatchMerger,
)
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel


def get_device():
    """Detect and return the appropriate device for model loading."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_vision_model(
    vision_encoder_path: str,
    projector_type: Literal["mlp", "patch_merger", "identity"],
    init_merger: bool = True,
    device: Optional[str] = None,
):
    # Determine device for model loading
    if device is None:
        device = get_device()

    if "clip" in vision_encoder_path.lower():
        vision_config = CLIPVisionConfig.from_pretrained(vision_encoder_path)
        vision_model = CLIPVisionModel.from_pretrained(
            vision_encoder_path,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    elif "siglip" in vision_encoder_path.lower():
        vision_config = SiglipVisionConfig.from_pretrained(vision_encoder_path)
        vision_model = SiglipVisionModel.from_pretrained(
            vision_encoder_path,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
    elif "qwen2_5_vl" in vision_encoder_path.lower():
        vision_config = Qwen2_5_VLVisionConfig.from_pretrained(vision_encoder_path)
        vision_model = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(
            vision_encoder_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        if init_merger:
            vision_model.merger = Qwen2_5_VLPatchMerger(
                dim=vision_config.out_hidden_size,
                context_dim=vision_config.hidden_size,
                spatial_merge_size=vision_config.spatial_merge_size,
            )
    else:
        raise ValueError(f"Unsupported vision model: {vision_encoder_path}")

    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path, use_fast=False)
    if projector_type == "mlp":
        image_processor.image_seq_length = (vision_config.image_size // vision_config.patch_size) ** 2
    elif projector_type == "patch_merger":
        # default scale_factor is 0.5
        image_seq_length = int((0.5 * vision_config.image_size // vision_config.patch_size) ** 2)
        image_processor.image_seq_length = image_seq_length
    elif projector_type == "identity":
        image_processor.image_seq_length = 1
    return vision_config, vision_model, image_processor


def load_language_model(language_model_path: str, use_flash_attn: bool = True, device: Optional[str] = None):
    # Determine device for model loading
    if device is None:
        device = get_device()

    if "qwen2" in language_model_path.lower():
        language_config = Qwen2Config.from_pretrained(language_model_path)
        language_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        language_model = Qwen2ForCausalLM.from_pretrained(
            language_model_path,
            config=language_config,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
    elif "qwen3" in language_model_path.lower():
        language_config = Qwen3Config.from_pretrained(language_model_path)
        language_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        language_model = Qwen3ForCausalLM.from_pretrained(
            language_model_path,
            config=language_config,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
    elif "llama" in language_model_path.lower():
        language_config = LlamaConfig.from_pretrained(language_model_path)
        language_config._attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        language_model = LlamaForCausalLM.from_pretrained(
            language_model_path,
            config=language_config,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
    else:
        raise ValueError(f"Unsupported language model: {language_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(language_model_path, use_fast=False)
    return language_config, language_model, tokenizer
