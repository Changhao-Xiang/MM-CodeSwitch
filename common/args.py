import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, cast

import transformers
from rich.logging import RichHandler
from transformers.hf_argparser import HfArgumentParser


@dataclass
class RunArgs:
    module: str = field(default="")
    action: str = field(default="main")
    start_pos: int = field(default=-1)
    end_pos: int = field(default=-1)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="<path_to_checkpoint>")
    vision_model_path: str = field(default="<path_to_vision_model>")
    language_model_path: str = field(default="<path_to_language_model>")
    projector_type: Literal["mlp", "patch_merger", "identity"] = field(default="mlp")
    init_merger: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    use_flash_attn: bool = field(default=True)
    # LoRA parameters
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning of the language model"},
    )
    lora_r: int = field(default=32, metadata={"help": "LoRA dimension of the smaller matrices"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout probability for LoRA layers"})


@dataclass
class DataArguments:
    data_recipe_path: str = field(default="", metadata={"help": "Path to the training data recipe file."})
    image_folder: str = field(default="<path_to_image_folder>")
    data_packing: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.training_args.TrainingArguments):
    training_stage: Literal["pretrain", "finetune"] = field(default="pretrain")
    mm_projector_lr: Optional[float] = None
    vision_model_lr: Optional[float] = None
    # Resolution arguments
    dynamic_resolution: Literal["no", "tile", "native"] = field(default="no")
    max_num_tiles: int = field(default=6)
    max_pixels: int = field(default=28 * 28 * 1024)
    min_pixels: int = field(default=28 * 28 * 16)
    # Multi-modal code-switching arguments
    enable_mmcs: bool = field(default=False)
    mmcs_type: Literal["bbox", "mask"] = field(default="bbox")
    add_whole_image: bool = field(default=False)
    enable_coordinate_tokens: bool = field(default=False)
    coordinate_token_bins: int = field(default=32)

    log_level: str = field(default="info")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


parser = HfArgumentParser([RunArgs, ModelArguments, DataArguments, TrainingArguments])  # type: ignore
run_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
run_args = cast(RunArgs, run_args)
model_args = cast(ModelArguments, model_args)
data_args = cast(DataArguments, data_args)
training_args = cast(TrainingArguments, training_args)

# Only setup logging on the main process in distributed training
if training_args.local_rank <= 0:
    logging.basicConfig(
        level=training_args.log_level.upper(), format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("rich")
