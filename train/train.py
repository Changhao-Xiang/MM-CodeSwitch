import torch
from peft import LoraConfig, TaskType, get_peft_model

from common.args import data_args, logger, model_args, training_args
from model import *
from model.build import load_language_model, load_vision_model
from train.dataset import make_data_module
from train.dataset_packed import make_packed_data_module
from train.replace_flash_attn import replace_attention_class_for_packed_data

# from transformers.trainer import Trainer
from train.trainer import Trainer  # Monkey patch optimizer for seperate lr


def set_parameters(module, trainable: bool = False):
    for param in module.parameters():
        param.requires_grad = trainable


def train():
    if data_args.data_packing:
        replace_attention_class_for_packed_data()

    # Determine language model type
    if "qwen2" in model_args.language_model_path.lower() or "qwen2" in model_args.model_name_or_path.lower():
        language_model_type = "qwen2"
    elif "qwen3" in model_args.language_model_path.lower() or "qwen3" in model_args.model_name_or_path.lower():
        language_model_type = "qwen3"
    elif "llama" in model_args.language_model_path.lower() or "llama" in model_args.model_name_or_path.lower():
        language_model_type = "llama"
    else:
        raise ValueError(f"Unsupported language model: {model_args.language_model_path}")

    if training_args.dynamic_resolution == "native":
        if language_model_type == "qwen2":
            config_class = LlavaQwen2NativeResConfig
            processor_class = LlavaQwen2NativeResProcessor
            model_class = LlavaQwen2NativeResForCausalLM
    elif training_args.dynamic_resolution == "tile":
        if language_model_type == "qwen2":
            config_class = LlavaNextQwen2Config
            processor_class = LlavaNextQwen2Processor
            model_class = LlavaNextQwen2ForCausalLM
        elif language_model_type == "qwen3":
            config_class = LlavaNextQwen3Config
            processor_class = LlavaNextQwen3Processor
            model_class = LlavaNextQwen3ForCausalLM
        elif language_model_type == "llama":
            config_class = LlavaNextLlamaConfig
            processor_class = LlavaNextLlamaProcessor
            model_class = LlavaNextLlamaForCausalLM
    else:
        raise ValueError(f"Unsupported dynamic resolution strategy: {training_args.dynamic_resolution}")

    # Load processor and model
    if training_args.training_stage == "pretrain":
        # Load vision model and language model separately
        vision_config, vision_model, image_processor = load_vision_model(
            model_args.vision_model_path, model_args.projector_type
        )
        vision_config.dynamic_resolution = training_args.dynamic_resolution
        vision_config.max_num_tiles = training_args.max_num_tiles
        language_config, language_model, tokenizer = load_language_model(
            model_args.language_model_path, use_flash_attn=model_args.use_flash_attn
        )
        config = config_class(vision_config.to_dict(), language_config.to_dict())
        processor = processor_class(image_processor, tokenizer)

        config.image_token_id = processor.image_token_id
        config.projector_type = model_args.projector_type

        # Config codeswitch
        if training_args.enable_mmcs:
            config.switch_embeds = True
            config.mmcs_type = training_args.mmcs_type
            config.add_whole_image = training_args.add_whole_image

        model = model_class(config, vision_model, language_model)  # type: ignore

    elif training_args.training_stage == "finetune":
        config = config_class.from_pretrained(model_args.model_name_or_path)
        processor = processor_class.from_pretrained(model_args.model_name_or_path, use_fast=False)
        if training_args.dynamic_resolution == "tile":
            config.vision_config.dynamic_resolution = training_args.dynamic_resolution
            config.vision_config.max_num_tiles = training_args.max_num_tiles

        _attn_implementation = "flash_attention_2" if model_args.use_flash_attn else "eager"
        config.llm_config._attn_implementation = _attn_implementation
        config.switch_embeds = False

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            attn_implementation=_attn_implementation,
        )

    # Resize token embeddings if <image> is added
    if hasattr(processor, "num_new_tokens") and processor.num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(processor.tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[: -processor.num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-processor.num_new_tokens :] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(processor.tokenizer)
        model.language_model.config.vocab_size = len(processor.tokenizer)

    # Freeze parameters
    if model_args.freeze_vit:
        # model.vision_model = model.vision_model.eval()
        set_parameters(model.vision_model, trainable=False)
        if hasattr(model.vision_model, "merger"):  # always unfreeze patch merger
            set_parameters(model.vision_model.merger, trainable=True)
    if model_args.freeze_llm:
        # model.language_model = model.language_model.eval()
        set_parameters(model.language_model, trainable=False)

    # Apply LoRA if specified
    if model_args.use_lora:
        if model_args.freeze_llm:
            logger.warning("Both freeze_llm and use_lora are set to True. Setting freeze_llm=False to enable LoRA.")
            # Unfreeze LLM parameters that will be trained with LoRA
            for param in model.language_model.parameters():
                param.requires_grad = True

        # Configure LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )

        # Apply LoRA to language model
        logger.info(f"Applying LoRA to language model with config: {peft_config}")
        model.language_model = get_peft_model(model.language_model, peft_config)  # type: ignore
        model.language_model.enable_input_require_grads()
        model.language_model.print_trainable_parameters()

    logger.info("Successfully loaded model and processor.")

    # Initialize dataset and data collator
    if data_args.data_packing:
        train_dataset, data_collator = make_packed_data_module(
            data_args.data_recipe_path, processor, config.vision_config
        )
    else:
        train_dataset, data_collator = make_data_module(data_args.data_recipe_path, processor, config.vision_config)

    # Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()

    # Save the model and processor
    if model_args.use_lora:
        # Merge LoRA weights into the base model for direct loading with from_pretrained
        logger.info("Merging LoRA weights into base model...")
        model.language_model = model.language_model.merge_and_unload()  # type: ignore
        logger.info("LoRA weights merged successfully.")

    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)  # type: ignore
    logger.info(f"Checkpoint saved to {training_args.output_dir}")


def main():
    train()
