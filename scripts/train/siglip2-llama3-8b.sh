#!/bin/bash
set -euo pipefail

# SigLIP2 + Llama3-8B: 773k MMCS pretraining followed by 779k LLaVA-NeXT LoRA SFT.
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NUM_GPUS=${NUM_GPUS:-8}
VISION_MODEL_PATH=${VISION_MODEL_PATH:-google/siglip2-so400m-patch16-384}
LANGUAGE_MODEL_PATH=${LANGUAGE_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}
PRETRAIN_RECIPE=${PRETRAIN_RECIPE:-train/recipe/mmcs.json}
SFT_RECIPE=${SFT_RECIPE:-train/recipe/sft_779k.json}
PRETRAIN_OUTPUT_DIR=${PRETRAIN_OUTPUT_DIR:-checkpoints/siglip2-llama3-next-8B-pretrain}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-checkpoints/siglip2-llama3-next-8B-sft-779k-lora}

torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
    run.py --module train.train \
    --training_stage pretrain \
    --dynamic_resolution tile \
    --max_num_tiles 1 \
    --enable_mmcs True \
    --mmcs_type bbox \
    --add_whole_image False \
    --deepspeed ./scripts/zero1.json \
    --vision_model_path "$VISION_MODEL_PATH" \
    --language_model_path "$LANGUAGE_MODEL_PATH" \
    --projector_type patch_merger \
    --data_recipe_path "$PRETRAIN_RECIPE" \
    --output_dir "$PRETRAIN_OUTPUT_DIR" \
    --do_train True \
    --freeze_vit True \
    --freeze_llm True \
    --mm_projector_lr 1e-3 \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 15000 \
    --save_total_limit 5 \
    --save_only_model True \
    --logging_steps 5 \
    --report_to none

torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
    run.py --module train.train \
    --training_stage finetune \
    --dynamic_resolution tile \
    --max_num_tiles 4 \
    --use_lora True \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path "$PRETRAIN_OUTPUT_DIR" \
    --data_packing False \
    --data_recipe_path "$SFT_RECIPE" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --do_train True \
    --freeze_vit True \
    --freeze_llm False \
    --mm_projector_lr 2e-4 \
    --learning_rate 2e-4 \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --model_max_length 8192 \
    --dataloader_num_workers 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --save_only_model False \
    --logging_steps 5 \
    --report_to none
