#!/bin/bash
# 773k MMCS pretraining followed by 779k LLaVA-NeXT LoRA SFT.
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

torchrun --nproc_per_node 8 --master_port $MASTER_PORT \
    run.py --module train.train \
    --training_stage pretrain \
    --dynamic_resolution tile \
    --max_num_tiles 1 \
    --enable_mmcs True \
    --mmcs_type bbox \
    --add_whole_image False \
    --deepspeed ./scripts/zero1.json \
    --vision_model_path google/siglip2-so400m-patch16-384 \
    --language_model_path Qwen/Qwen2.5-3B-Instruct \
    --projector_type patch_merger \
    --data_recipe_path train/recipe/mmcs.json \
    --output_dir checkpoints/siglip2-qwen25-next-3B-pretrain \
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
    --dataloader_num_workers 16 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 15000 \
    --save_total_limit 5 \
    --save_only_model True \
    --logging_steps 5 \
    --report_to none

torchrun --nproc_per_node 8 --master_port $MASTER_PORT \
    run.py --module train.train \
    --training_stage finetune \
    --dynamic_resolution tile \
    --max_num_tiles 4 \
    --use_lora True \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path checkpoints/siglip2-qwen25-next-3B-pretrain \
    --data_packing False \
    --data_recipe_path train/recipe/sft_779k.json \
    --output_dir checkpoints/siglip2-qwen25-next-3B-sft-779k-lora \
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --model_max_length 8192 \
    --dataloader_num_workers 16 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --save_only_model True \
    --logging_steps 5 \
    --report_to none
