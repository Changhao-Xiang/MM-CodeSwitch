export HF_DATASETS_OFFLINE=1
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export MMBENCH_DISABLE_GPT_EVAL=${MMBENCH_DISABLE_GPT_EVAL:-1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NUM_PROCESSES=${NUM_PROCESSES:-8}
TASKS=${TASKS:-mmvet,mmbench_en_dev,mme,mmstar,mmmu_val_group_img,gqa,ai2d,cvbench,chartqa,ocrbench,textvqa_val,vstar_bench,realworldqa,refcoco_bbox_rec,refcoco+_bbox_rec,refcocog_bbox_rec}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/all}
MODEL=${MODEL:-llava_mmcs}
MODEL_CKPT=${MODEL_CKPT:-checkpoints/siglip2-qwen25-next-3B-sft-779k-lora}

python3 -m accelerate.commands.launch --main_process_port "$MASTER_PORT" \
    --num_processes="$NUM_PROCESSES" \
    -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained="$MODEL_CKPT" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    "$@"
