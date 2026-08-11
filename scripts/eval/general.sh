TASKS=${TASKS:-mmbench_en_dev,mme,mmstar,mmvet,mmmu_val_group_img,gqa}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/general_vqa}

TASKS="$TASKS" OUTPUT_PATH="$OUTPUT_PATH" \
    bash "$(dirname "$0")/all.sh" "$@"
