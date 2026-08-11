TASKS=${TASKS:-refcoco_bbox_rec,refcoco+_bbox_rec,refcocog_bbox_rec}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/visual_grounding}

TASKS="$TASKS" OUTPUT_PATH="$OUTPUT_PATH" \
    bash "$(dirname "$0")/all.sh" "$@"
