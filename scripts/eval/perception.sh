TASKS=${TASKS:-ai2d,chartqa,ocrbench,textvqa_val,cvbench,realworldqa,vstar_bench}
OUTPUT_PATH=${OUTPUT_PATH:-./logs/perception_centric}

TASKS="$TASKS" OUTPUT_PATH="$OUTPUT_PATH" \
    bash "$(dirname "$0")/all.sh" "$@"
