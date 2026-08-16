#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_NAME="${DATASET_NAME:-CUB_imsize224}"
NCLS="${NCLS:-200}"
IPC="${IPC:-29}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-256}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
OVERWRITE="${OVERWRITE:-0}"

SRC_DIR="$Main_Data_Path/$DATASET_NAME"
SAVE_DIR="${SAVE_DIR:-$Main_Data_Path/patches/$DATASET_NAME/2}"
CKPT_PATH="$Main_Data_Path/pretrained_models/$DATASET_NAME/ResNet18.pth"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/RDED_patch}"
SPLIT_POINT=$(( (NCLS + 1) / 2 ))

mkdir -p "$SAVE_DIR" "$LOG_DIR"

run_shard() {
    local gpu="$1"
    local class_start="$2"
    local class_end="$3"
    local log_path="$4"

    local overwrite_args=()
    if [[ "$OVERWRITE" == "1" ]]; then
        overwrite_args+=(--overwrite)
    fi

    CUDA_VISIBLE_DEVICES="$gpu" \
    python -u "$SCRIPT_DIR/RDED_patch.py" \
        --dataset-name "$DATASET_NAME" \
        --ncls "$NCLS" \
        --ipc "$IPC" \
        --src-dir "$SRC_DIR" \
        --save-dir "$SAVE_DIR" \
        --ckpt-path "$CKPT_PATH" \
        --model-source torchvision \
        --class-start "$class_start" \
        --class-end "$class_end" \
        --workers "$WORKERS_PER_GPU" \
        --forward-batch-size "$FORWARD_BATCH_SIZE" \
        "${overwrite_args[@]}" \
        > "$log_path" 2>&1
}

echo "Starting RDED shards: GPU $GPU0 classes [0,$SPLIT_POINT), GPU $GPU1 classes [$SPLIT_POINT,$NCLS), overwrite=$OVERWRITE"
run_shard "$GPU0" 0 "$SPLIT_POINT" "$LOG_DIR/gpu${GPU0}_classes_0_${SPLIT_POINT}.log" &
PID0=$!
run_shard "$GPU1" "$SPLIT_POINT" "$NCLS" "$LOG_DIR/gpu${GPU1}_classes_${SPLIT_POINT}_${NCLS}.log" &
PID1=$!

cleanup() {
    kill "$PID0" "$PID1" 2>/dev/null || true
}
trap cleanup INT TERM

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?

if [[ "$STATUS" -ne 0 ]]; then
    echo "At least one RDED shard failed. Inspect $LOG_DIR" >&2
    exit "$STATUS"
fi

FILE_COUNT=$(find "$SAVE_DIR" -type f -name '*.jpg' | wc -l)
EXPECTED_COUNT=$((NCLS * 5))
echo "RDED patch generation complete: $FILE_COUNT/$EXPECTED_COUNT files in $SAVE_DIR"
