#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FD2_DIR="$(dirname "$VALIDATE_DIR")"
source "$FD2_DIR/config.sh"

GPU_RESNET18="${GPU_RESNET18:-0}"
GPU_RESNET50="${GPU_RESNET50:-1}"
WORKERS_PER_RUN="${WORKERS_PER_RUN:-4}"
EPOCHS="${EPOCHS:-300}"
PATCH_DIR="${PATCH_DIR:-${Main_Data_Path}/patches/CUB_imsize224/2}"
VAL_DIR="${Main_Data_Path}/CUB_imsize224/test"
MODEL_DIR="${Main_Data_Path}/pretrained_models/CUB_imsize224"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/rded_ipc3_2gpu}"
mkdir -p "$LOG_DIR"

fail() {
    echo "RDED preflight failed: $*" >&2
    exit 1
}

[[ -d "$PATCH_DIR" ]] || fail "missing patches: $PATCH_DIR"
[[ -d "$VAL_DIR" ]] || fail "missing validation data: $VAL_DIR"
patch_count="$(find "$PATCH_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
(( patch_count >= 600 )) || fail "found only $patch_count patch images; IPC3 needs 600"

run_eval() {
    local gpu="$1"
    local model="$2"
    local log="$3"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$VALIDATE_DIR/train_rded_CUB.py" \
        --syn-data-path "$PATCH_DIR" \
        --val-dir "$VAL_DIR" \
        --model-pool-dir "$MODEL_DIR" \
        --model "$model" \
        --ipc 3 \
        --epochs "$EPOCHS" \
        --batch-size 20 \
        --workers "$WORKERS_PER_RUN" > "$log" 2>&1
}

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

log18="$LOG_DIR/resnet18.log"
log50="$LOG_DIR/resnet50.log"
echo "Starting RDED IPC3: ResNet18 on GPU $GPU_RESNET18, ResNet50 on GPU $GPU_RESNET50"
echo "Evaluation epochs: $EPOCHS"
echo "Logs: $log18 and $log50"
run_eval "$GPU_RESNET18" ResNet18 "$log18" & pid18=$!
run_eval "$GPU_RESNET50" ResNet50 "$log50" & pid50=$!
pids=("$pid18" "$pid50")

status18=0
status50=0
wait "$pid18" || status18=$?
wait "$pid50" || status50=$?
pids=()
trap - INT TERM

tail -n 1 "$log18" || true
tail -n 1 "$log50" || true
if (( status18 != 0 || status50 != 0 )); then
    echo "RDED evaluation failed: ResNet18=$status18 ResNet50=$status50" >&2
    exit 1
fi
echo "RDED IPC3 two-GPU evaluation completed."
