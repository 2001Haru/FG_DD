#!/usr/bin/env bash
set -euo pipefail

FD2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FD2_DIR/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-10}"
DATASET=A_imsize224
SRC_DIR="${Main_Data_Path}/${DATASET}"
SAVE_DIR="${Main_Data_Path}/patches/${DATASET}/2"
CKPT="${Main_Data_Path}/pretrained_models/${DATASET}/ResNet18_M32_3e-1cal.pth"
LOG_DIR="$FD2_DIR/recover/scripts/A_imsize224_experiments/logs/rded_patches_2gpu"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[[ -d "$SRC_DIR/train" ]] || { echo "Missing Aircraft train set: $SRC_DIR/train" >&2; exit 1; }
[[ -f "$CKPT" ]] || { echo "Missing Aircraft teacher: $CKPT" >&2; exit 1; }

run_half() {
    local gpu="$1" start="$2" end="$3" log="$4"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$FD2_DIR/RDED_patch.py" \
        --dataset-name "$DATASET" --ncls 100 --ipc 33 --imsize 224 \
        --model-name ResNet18 --model-source auto \
        --src-dir "$SRC_DIR" --save-dir "$SAVE_DIR" --ckpt-path "$CKPT" \
        --class-start "$start" --class-end "$end" \
        --patch-num 5 --num-crop 5 --workers "$WORKERS_PER_GPU" \
        --forward-batch-size 256 --device cuda > "$log" 2>&1
}

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
run_half "$GPU0" 0 50 "$LOG_DIR/gpu${GPU0}_classes_0_50.log" & pid0=$!
run_half "$GPU1" 50 100 "$LOG_DIR/gpu${GPU1}_classes_50_100.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?; wait "$pid1" || status1=$?
pids=()
trap - INT TERM
(( status0 == 0 && status1 == 0 )) || { echo "Patch generation failed: $status0/$status1" >&2; exit 1; }
count="$(find "$SAVE_DIR" -type f -name '*.jpg' | wc -l)"
(( count == 500 )) || { echo "Patch set incomplete: $count/500" >&2; exit 1; }
echo "Aircraft RDED initialization complete: $count patches in $SAVE_DIR"
