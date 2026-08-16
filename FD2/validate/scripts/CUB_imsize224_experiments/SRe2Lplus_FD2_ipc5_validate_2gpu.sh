#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FD2_DIR="$(dirname "$VALIDATE_DIR")"
source "$SCRIPT_DIR/constants.sh"

GPU_RESNET18="${GPU_RESNET18:-0}"
GPU_RESNET50="${GPU_RESNET50:-1}"
WORKERS_PER_RUN="${WORKERS_PER_RUN:-2}"
EPOCHS="${EPOCHS:-400}"
IPC="${IPC:-5}"
BATCH_SIZE=20
REC_NAME="rec_res18"
REL_NAME="rel_res18"

SYN_DIR="${Generated_Data_Path}/syn_data/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${IPC}"
FKD_DIR="${Generated_Data_Path}/new_labels/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${IPC}_${REL_NAME}_bs${BATCH_SIZE}_ipc${IPC}"
OUTPUT_DIR="${Generated_Data_Path}/validate_output"
LOG_DIR="${SCRIPT_DIR}/logs/validate_ipc${IPC}_2gpu"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

fail() {
    echo "Preflight failed: $*" >&2
    exit 1
}

[[ -d "$SYN_DIR" ]] || fail "missing synthetic dataset: $SYN_DIR"
[[ -d "$FKD_DIR" ]] || fail "missing FKD labels: $FKD_DIR"
[[ -d "$val_dir" ]] || fail "missing validation dataset: $val_dir"

syn_count="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
epoch_count="$(find "$FKD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | wc -l)"
batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
val_count="$(find "$val_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"

expected_images=$((200 * IPC))
(( syn_count == expected_images )) || fail "found $syn_count synthetic JPGs, expected $expected_images"
(( epoch_count >= EPOCHS )) || fail "found $epoch_count FKD epochs, need at least $EPOCHS"
if (( expected_images % BATCH_SIZE != 0 )); then
    fail "$expected_images images are not divisible by batch size $BATCH_SIZE"
fi
expected_batches=$((EPOCHS * expected_images / BATCH_SIZE))
(( batch_count >= expected_batches )) || fail "found $batch_count FKD batches, need at least $expected_batches"
(( val_count > 0 )) || fail "no validation images found under $val_dir"

run_validation() {
    local gpu="$1"
    local model="$2"
    local val_name="$3"
    local log_file="$4"
    local exp_name="SRe2Lplus_FD2_${REC_NAME}_ipc${IPC}_${REL_NAME}_bs${BATCH_SIZE}_${val_name}_09FC05_01SC4"
    local project="SRe2Lplus_FD2_${Dataset_Name}_${val_name}"

    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    python -u "$VALIDATE_DIR/train_fkd_FD2.py" \
        --model "$model" \
        --model_source torchvision \
        --fkd_source backbone \
        --ipc "$IPC" \
        --matplotlib \
        --project "$project" \
        --exp_name "$exp_name" \
        --original_data_path "$SYN_DIR" \
        --fkd_path "$FKD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --dataset_name "$Dataset_Name" \
        --gradient_accumulation_steps 2 \
        --mix_type cutmix \
        --cos \
        --eta 2.0 \
        --workers "$WORKERS_PER_RUN" \
        --temperature 20 \
        --lr 1e-3 \
        --momentum 0.9 \
        --weight_decay 1e-5 \
        --val_dir "$val_dir" > "$log_file" 2>&1
}

summarize_log() {
    local model="$1"
    local log_file="$2"
    local final_line
    local best_top1
    final_line="$(grep 'Test epoch' "$log_file" | tail -n 1 || true)"
    best_top1="$(grep 'Test epoch' "$log_file" | sed -E 's/.*Top1=([0-9.]+).*/\1/' | sort -nr | head -n 1 || true)"
    echo "$model best observed Top1: ${best_top1:-N/A}"
    echo "$model final: ${final_line:-no validation result found}"
}

cd "$FD2_DIR"
log18="$LOG_DIR/resnet18.log"
log50="$LOG_DIR/resnet50.log"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

echo "Starting ResNet18 on GPU $GPU_RESNET18 and ResNet50 on GPU $GPU_RESNET50"
echo "Logs: $log18 and $log50"
run_validation "$GPU_RESNET18" ResNet18 val_ResNet18 "$log18" &
pid18=$!
run_validation "$GPU_RESNET50" ResNet50 val_ResNet50 "$log50" &
pid50=$!
pids=("$pid18" "$pid50")

status18=0
status50=0
wait "$pid18" || status18=$?
wait "$pid50" || status50=$?
pids=()
trap - INT TERM

summarize_log ResNet18 "$log18"
summarize_log ResNet50 "$log50"

if (( status18 != 0 || status50 != 0 )); then
    echo "Validation failed: ResNet18 status=$status18, ResNet50 status=$status50" >&2
    exit 1
fi

echo "Both validation runs completed successfully."
