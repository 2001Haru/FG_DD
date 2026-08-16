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
IPC="${IPC:-3}"
STUDENT_INIT="${STUDENT_INIT:-random}"
BATCH_SIZE=20
REC_NAME=rec_res18
REL_NAME=rel_res18

SYN_DIR="${Generated_Data_Path}/syn_data/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${IPC}"
FKD_DIR="${Generated_Data_Path}/new_labels/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4/${REC_NAME}_ipc${IPC}_${REL_NAME}_bs${BATCH_SIZE}_ipc${IPC}"
OUTPUT_DIR="${Generated_Data_Path}/validate_output"
case "$STUDENT_INIT" in
    random)
        INIT_ARGS=()
        INIT_TAG=random
        ;;
    imagenet)
        INIT_ARGS=(--pretrained_weights --pretrained_bn)
        INIT_TAG=imagenet
        ;;
    *)
        echo "STUDENT_INIT must be 'random' or 'imagenet', got: $STUDENT_INIT" >&2
        exit 1
        ;;
esac

LOG_DIR="${SCRIPT_DIR}/logs/validate_ipc${IPC}_${INIT_TAG}_2gpu"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

fail() { echo "Preflight failed: $*" >&2; exit 1; }
[[ -d "$SYN_DIR" ]] || fail "missing synthetic dataset: $SYN_DIR"
[[ -d "$FKD_DIR" ]] || fail "missing FKD labels: $FKD_DIR"
[[ -d "$val_dir" ]] || fail "missing Aircraft test set: $val_dir"

syn_count="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
epoch_count="$(find "$FKD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | wc -l)"
batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
expected_images=$((100 * IPC))
expected_batches=$((EPOCHS * expected_images / BATCH_SIZE))
(( syn_count == expected_images )) || fail "found $syn_count synthetic JPGs, expected $expected_images"
(( epoch_count >= EPOCHS )) || fail "found $epoch_count FKD epochs, expected at least $EPOCHS"
(( expected_images % BATCH_SIZE == 0 )) || fail "$expected_images images are not divisible by batch size $BATCH_SIZE"
(( batch_count >= expected_batches )) || fail "found $batch_count FKD batches, expected at least $expected_batches"

run_validation() {
    local gpu="$1" model="$2" log_file="$3"
    local val_name="val_${model}_${INIT_TAG}"
    local exp_name="SRe2Lplus_FD2_${REC_NAME}_ipc${IPC}_${REL_NAME}_bs${BATCH_SIZE}_${val_name}_09FC05_01SC4"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    python -u "$VALIDATE_DIR/train_fkd_FD2.py" \
        --model "$model" \
        --model_source torchvision \
        "${INIT_ARGS[@]}" \
        --fkd_source backbone \
        --ipc "$IPC" \
        --matplotlib \
        --project "SRe2Lplus_FD2_${Dataset_Name}_${val_name}" \
        --exp_name "$exp_name" \
        --original_data_path "$SYN_DIR" \
        --fkd_path "$FKD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --dataset_name "$Dataset_Name" \
        --gradient_accumulation_steps 2 \
        --mix_type cutmix \
        --cos --eta 2.0 \
        --workers "$WORKERS_PER_RUN" \
        --temperature 20 \
        --lr 1e-3 \
        --momentum 0.9 \
        --weight_decay 1e-5 \
        --val_dir "$val_dir" > "$log_file" 2>&1
}

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM

log18="$LOG_DIR/resnet18.log"
log50="$LOG_DIR/resnet50.log"
echo "Aircraft IPC${IPC}: student_init=${STUDENT_INIT}; ResNet18 on GPU ${GPU_RESNET18}, ResNet50 on GPU ${GPU_RESNET50}"
run_validation "$GPU_RESNET18" ResNet18 "$log18" & pid18=$!
run_validation "$GPU_RESNET50" ResNet50 "$log50" & pid50=$!
pids=("$pid18" "$pid50")
status18=0; status50=0
wait "$pid18" || status18=$?
wait "$pid50" || status50=$?
pids=()
trap - INT TERM

for item in "ResNet18:$log18" "ResNet50:$log50"; do
    model="${item%%:*}"; log_file="${item#*:}"
    best="$(grep 'Test epoch' "$log_file" | sed -E 's/.*Top1=([0-9.]+).*/\1/' | sort -nr | head -n 1 || true)"
    final="$(grep 'Test epoch' "$log_file" | tail -n 1 || true)"
    echo "$model best observed Top1: ${best:-N/A}"
    echo "$model final: ${final:-no validation result}"
done

(( status18 == 0 && status50 == 0 )) || { echo "Validation failed: R18=$status18 R50=$status50" >&2; exit 1; }
echo "Aircraft IPC${IPC} validation completed."
