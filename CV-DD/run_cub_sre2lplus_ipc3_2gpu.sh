#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD}"
CUB_DATA_DIR="${CUB_DATA_DIR:-/linxi/dataset/FD2/CUB_imsize224}"
PATCH_ROOT="${PATCH_ROOT:-/linxi/dataset/FD2/patches/CUB_imsize224}"
TEACHER_SOURCE="${TEACHER_SOURCE:-/linxi/dataset/FD2/pretrained_models/CUB_imsize224/ResNet18_M8_5e-1cal.pth}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_RELABEL="${GPU_RELABEL:-0}"
RECOVER_ITERATIONS="${RECOVER_ITERATIONS:-4000}"
RECOVER_BATCH_SIZE="${RECOVER_BATCH_SIZE:-100}"
RELABEL_WORKERS="${RELABEL_WORKERS:-8}"
VALIDATE_WORKERS="${VALIDATE_WORKERS:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EPOCHS="${EPOCHS:-400}"
IPC=3
NUM_CLASSES=200
FKD_BATCH_SIZE=20

MODEL_DIR="$DATA_ROOT/offline_models/CUB_imsize224"
TEACHER="$MODEL_DIR/ResNet18.pth"
SYN_BASE="$DATA_ROOT/generated_data/syn_data/CUB_imsize224"
SYN_IPC5="$SYN_BASE/sre2l_ipc5"
SYN_IPC3="$SYN_BASE/sre2l_ipc3"
FKD_BASE="$DATA_ROOT/generated_data/new_labels/CUB_imsize224/sre2l_ipc3"
FKD_DIR="${FKD_BASE}_bs${FKD_BATCH_SIZE}_ipc${IPC}"
OUTPUT_DIR="$DATA_ROOT/generated_data/validate_output"
LOG_DIR="$ROOT_DIR/logs/CUB_imsize224_sre2l_ipc3"
VAL_DIR="$CUB_DATA_DIR/test"

mkdir -p "$MODEL_DIR" "$SYN_BASE" "$(dirname "$FKD_BASE")" "$OUTPUT_DIR" "$LOG_DIR"

fail() { echo "Preflight failed: $*" >&2; exit 1; }
[[ -d "$VAL_DIR" ]] || fail "missing CUB test set: $VAL_DIR"
[[ -d "$PATCH_ROOT/2" ]] || fail "missing CUB patch directory: $PATCH_ROOT/2"
[[ -f "$TEACHER_SOURCE" ]] || fail "missing teacher checkpoint: $TEACHER_SOURCE"

patch_count="$(find "$PATCH_ROOT/2" -type f -name '*.jpg' | wc -l)"
(( patch_count == NUM_CLASSES * 5 )) || fail "found $patch_count patches, expected $((NUM_CLASSES * 5))"

if [[ ! -f "$TEACHER" ]]; then
    python "$ROOT_DIR/tools/export_cub_teacher.py" \
        --source "$TEACHER_SOURCE" --output "$TEACHER"
fi

run_recover() {
    local gpu="$1" ipc_start="$2" ipc_end="$3" log_file="$4"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/recover/recover.py" \
        --exp-name sre2l_ipc5 \
        --apply-data-augmentation \
        --dataset-name CUB_imsize224 \
        --batch-size "$RECOVER_BATCH_SIZE" \
        --syn-data-path "$SYN_BASE" \
        --patch-dir "$PATCH_ROOT" \
        --model-pool-dir "$MODEL_DIR" \
        --pretrained-model-type offline \
        --model-setting 0 \
        --sre2l-model ResNet18 \
        --voter-type equal \
        --selected-size 1 \
        --lr 0.25 \
        --iteration "$RECOVER_ITERATIONS" \
        --r-bn 0.01 \
        --store-best-images \
        --skip-completed \
        --ipc-start "$ipc_start" \
        --ipc-end "$ipc_end" \
        --initialisation-method Patches \
        --patch-diff 2 > "$log_file" 2>&1
}

echo "[1/4] Recovering native SRe2L++ IPC5 with CV-DD (iterations=$RECOVER_ITERATIONS)"
pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
run_recover "$GPU0" 0 3 "$LOG_DIR/recover_gpu${GPU0}_ipc0_3.log" & pid0=$!
run_recover "$GPU1" 3 5 "$LOG_DIR/recover_gpu${GPU1}_ipc3_5.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
(( status0 == 0 && status1 == 0 )) || fail "recovery failed: GPU0=$status0 GPU1=$status1; see $LOG_DIR"

ipc5_count="$(find "$SYN_IPC5" -type f -name '*.jpg' | wc -l)"
(( ipc5_count == NUM_CLASSES * 5 )) || fail "recovered IPC5 has $ipc5_count images, expected $((NUM_CLASSES * 5))"

echo "[2/4] Sampling IPC3 from IPC5"
ipc3_count=0
[[ -d "$SYN_IPC3" ]] && ipc3_count="$(find "$SYN_IPC3" -type f -name '*.jpg' | wc -l)"
if (( ipc3_count == 0 )); then
    python "$ROOT_DIR/tools/sample_ipc.py" \
        --source "$SYN_IPC5" --target "$SYN_IPC3" --ipc "$IPC" --classes "$NUM_CLASSES"
elif (( ipc3_count != NUM_CLASSES * IPC )); then
    fail "$SYN_IPC3 contains $ipc3_count images; archive it before rerunning"
fi

echo "[3/4] Generating 400-epoch batch-specific soft labels"
epoch_count=0; batch_count=0
if [[ -d "$FKD_DIR" ]]; then
    epoch_count="$(find "$FKD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | wc -l)"
    batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
fi
expected_batches=$((EPOCHS * NUM_CLASSES * IPC / FKD_BATCH_SIZE))
if (( epoch_count < EPOCHS || batch_count < expected_batches )); then
    CUDA_VISIBLE_DEVICES="$GPU_RELABEL" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/relabel/relabel.py" \
        --syn-data-path "$SYN_IPC3" \
        --fkd-path "$FKD_BASE" \
        --model-pool-dir "$MODEL_DIR" \
        --teacher-model-name ResNet18 \
        --gpu 0 \
        --batch-size "$FKD_BATCH_SIZE" \
        --workers "$RELABEL_WORKERS" \
        --dataset-name CUB_imsize224 \
        --epochs "$EPOCHS" \
        --fkd-seed 42 \
        --min-scale-crops 0.08 \
        --max-scale-crops 1 \
        --use-fp16 \
        --mode fkd_save \
        --mix-type cutmix > "$LOG_DIR/relabel.log" 2>&1
fi

batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
(( batch_count == expected_batches )) || fail "FKD labels have $batch_count batches, expected $expected_batches"

run_validate() {
    local gpu="$1" model="$2" log_file="$3"
    local worker_args=()
    if [[ "$PERSISTENT_WORKERS" == "1" ]]; then
        worker_args+=(--persistent-workers)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/validate/train_fkd.py" \
        --model "$model" \
        --ipc "$IPC" \
        --exp-name "sre2l_ipc${IPC}_${model}" \
        --original-data-path "$SYN_IPC3" \
        --fkd-path "$FKD_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$FKD_BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --dataset-name CUB_imsize224 \
        --gradient-accumulation-steps 2 \
        --mix-type cutmix \
        --cos \
        --workers "$VALIDATE_WORKERS" \
        "${worker_args[@]}" \
        --train-seed "$TRAIN_SEED" \
        --temperature 20 \
        --adamw-weight-decay 1e-5 \
        --val-dir "$VAL_DIR" \
        --disable-wandb > "$log_file" 2>&1
}

echo "[4/4] Validating random-init ResNet18 and ResNet50 on two GPUs"
run_validate "$GPU0" ResNet18 "$LOG_DIR/validate_resnet18.log" & pid0=$!
run_validate "$GPU1" ResNet50 "$LOG_DIR/validate_resnet50.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
trap - INT TERM
(( status0 == 0 && status1 == 0 )) || fail "validation failed: ResNet18=$status0 ResNet50=$status1"

for model in resnet18 resnet50; do
    log_file="$LOG_DIR/validate_${model}.log"
    best="$(grep 'TEST Iter' "$log_file" | sed -E 's/.*Top-1 err = ([0-9.]+).*/\1/' | awk 'BEGIN { best=-1 } { acc=100-$1; if (acc>best) best=acc } END { if (best>=0) printf "%.2f", best }')"
    echo "$model best Top1: ${best:-N/A}"
    grep 'TEST Iter' "$log_file" | tail -n 1 || true
done

echo "CUB IPC3 native SRe2L++ pipeline completed. Logs: $LOG_DIR"
