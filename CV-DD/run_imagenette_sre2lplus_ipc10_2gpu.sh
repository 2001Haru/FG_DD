#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$DATA_ROOT/_archives}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_RELABEL="${GPU_RELABEL:-0}"
RECOVER_ITERATIONS="${RECOVER_ITERATIONS:-4000}"
RELABEL_WORKERS="${RELABEL_WORKERS:-2}"
VALIDATE_WORKERS="${VALIDATE_WORKERS:-8}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

DATASET=imagenet-nette
NUM_CLASSES=10
RECOVER_IPC=50
TARGET_IPC=10
EPOCHS=300
BATCH_SIZE=10

MODEL_DIR="$DATA_ROOT/offline_models/$DATASET"
PATCH_DIR="$DATA_ROOT/patches/$DATASET"
VAL_DIR="$DATA_ROOT/test_data/$DATASET/test"
SYN_BASE="$DATA_ROOT/generated_data/syn_data/$DATASET"
SYN_IPC50="$SYN_BASE/sre2l_ipc50"
SYN_IPC10="$SYN_BASE/sre2l_ipc10"
FKD_BASE="$DATA_ROOT/generated_data/new_labels/$DATASET/sre2l"
FKD_DIR="${FKD_BASE}_bs${BATCH_SIZE}_ipc${TARGET_IPC}"
OUTPUT_DIR="$DATA_ROOT/generated_data/validate_output"
LOG_DIR="$ROOT_DIR/logs/imagenette_sre2l_ipc10"

if [[ "$PERSISTENT_WORKERS" == "1" ]]; then
    LOADER_TAG=persistent
else
    LOADER_TAG=released
fi
if [[ -n "$TRAIN_SEED" ]]; then
    RUN_TAG="${LOADER_TAG}_seed${TRAIN_SEED}"
else
    RUN_TAG="${LOADER_TAG}_unseeded"
fi

mkdir -p "$DATA_ROOT/offline_models" "$DATA_ROOT/patches" "$DATA_ROOT/test_data" \
         "$SYN_BASE" "$(dirname "$FKD_BASE")" "$OUTPUT_DIR" "$LOG_DIR"

fail() { echo "Preflight failed: $*" >&2; exit 1; }

extract_if_needed() {
    local target="$1" archive="$2" destination="$3"
    if [[ ! -d "$target" ]]; then
        [[ -f "$archive" ]] || fail "missing archive: $archive"
        echo "Extracting $(basename "$archive")"
        tar -xzf "$archive" -C "$destination"
    fi
}

extract_if_needed "$MODEL_DIR" "$ARCHIVE_DIR/imagenet-nette.tar.gz" "$DATA_ROOT/offline_models"
extract_if_needed "$PATCH_DIR" "$ARCHIVE_DIR/imagenet-nette-patches.tar.gz" "$DATA_ROOT/patches"
extract_if_needed "$DATA_ROOT/test_data/$DATASET" "$ARCHIVE_DIR/imagenet-nette-test.tar.gz" "$DATA_ROOT/test_data"

for model in ResNet18 ResNet50 Densenet121 ShuffleNetV2 MobileNetV2; do
    [[ -f "$MODEL_DIR/$model.pth" ]] || fail "missing model: $MODEL_DIR/$model.pth"
done
patch_count="$(find "$PATCH_DIR/medium" -type f -name '*.jpg' | wc -l)"
test_count="$(find "$VAL_DIR" -type f | wc -l)"
(( patch_count == NUM_CLASSES * RECOVER_IPC )) || fail "found $patch_count patches, expected 500"
(( test_count == 3925 )) || fail "found $test_count test images, expected 3925"

run_recover() {
    local gpu="$1" ipc_start="$2" ipc_end="$3" log_file="$4"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/recover/recover.py" \
        --exp-name sre2l_ipc50 \
        --apply-data-augmentation \
        --dataset-name "$DATASET" \
        --batch-size 10 \
        --syn-data-path "$SYN_BASE" \
        --patch-dir "$PATCH_DIR" \
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
        --patch-diff medium > "$log_file" 2>&1
}

expected_batches=$((EPOCHS * NUM_CLASSES * TARGET_IPC / BATCH_SIZE))

if [[ "$VALIDATE_ONLY" != "1" ]]; then
echo "[1/4] Recovering official SRe2L++ IPC50 on two GPUs"
pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
run_recover "$GPU0" 0 25 "$LOG_DIR/recover_gpu${GPU0}_ipc0_25.log" & pid0=$!
run_recover "$GPU1" 25 50 "$LOG_DIR/recover_gpu${GPU1}_ipc25_50.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
(( status0 == 0 && status1 == 0 )) || fail "recovery failed: GPU0=$status0 GPU1=$status1"

ipc50_count="$(find "$SYN_IPC50" -type f -name '*.jpg' | wc -l)"
(( ipc50_count == NUM_CLASSES * RECOVER_IPC )) || fail "IPC50 has $ipc50_count images, expected 500"

echo "[2/4] Sampling IPC10"
ipc10_count=0
[[ -d "$SYN_IPC10" ]] && ipc10_count="$(find "$SYN_IPC10" -type f -name '*.jpg' | wc -l)"
if (( ipc10_count == 0 )); then
    python "$ROOT_DIR/tools/sample_ipc.py" \
        --source "$SYN_IPC50" --target "$SYN_IPC10" \
        --ipc "$TARGET_IPC" --classes "$NUM_CLASSES"
elif (( ipc10_count != NUM_CLASSES * TARGET_IPC )); then
    fail "$SYN_IPC10 contains $ipc10_count images; archive it before rerunning"
fi

echo "[3/4] Generating official 300-epoch BSSL labels (batch size 10)"
batch_count=0
[[ -d "$FKD_DIR" ]] && batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
if (( batch_count < expected_batches )); then
    CUDA_VISIBLE_DEVICES="$GPU_RELABEL" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/relabel/relabel.py" \
        --syn-data-path "$SYN_IPC10" \
        --fkd-path "$FKD_BASE" \
        --model-pool-dir "$MODEL_DIR" \
        --teacher-model-name ResNet18 \
        --gpu 0 \
        --batch-size "$BATCH_SIZE" \
        --workers "$RELABEL_WORKERS" \
        --dataset-name "$DATASET" \
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
else
    echo "[1-3/4] VALIDATE_ONLY=1: reusing existing IPC10 images and FKD labels"
    ipc10_count="$(find "$SYN_IPC10" -type f -name '*.jpg' | wc -l)"
    (( ipc10_count == NUM_CLASSES * TARGET_IPC )) || \
        fail "IPC10 has $ipc10_count images, expected $((NUM_CLASSES * TARGET_IPC))"
    batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    (( batch_count == expected_batches )) || \
        fail "FKD labels have $batch_count batches, expected $expected_batches"
fi

run_validate() {
    local gpu="$1" model="$2" log_file="$3"
    local optional_args=()
    [[ "$PERSISTENT_WORKERS" == "1" ]] && optional_args+=(--persistent-workers)
    [[ -n "$TRAIN_SEED" ]] && optional_args+=(--train-seed "$TRAIN_SEED")
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/validate/train_fkd.py" \
        --model "$model" \
        --ipc "$TARGET_IPC" \
        --exp-name "sre2l_ipc10_${model}_${RUN_TAG}" \
        --original-data-path "$SYN_IPC10" \
        --fkd-path "$FKD_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --dataset-name "$DATASET" \
        --gradient-accumulation-steps 2 \
        --mix-type cutmix \
        --cos \
        --workers "$VALIDATE_WORKERS" \
        --temperature 20 \
        --val-dir "$VAL_DIR" \
        --disable-wandb \
        "${optional_args[@]}" > "$log_file" 2>&1
}

echo "[4/4] Validating ResNet18/50: loader=$LOADER_TAG, workers=$VALIDATE_WORKERS, seed=${TRAIN_SEED:-unset}"
run_validate "$GPU0" ResNet18 "$LOG_DIR/validate_resnet18_${RUN_TAG}.log" & pid0=$!
run_validate "$GPU1" ResNet50 "$LOG_DIR/validate_resnet50_${RUN_TAG}.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
trap - INT TERM
(( status0 == 0 && status1 == 0 )) || fail "validation failed: ResNet18=$status0 ResNet50=$status1"

for model in resnet18 resnet50; do
    log_file="$LOG_DIR/validate_${model}_${RUN_TAG}.log"
    best="$(grep 'TEST Iter' "$log_file" | sed -E 's/.*Top-1 err = ([0-9.]+).*/\1/' | awk 'BEGIN { best=-1 } { acc=100-$1; if (acc>best) best=acc } END { if (best>=0) printf "%.2f", best }')"
    echo "$model best Top1: ${best:-N/A}"
    grep 'TEST Iter' "$log_file" | tail -n 1 || true
done

echo "Repository snapshot targets (stale/v1): ResNet18=62.4, ResNet50=57.4"
echo "Current arXiv v2 targets: ResNet18=73.7, ResNet50=72.6"
echo "ImageNette IPC10 experiment completed. Logs: $LOG_DIR"
