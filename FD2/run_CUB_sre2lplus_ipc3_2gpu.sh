#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU_RELABEL="${GPU_RELABEL:-0}"
SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-10000}"
RELABEL_WORKERS="${RELABEL_WORKERS:-8}"
VALIDATE_WORKERS="${VALIDATE_WORKERS:-8}"
SQUEEZE_WORKERS="${SQUEEZE_WORKERS:-12}"

DATASET=CUB_imsize224
NUM_CLASSES=200
RECOVER_IPC=5
TARGET_IPC=3
BATCH_SIZE=20
EPOCHS=400

DATASET_DIR="$Main_Data_Path/$DATASET"
TEACHER_DIR="$Main_Data_Path/pretrained_models/$DATASET"
TEACHER="$TEACHER_DIR/SRe2Lplus_plain_ResNet18.pth"
PATCH_DIR="$Main_Data_Path/patches/$DATASET/sre2lplus_plain"
SYN_ROOT="$Main_Data_Path/generated_data/syn_data/SRe2Lplus_${DATASET}"
SYN_IPC5="$SYN_ROOT/rec_res18_ipc5"
SYN_IPC3="$SYN_ROOT/rec_res18_ipc3"
FKD_DIR="$Main_Data_Path/generated_data/new_labels/SRe2Lplus_${DATASET}/rec_res18_ipc3_rel_res18_bs20_ipc3"
OUTPUT_DIR="$Main_Data_Path/generated_data/validate_output"
LOG_DIR="$ROOT_DIR/logs/CUB_sre2lplus_ipc3"

mkdir -p "$TEACHER_DIR" "$SYN_ROOT" "$(dirname "$PATCH_DIR")" "$(dirname "$FKD_DIR")" "$OUTPUT_DIR" "$LOG_DIR"
fail() { echo "Preflight failed: $*" >&2; exit 1; }

[[ -d "$DATASET_DIR/train" ]] || fail "missing CUB train set: $DATASET_DIR/train"
[[ -d "$DATASET_DIR/test" ]] || fail "missing CUB test set: $DATASET_DIR/test"
cat <<EOF
===== Restored CUB SRe2L++ IPC3 protocol =====
Teacher: torchvision ResNet18, ImageNet V1+BN init, SGD
Teacher train: epochs=51, batch=4, lr=1e-3, momentum=0.9, wd=1e-5, workers=$SQUEEZE_WORKERS
Patches: RDED 2x2 composition, 29 real images/class, 5 seeded candidates/class
Recovery: deterministic patch-id, IPC5, Adam, iterations=$ITERATIONS
Recovery loss: CE + 1e-3 * BN, lr=1e-3, jitter=32, first_bn_multiplier=10
Relabel: backbone BSSL, epochs=$EPOCHS, batch=$BATCH_SIZE, CutMix, fp16
Post-eval: random torchvision R18/R50, epochs=$EPOCHS, batch=$BATCH_SIZE
Post-eval optimizer: AdamW lr=1e-3 wd=1e-5, T=20, eta=2, grad_accum=2
Seed: $SEED
================================================
EOF

if [[ ! -f "$TEACHER" || ! -f "$TEACHER.done" ]]; then
    echo "[1/6] Training the restored plain ResNet18 teacher on GPU $GPU0"
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/squeeze/squeeze_plain.py" \
        --dataset-dir "$DATASET_DIR" \
        --output "$TEACHER" \
        --epochs 51 --batch-size 4 --lr 1e-3 --weight-decay 1e-5 --seed "$SEED" \
        --workers "$SQUEEZE_WORKERS" --eval-interval 10 \
        > "$LOG_DIR/squeeze_plain_resnet18.log" 2>&1
else
    echo "[1/6] Reusing plain teacher: $TEACHER"
fi

patch_count=0
[[ -d "$PATCH_DIR" ]] && patch_count="$(find "$PATCH_DIR" -type f -name '*.jpg' | wc -l)"
if (( patch_count != NUM_CLASSES * RECOVER_IPC )); then
    echo "[2/6] Generating seeded plain-teacher RDED patches on two GPUs"
    run_patch_half() {
        local gpu="$1" class_start="$2" class_end="$3" log_file="$4"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT_DIR/RDED_patch.py" \
            --model-name ResNet18 --model-source torchvision \
            --dataset-name "$DATASET" --ncls "$NUM_CLASSES" --ipc 29 --imsize 224 \
            --src-dir "$DATASET_DIR" --save-dir "$PATCH_DIR" --ckpt-path "$TEACHER" \
            --class-start "$class_start" --class-end "$class_end" \
            --patch-num "$RECOVER_IPC" --num-crop 5 --workers 8 \
            --forward-batch-size 256 --model-mode eval --device cuda --seed "$SEED" \
            > "$log_file" 2>&1
    }
    run_patch_half "$GPU0" 0 100 "$LOG_DIR/patches_gpu${GPU0}_classes0_100.log" & patch_pid0=$!
    run_patch_half "$GPU1" 100 200 "$LOG_DIR/patches_gpu${GPU1}_classes100_200.log" & patch_pid1=$!
    patch_status0=0; patch_status1=0
    wait "$patch_pid0" || patch_status0=$?
    wait "$patch_pid1" || patch_status1=$?
    (( patch_status0 == 0 && patch_status1 == 0 )) || \
        fail "patch generation failed: GPU0=$patch_status0 GPU1=$patch_status1"
else
    echo "[2/6] Reusing complete plain-teacher patches: $PATCH_DIR"
fi
patch_count="$(find "$PATCH_DIR" -type f -name '*.jpg' | wc -l)"
(( patch_count == NUM_CLASSES * RECOVER_IPC )) || fail "found $patch_count patches, expected 1000"

run_recover() {
    local gpu="$1" class_start="$2" class_end="$3" log_file="$4"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/recover/recover_sre2lplus.py" \
        --teacher "$TEACHER" \
        --patch-dir "$PATCH_DIR" \
        --output-dir "$SYN_IPC5" \
        --class-start "$class_start" --class-end "$class_end" --class-batch 100 \
        --ipc-start 0 --ipc-end "$RECOVER_IPC" \
        --iterations "$ITERATIONS" --lr 1e-3 --r-bn 1e-3 \
        --first-bn-multiplier 10 --jitter 32 --seed "$SEED" --skip-completed \
        > "$log_file" 2>&1
}

echo "[3/6] Recovering IPC5 on two GPUs"
pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
run_recover "$GPU0" 0 100 "$LOG_DIR/recover_gpu${GPU0}_classes0_100.log" & pid0=$!
run_recover "$GPU1" 100 200 "$LOG_DIR/recover_gpu${GPU1}_classes100_200.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
(( status0 == 0 && status1 == 0 )) || fail "recovery failed: GPU0=$status0 GPU1=$status1"

ipc5_count="$(find "$SYN_IPC5" -type f -name '*.jpg' | wc -l)"
(( ipc5_count == NUM_CLASSES * RECOVER_IPC )) || fail "IPC5 has $ipc5_count images, expected 1000"

echo "[4/6] Sampling deterministic IPC3 from IPC5"
ipc3_count=0
[[ -d "$SYN_IPC3" ]] && ipc3_count="$(find "$SYN_IPC3" -type f -name '*.jpg' | wc -l)"
if (( ipc3_count == 0 )); then
    python "$ROOT_DIR/get_small_ipc_from_big_ipc.py" \
        --source-dir "$SYN_IPC5" --target-dir "$SYN_IPC3" --target-ipc "$TARGET_IPC"
elif (( ipc3_count != NUM_CLASSES * TARGET_IPC )); then
    fail "$SYN_IPC3 contains $ipc3_count images; archive it before rerunning"
fi

expected_batches=$((EPOCHS * NUM_CLASSES * TARGET_IPC / BATCH_SIZE))
fkd_batches=0
[[ -d "$FKD_DIR" ]] && fkd_batches="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
if (( fkd_batches == 0 )); then
    echo "[5/6] Generating backbone-only BSSL on GPU $GPU_RELABEL"
    CUDA_VISIBLE_DEVICES="$GPU_RELABEL" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/relabel/relabel_sre2lplus.py" \
        --syn-data-path "$SYN_IPC3" --teacher "$TEACHER" --fkd-path "$FKD_DIR" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --workers "$RELABEL_WORKERS" \
        --fkd-seed "$SEED" --seed "$SEED" --min-scale 0.08 --use-fp16 --mix-type cutmix \
        > "$LOG_DIR/relabel.log" 2>&1
elif (( fkd_batches != expected_batches )); then
    fail "$FKD_DIR contains $fkd_batches batches, expected $expected_batches; archive it before rerunning"
else
    echo "[5/6] Reusing complete FKD labels: $FKD_DIR"
fi

fkd_batches="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
(( fkd_batches == expected_batches )) || fail "FKD labels incomplete: $fkd_batches/$expected_batches"

run_validate() {
    local gpu="$1" model="$2" log_file="$3"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT_DIR/validate/train_fkd_FD2.py" \
        --model "$model" --model_source torchvision --fkd_source backbone \
        --ipc "$TARGET_IPC" \
        --project "SRe2Lplus_${DATASET}_${model}" \
        --exp_name "SRe2Lplus_rec_res18_ipc3_rel_res18_bs20_val_${model}_seed${SEED}" \
        --original_data_path "$SYN_IPC3" --fkd_path "$FKD_DIR" --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --dataset_name "$DATASET" \
        --gradient_accumulation_steps 2 --mix_type cutmix --cos --eta 2 \
        --workers "$VALIDATE_WORKERS" --temperature 20 --lr 1e-3 --momentum 0.9 \
        --weight_decay 1e-5 --fkd_seed "$SEED" --train_seed "$SEED" \
        --val_dir "$DATASET_DIR/test" > "$log_file" 2>&1
}

echo "[6/6] Validating ResNet18/50 on two GPUs"
run_validate "$GPU0" ResNet18 "$LOG_DIR/validate_resnet18_seed${SEED}.log" & pid0=$!
run_validate "$GPU1" ResNet50 "$LOG_DIR/validate_resnet50_seed${SEED}.log" & pid1=$!
pids=("$pid0" "$pid1")
status0=0; status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
pids=()
trap - INT TERM
(( status0 == 0 && status1 == 0 )) || fail "validation failed: ResNet18=$status0 ResNet50=$status1"

for model in resnet18 resnet50; do
    log_file="$LOG_DIR/validate_${model}_seed${SEED}.log"
    best="$(grep 'Test epoch' "$log_file" | sed -E 's/.*Top1=([0-9.]+).*/\1/' | sort -nr | head -n 1 || true)"
    echo "$model best observed Top1: ${best:-N/A}"
    grep 'Test epoch' "$log_file" | tail -n 1 || true
done

echo "Restored CUB SRe2L++ IPC3 pipeline completed. Logs: $LOG_DIR"
