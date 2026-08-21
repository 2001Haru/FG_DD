#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RECOVERY_SEEDS_TEXT="${RECOVERY_SEEDS:-41}"; read -r -a RSEEDS <<< "$RECOVERY_SEEDS_TEXT"
STUDENT_SEEDS_TEXT="${STUDENT_SEEDS:-42 43}"; read -r -a SSEEDS <<< "$STUDENT_SEEDS_TEXT"
TEMPERATURE="${TEMPERATURE:-20}"; VIEW_SEED="${VIEW_SEED:-42}"

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/native_sre2l_cifar100_ipc10_50}"
SYN_ROOT="$EXP_ROOT/synthetic"; FKD_ROOT="$EXP_ROOT/fkd"
POST_ROOT="$EXP_ROOT/post_eval"; PER_CLASS="$EXP_ROOT/per_class"
ANALYSIS="$EXP_ROOT/analysis"; LOGS="$ROOT/logs/native_sre2l_cifar100_ipc10_50"
MODEL_POOL="${MODEL_POOL:-$Main_Data_Path/offline_models/cifar100}"
PATCH_ROOT="${PATCH_ROOT:-$Main_Data_Path/patches/cifar100}"
VAL_DIR="${VAL_DIR:-$val_dir/cifar100}"
mkdir -p "$SYN_ROOT" "$FKD_ROOT" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"

fail(){ echo "Native CIFAR100 SRe2L++ failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

[[ -f "$MODEL_POOL/ResNet18.pth" ]] || fail "missing official Teacher: $MODEL_POOL/ResNet18.pth"
[[ -f "$PATCH_ROOT/medium/00000/class00000_id00049.jpg" ]] \
    || fail "official patches do not contain IPC id49: $PATCH_ROOT/medium"
[[ -d "$VAL_DIR" ]] || fail "missing official CIFAR100 validation set: $VAL_DIR"

recover_one(){
    local gpu="$1" ipc="$2" rseed="$3"
    local exp="sre2l_ipc${ipc}_rseed${rseed}"
    local output="$SYN_ROOT/$exp" count=0
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count == 100*ipc )) && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "$exp" --apply-data-augmentation \
        --dataset-name cifar100 --batch-size 100 --syn-data-path "$SYN_ROOT" \
        --patch-dir "$PATCH_ROOT" --model-pool-dir "$MODEL_POOL" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --voter-type equal --selected-size 1 --lr 0.25 --iteration 4000 --r-bn 0.01 \
        --store-best-images --ipc-start 0 --ipc-end "$ipc" --initialisation-method Patches \
        --patch-diff medium --seed "$rseed" --skip-completed \
        > "$LOGS/recover_ipc${ipc}_rseed${rseed}.log" 2>&1
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count == 100*ipc )) || fail "recovery incomplete: $output ($count/$((100*ipc)))"
}

echo "[1/3] Native recovery IPC10/50"
for rseed in "${RSEEDS[@]}"; do
    recover_one "$GPU0" 10 "$rseed" & p0=$!
    recover_one "$GPU1" 50 "$rseed" & p1=$!
    wait_jobs "$p0" "$p1" || fail "recovery seed $rseed"
done

relabel_one(){
    local gpu="$1" ipc="$2" rseed="$3"
    local syn="$SYN_ROOT/sre2l_ipc${ipc}_rseed${rseed}"
    local base="$FKD_ROOT/sre2l_ipc${ipc}_rseed${rseed}"
    local final="${base}_bs16_ipc${ipc}" count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    local expected=$((300*((100*ipc + 15)/16)))
    (( count == expected )) && return
    (( count == 0 )) || fail "partial FKD: $final ($count/$expected)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$MODEL_POOL" --teacher-model-name ResNet18 --gpu 0 \
        --batch-size 16 --workers "$WORKERS" --dataset-name cifar100 --epochs 300 \
        --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_ipc${ipc}_rseed${rseed}.log" 2>&1
}

echo "[2/3] Native relabel"
pids=()
for rseed in "${RSEEDS[@]}"; do for ipc in 10 50; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    relabel_one "$gpu" "$ipc" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
done; done

validate_one(){
    local gpu="$1" ipc="$2" rseed="$3" sseed="$4"
    local syn="$SYN_ROOT/sre2l_ipc${ipc}_rseed${rseed}"
    local fkd="$FKD_ROOT/sre2l_ipc${ipc}_rseed${rseed}_bs16_ipc${ipc}"
    local result="$PER_CLASS/ipc${ipc}_rseed${rseed}_sseed${sseed}.json"
    [[ -f "$result" ]] && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "native_sre2l_ipc${ipc}_rseed${rseed}_sseed${sseed}" \
        --original-data-path "$syn" --fkd-path "$fkd" --output-dir "$POST_ROOT" \
        --batch-size 16 --epochs 300 --dataset-name cifar100 \
        --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --train-seed "$sseed" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$LOGS/validate_ipc${ipc}_rseed${rseed}_sseed${sseed}.log" 2>&1
}

echo "[3/3] Native post-eval"
pids=()
for sseed in "${SSEEDS[@]}"; do for rseed in "${RSEEDS[@]}"; do for ipc in 10 50; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    validate_one "$gpu" "$ipc" "$rseed" "$sseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail post_eval; pids=(); fi
done; done; done

python "$ROOT/class_in_class/summarize_native_sre2l_baselines.py" \
    --per-class-dir "$PER_CLASS" --recovery-seeds "${RSEEDS[@]}" \
    --student-seeds "${SSEEDS[@]}" --output-dir "$ANALYSIS" > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
