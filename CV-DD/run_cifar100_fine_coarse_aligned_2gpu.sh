#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RECOVERY_SEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RSEEDS <<< "$RECOVERY_SEEDS_TEXT"
STUDENT_SEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$STUDENT_SEEDS_TEXT"
TEMPERATURE="${TEMPERATURE:-20}"; VIEW_SEED="${VIEW_SEED:-42}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/cifar100_v2_bs100}"
SYN_PARENT="$EXP_ROOT/synthetic"; MODEL_DIR="$EXP_ROOT/models/fine100"
MAPPING="$EXP_ROOT/data/hierarchy.json"; COARSE_TEST="$EXP_ROOT/data/coarse/test"
FKD_PARENT="$EXP_ROOT/relabel_alignment_fkd"; OUTPUT="$EXP_ROOT/relabel_alignment_post_eval"
PER_CLASS="$EXP_ROOT/relabel_alignment_per_class"
LOGS="$ROOT/logs/cifar100_class_in_class_v2_bs100/fine_coarse_aligned"
mkdir -p "$FKD_PARENT" "$OUTPUT" "$PER_CLASS" "$LOGS"
fail(){ echo "Fine-coarse-aligned failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

relabel_one(){
    local gpu="$1" rseed="$2"
    local syn="$SYN_PARENT/seed${rseed}/fine100_coarse_target_ipc25"
    local base="$FKD_PARENT/seed${rseed}/fine_coarse_aligned"
    local final="${base}_bs16_ipc25" count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count==9600 )) && return; (( count==0 )) || fail "partial FKD $final"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$MODEL_DIR" --teacher-model-name ResNet18 --teacher-num-classes 100 \
        --teacher-mapping "$MAPPING" --marginalize-temperature "$TEMPERATURE" \
        --gpu 0 --batch-size 16 --workers "$WORKERS" --dataset-name cifar20 --epochs 300 \
        --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_rseed${rseed}.log" 2>&1
}

pids=()
for rseed in "${RSEEDS[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    relabel_one "$gpu" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail relabel; fi

validate_one(){
    local gpu="$1" rseed="$2" sseed="$3"
    local syn="$SYN_PARENT/seed${rseed}/fine100_coarse_target_ipc25"
    local fkd="$FKD_PARENT/seed${rseed}/fine_coarse_aligned_bs16_ipc25"
    local result="$PER_CLASS/fine_coarse_aligned_rseed${rseed}_sseed${sseed}.json"
    [[ -f "$result" ]] && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 25 \
        --exp-name "fine_coarse_aligned_rseed${rseed}_sseed${sseed}" \
        --original-data-path "$syn" --fkd-path "$fkd" --output-dir "$OUTPUT" \
        --batch-size 16 --epochs 300 --dataset-name cifar20 --gradient-accumulation-steps 2 \
        --mix-type cutmix --cos --workers "$WORKERS" --temperature "$TEMPERATURE" \
        --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 --train-seed "$sseed" \
        --persistent-workers --val-dir "$COARSE_TEST" --disable-wandb \
        --per-class-output "$result" > "$LOGS/validate_rseed${rseed}_sseed${sseed}.log" 2>&1
}

pids=()
for sseed in "${SSEEDS[@]}"; do for rseed in "${RSEEDS[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    validate_one "$gpu" "$rseed" "$sseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail post_eval; pids=(); fi
done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail post_eval; fi

python "$ROOT/class_in_class/summarize_fine_coarse_aligned.py" \
    --experiment-root "$EXP_ROOT" --recovery-seeds "${RSEEDS[@]}" \
    --student-seeds "${SSEEDS[@]}" --output-dir "$EXP_ROOT/analysis/fine_coarse_aligned" \
    > "$LOGS/summary.log" 2>&1
echo "Complete: $EXP_ROOT/analysis/fine_coarse_aligned/summary.json"
