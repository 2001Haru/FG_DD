#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RECOVERY_SEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RECOVERY_SEEDS_ARRAY <<< "$RECOVERY_SEEDS_TEXT"
EXTRA_STUDENT_SEEDS_TEXT="${EXTRA_STUDENT_SEEDS:-43 44}"; read -r -a EXTRA_STUDENT_SEEDS_ARRAY <<< "$EXTRA_STUDENT_SEEDS_TEXT"
ALL_STUDENT_SEEDS_TEXT="${ALL_STUDENT_SEEDS:-42 43 44}"; read -r -a ALL_STUDENT_SEEDS_ARRAY <<< "$ALL_STUDENT_SEEDS_TEXT"
RANDOM_PARTITION_SEED="${RANDOM_PARTITION_SEED:-42}"

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/cifar100_v2_bs100}"
SYN_PARENT="$EXP_ROOT/synthetic"; FKD_PARENT="$EXP_ROOT/fkd"
OUTPUT="$EXP_ROOT/post_eval"; PER_CLASS="$EXP_ROOT/per_class"
COARSE_TEST="$EXP_ROOT/data/coarse/test"
LOGS="$ROOT/logs/cifar100_class_in_class_v2_bs100/post_eval_extra_seeds"
ANALYSIS="$EXP_ROOT/analysis/post_eval_3x3"
mkdir -p "$OUTPUT" "$PER_CLASS" "$LOGS" "$ANALYSIS"

fail() { echo "Post-eval preflight failed: $*" >&2; exit 1; }
wait_jobs() {
    local status=0 pid
    for pid in "$@"; do wait "$pid" || status=$?; done
    return "$status"
}

[[ -d "$COARSE_TEST" ]] || fail "missing coarse validation set: $COARSE_TEST"
(( ${#EXTRA_STUDENT_SEEDS_ARRAY[@]} == 2 )) \
    || fail "this check requires exactly two EXTRA_STUDENT_SEEDS"

resolve_arm_paths() {
    local arm="$1" recovery_seed="$2"
    local seed_syn="$SYN_PARENT/seed${recovery_seed}"
    local seed_fkd="$FKD_PARENT/seed${recovery_seed}"
    case "$arm" in
        baseline)
            SYN_PATH="$seed_syn/baseline_coarse20_ipc25"
            FKD_PATH="$seed_fkd/baseline_bs16_ipc25"
            ;;
        fine_coarse_target)
            SYN_PATH="$seed_syn/fine100_coarse_target_ipc25"
            FKD_PATH="$seed_fkd/coarse_target_bs16_ipc25"
            ;;
        random_coarse_target)
            SYN_PATH="$seed_syn/random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_ipc25"
            FKD_PATH="$seed_fkd/random_coarse_target_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25"
            ;;
        random_pseudo_target)
            SYN_PATH="$seed_syn/random_merged_coarse20_pseed${RANDOM_PARTITION_SEED}_ipc25"
            FKD_PATH="$seed_fkd/random_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25"
            ;;
        oracle_fine_target)
            SYN_PATH="$seed_syn/oracle_merged_coarse20_ipc25"
            FKD_PATH="$seed_fkd/oracle_bs16_ipc25"
            ;;
        *) fail "unknown arm: $arm" ;;
    esac
}

validate_one() {
    local gpu="$1" arm="$2" recovery_seed="$3" student_seed="$4"
    resolve_arm_paths "$arm" "$recovery_seed"
    local per_class="$PER_CLASS/${arm}_rseed${recovery_seed}_sseed${student_seed}.json"
    local log="$LOGS/${arm}_rseed${recovery_seed}_sseed${student_seed}.log"
    [[ -f "$per_class" ]] && return
    [[ -d "$SYN_PATH" ]] || fail "missing synthetic set: $SYN_PATH"
    [[ -d "$FKD_PATH" ]] || fail "missing FKD set: $FKD_PATH"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 25 \
        --exp-name "class_in_class_${arm}_rseed${recovery_seed}_sseed${student_seed}" \
        --original-data-path "$SYN_PATH" --fkd-path "$FKD_PATH" --output-dir "$OUTPUT" \
        --batch-size 16 --epochs 300 --dataset-name cifar20 \
        --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature 20 --fkd_seed 42 --adamw-weight-decay 0.01 \
        --train-seed "$student_seed" --persistent-workers --val-dir "$COARSE_TEST" \
        --disable-wandb --per-class-output "$per_class" > "$log" 2>&1
}

ARMS=(baseline fine_coarse_target random_coarse_target random_pseudo_target oracle_fine_target)
echo "===== Extra post-eval seeds only ====="
echo "Recovery seeds: ${RECOVERY_SEEDS_ARRAY[*]}"
echo "New student seeds: ${EXTRA_STUDENT_SEEDS_ARRAY[*]}"
echo "Arms: ${ARMS[*]}"
echo "New jobs: $((${#RECOVERY_SEEDS_ARRAY[@]} * ${#EXTRA_STUDENT_SEEDS_ARRAY[@]} * ${#ARMS[@]}))"

pids=()
for student_seed in "${EXTRA_STUDENT_SEEDS_ARRAY[@]}"; do
    for recovery_seed in "${RECOVERY_SEEDS_ARRAY[@]}"; do
        for arm in "${ARMS[@]}"; do
            gpu="$GPU0"; (( ${#pids[@]} == 1 )) && gpu="$GPU1"
            validate_one "$gpu" "$arm" "$recovery_seed" "$student_seed" & pids+=("$!")
            if (( ${#pids[@]} == 2 )); then
                wait_jobs "${pids[@]}" || fail "a post-eval job failed; inspect $LOGS"
                pids=()
            fi
        done
    done
done
if (( ${#pids[@]} > 0 )); then
    wait_jobs "${pids[@]}" || fail "a post-eval job failed; inspect $LOGS"
fi

python "$ROOT/class_in_class/summarize_post_eval_seeds.py" \
    --per-class-dir "$PER_CLASS" \
    --recovery-seeds "${RECOVERY_SEEDS_ARRAY[@]}" \
    --student-seeds "${ALL_STUDENT_SEEDS_ARRAY[@]}" \
    --legacy-student-seed 42 --random-partition-seed "$RANDOM_PARTITION_SEED" \
    --output-dir "$ANALYSIS" > "$LOGS/post_eval_3x3_summary.log" 2>&1

echo "Complete: $ANALYSIS/post_eval_crossed_summary.json"
