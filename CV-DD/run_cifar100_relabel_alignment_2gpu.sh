#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RECOVERY_SEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RECOVERY_SEEDS_ARRAY <<< "$RECOVERY_SEEDS_TEXT"
STUDENT_SEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a STUDENT_SEEDS_ARRAY <<< "$STUDENT_SEEDS_TEXT"
RANDOM_PARTITION_SEED="${RANDOM_PARTITION_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-20}"
RELABEL_SEED="${RELABEL_SEED:-42}"
FKD_VIEW_SEED="${FKD_VIEW_SEED:-42}"

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/cifar100_v2_bs100}"
SYN_PARENT="$EXP_ROOT/synthetic"
FINE_MODELS="$EXP_ROOT/models/fine100"
RANDOM_MODELS="$EXP_ROOT/models/random100_seed${RANDOM_PARTITION_SEED}"
FINE_MAPPING="$EXP_ROOT/data/hierarchy.json"
RANDOM_MAPPING="$EXP_ROOT/data/random100_seed${RANDOM_PARTITION_SEED}/hierarchy.json"
COARSE_TEST="$EXP_ROOT/data/coarse/test"; FINE_TEST="$EXP_ROOT/data/fine/test"
RANDOM_TEST="$EXP_ROOT/data/random100_seed${RANDOM_PARTITION_SEED}/test"
FKD_PARENT="$EXP_ROOT/relabel_alignment_fkd"
OUTPUT="$EXP_ROOT/relabel_alignment_post_eval"
PER_CLASS="$EXP_ROOT/relabel_alignment_per_class"
LOGS="$ROOT/logs/cifar100_class_in_class_v2_bs100/relabel_alignment"
mkdir -p "$FKD_PARENT" "$OUTPUT" "$PER_CLASS" "$LOGS"

fail() { echo "Relabel-alignment preflight failed: $*" >&2; exit 1; }
wait_jobs() {
    local status=0 pid
    for pid in "$@"; do wait "$pid" || status=$?; done
    return "$status"
}
for required in "$FINE_MODELS/ResNet18.pth" "$RANDOM_MODELS/ResNet18.pth" \
                "$FINE_MAPPING" "$RANDOM_MAPPING" "$COARSE_TEST" "$FINE_TEST" "$RANDOM_TEST"; do
    [[ -e "$required" ]] || fail "missing $required"
done

resolve_arm() {
    local arm="$1" recovery_seed="$2"
    local seed_syn="$SYN_PARENT/seed${recovery_seed}"
    case "$arm" in
        oracle_aligned)
            SYN_PATH="$seed_syn/oracle_merged_coarse20_ipc25"
            MODEL_DIR="$FINE_MODELS"; RELABEL_DATASET=cifar20; STUDENT_DATASET=cifar20
            TEACHER_CLASSES=100; TEACHER_MAPPING="$FINE_MAPPING"; NATIVE100=0; ARM_IPC=25
            ;;
        baseline_mismatched)
            SYN_PATH="$seed_syn/baseline_coarse20_ipc25"
            MODEL_DIR="$FINE_MODELS"; RELABEL_DATASET=cifar20; STUDENT_DATASET=cifar20
            TEACHER_CLASSES=100; TEACHER_MAPPING="$FINE_MAPPING"; NATIVE100=0; ARM_IPC=25
            ;;
        random_aligned)
            SYN_PATH="$seed_syn/random_merged_coarse20_pseed${RANDOM_PARTITION_SEED}_ipc25"
            MODEL_DIR="$RANDOM_MODELS"; RELABEL_DATASET=cifar20; STUDENT_DATASET=cifar20
            TEACHER_CLASSES=100; TEACHER_MAPPING="$RANDOM_MAPPING"; NATIVE100=0; ARM_IPC=25
            ;;
        oracle_100dim)
            SYN_PATH="$seed_syn/oracle_fine100_ipc5"
            MODEL_DIR="$FINE_MODELS"; RELABEL_DATASET=cifar100; STUDENT_DATASET=cifar100
            TEACHER_CLASSES=100; TEACHER_MAPPING=""; NATIVE100=1; ARM_IPC=5
            EVAL_MAPPING="$FINE_MAPPING"; EVAL_VAL_DIR="$FINE_TEST"
            ;;
        baseline_random_marg20)
            SYN_PATH="$seed_syn/baseline_coarse20_ipc25"
            MODEL_DIR="$RANDOM_MODELS"; RELABEL_DATASET=cifar20; STUDENT_DATASET=cifar20
            TEACHER_CLASSES=100; TEACHER_MAPPING="$RANDOM_MAPPING"; NATIVE100=0; ARM_IPC=25
            ;;
        baseline_random_100dim)
            SYN_PATH="$seed_syn/baseline_coarse20_ipc25"
            MODEL_DIR="$RANDOM_MODELS"; RELABEL_DATASET=cifar20; STUDENT_DATASET=cifar100
            TEACHER_CLASSES=100; TEACHER_MAPPING=""; NATIVE100=1; ARM_IPC=25
            EVAL_MAPPING="$RANDOM_MAPPING"; EVAL_VAL_DIR="$RANDOM_TEST"
            ;;
        *) fail "unknown arm: $arm" ;;
    esac
    FKD_BASE="$FKD_PARENT/seed${recovery_seed}/${arm}"
    FKD_PATH="${FKD_BASE}_bs16_ipc${ARM_IPC}"
}

relabel_one() {
    local gpu="$1" arm="$2" recovery_seed="$3"
    resolve_arm "$arm" "$recovery_seed"
    local count=0 log="$LOGS/relabel_${arm}_rseed${recovery_seed}.log"
    [[ -d "$FKD_PATH" ]] && count="$(find "$FKD_PATH" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 9600 )) && return
    (( count == 0 )) || fail "partial FKD directory: $FKD_PATH ($count/9600)"
    local extra_args=(--teacher-num-classes "$TEACHER_CLASSES")
    if [[ -n "$TEACHER_MAPPING" ]]; then
        extra_args+=(--teacher-mapping "$TEACHER_MAPPING"
                     --marginalize-temperature "$TEMPERATURE")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$SYN_PATH" \
        --fkd-path "$FKD_BASE" --model-pool-dir "$MODEL_DIR" \
        --teacher-model-name ResNet18 --gpu 0 --batch-size 16 --workers "$WORKERS" \
        --dataset-name "$RELABEL_DATASET" --epochs 300 --fkd-seed "$FKD_VIEW_SEED" \
        --seed "$RELABEL_SEED" --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix "${extra_args[@]}" > "$log" 2>&1
}

ARMS=(oracle_aligned baseline_mismatched random_aligned oracle_100dim \
      baseline_random_marg20 baseline_random_100dim)
echo "===== Relabel alignment matrix ====="
echo "Recovery seeds: ${RECOVERY_SEEDS_ARRAY[*]}"
echo "Student seeds: ${STUDENT_SEEDS_ARRAY[*]}"
echo "Temperature: $TEMPERATURE (100-way value not retuned)"
echo "Relabel/FKD view seeds: $RELABEL_SEED / $FKD_VIEW_SEED"

pids=()
for recovery_seed in "${RECOVERY_SEEDS_ARRAY[@]}"; do
    for arm in "${ARMS[@]}"; do
        gpu="$GPU0"; (( ${#pids[@]} == 1 )) && gpu="$GPU1"
        relabel_one "$gpu" "$arm" "$recovery_seed" & pids+=("$!")
        if (( ${#pids[@]} == 2 )); then
            wait_jobs "${pids[@]}" || fail "relabel job failed; inspect $LOGS"
            pids=()
        fi
    done
done
if (( ${#pids[@]} > 0 )); then wait_jobs "${pids[@]}" || fail "relabel job failed"; fi

validate_one() {
    local gpu="$1" arm="$2" recovery_seed="$3" student_seed="$4"
    resolve_arm "$arm" "$recovery_seed"
    local per_class="$PER_CLASS/${arm}_rseed${recovery_seed}_sseed${student_seed}.json"
    local log="$LOGS/validate_${arm}_rseed${recovery_seed}_sseed${student_seed}.log"
    [[ -f "$per_class" ]] && return
    local val_dir="$COARSE_TEST" eval_args=()
    if (( NATIVE100 == 1 )); then
        val_dir="$EVAL_VAL_DIR"
        eval_args+=(--eval-hierarchy-mapping "$EVAL_MAPPING"
                    --primary-eval-collapsed-coarse)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc "$ARM_IPC" \
        --exp-name "relabel_alignment_${arm}_rseed${recovery_seed}_sseed${student_seed}" \
        --original-data-path "$SYN_PATH" --fkd-path "$FKD_PATH" --output-dir "$OUTPUT" \
        --batch-size 16 --epochs 300 --dataset-name "$STUDENT_DATASET" \
        --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$FKD_VIEW_SEED" --adamw-weight-decay 0.01 \
        --train-seed "$student_seed" --persistent-workers --val-dir "$val_dir" \
        --disable-wandb --per-class-output "$per_class" "${eval_args[@]}" > "$log" 2>&1
}

pids=()
for student_seed in "${STUDENT_SEEDS_ARRAY[@]}"; do
    for recovery_seed in "${RECOVERY_SEEDS_ARRAY[@]}"; do
        for arm in "${ARMS[@]}"; do
            gpu="$GPU0"; (( ${#pids[@]} == 1 )) && gpu="$GPU1"
            validate_one "$gpu" "$arm" "$recovery_seed" "$student_seed" & pids+=("$!")
            if (( ${#pids[@]} == 2 )); then
                wait_jobs "${pids[@]}" || fail "post-eval job failed; inspect $LOGS"
                pids=()
            fi
        done
    done
done
if (( ${#pids[@]} > 0 )); then wait_jobs "${pids[@]}" || fail "post-eval job failed"; fi

python "$ROOT/class_in_class/summarize_relabel_alignment.py" \
    --alignment-per-class "$PER_CLASS" --reference-per-class "$EXP_ROOT/per_class" \
    --recovery-seeds "${RECOVERY_SEEDS_ARRAY[@]}" \
    --student-seeds "${STUDENT_SEEDS_ARRAY[@]}" \
    --random-partition-seed "$RANDOM_PARTITION_SEED" \
    --temperature "$TEMPERATURE" \
    --output-dir "$EXP_ROOT/analysis/relabel_alignment" \
    > "$LOGS/relabel_alignment_summary.log" 2>&1

echo "Complete: $EXP_ROOT/analysis/relabel_alignment/relabel_alignment_summary.json"
