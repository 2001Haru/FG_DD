#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
TEACHER_SEEDS_TEXT="${TEACHER_SEEDS:-43 44}"; read -r -a TEACHER_SEEDS_ARRAY <<< "$TEACHER_SEEDS_TEXT"
RECOVERY_SEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RECOVERY_SEEDS_ARRAY <<< "$RECOVERY_SEEDS_TEXT"
STUDENT_SEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a STUDENT_SEEDS_ARRAY <<< "$STUDENT_SEEDS_TEXT"
C_VALUES_TEXT="${C_VALUES:-1 2 5 10}"; read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
readonly FORMAL_RECOVERY_ITERATIONS=4000
readonly FORMAL_RECOVERY_LR=0.1
readonly FORMAL_RECOVERY_R_BN=0.01

MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
mkdir -p "$MASTER_ROOT/analysis" "$MASTER_LOGS"

fail(){ echo "ImageNette multi-Teacher CiC-T experiment failed: $*" >&2; exit 1; }

[[ "${TEACHER_SEEDS_ARRAY[*]}" == "43 44" ]] \
    || fail "formal protocol requires exactly Teacher seeds 43 44"
[[ "${C_VALUES_ARRAY[*]}" == "1 2 5 10" ]] \
    || fail "formal protocol requires C=1 2 5 10"

echo "[A/3] Train and audit 8 Teachers: tseeds=${TEACHER_SEEDS_ARRAY[*]}, C=${C_VALUES_ARRAY[*]}"
for teacher_seed in "${TEACHER_SEEDS_ARRAY[@]}"; do
    teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    teacher_logs="$MASTER_LOGS/tseed${teacher_seed}/teachers"
    TEACHER_SEED="$teacher_seed" C_VALUES="${C_VALUES_ARRAY[*]}" \
    SOURCE_ROOT="$val_dir/imagenet-nette" SOURCE_VALIDATION_SPLIT=test \
    EXP_ROOT="$teacher_root" LOGS="$teacher_logs" \
    GPU0="$GPU0" GPU1="$GPU1" WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_teachers_2gpu.sh" \
        || fail "Teacher stage failed for seed=$teacher_seed"
done

echo "[B/3] Run 4 arms x 2 Teacher seeds x 3 recovery x 3 student"
for teacher_seed in "${TEACHER_SEEDS_ARRAY[@]}"; do
    teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    full_logs="$MASTER_LOGS/tseed${teacher_seed}/full"
    TEACHER_SEED="$teacher_seed" C_VALUES="${C_VALUES_ARRAY[*]}" \
    RECOVERY_SEEDS="${RECOVERY_SEEDS_ARRAY[*]}" \
    STUDENT_SEEDS="${STUDENT_SEEDS_ARRAY[*]}" \
    RECOVERY_ITERATIONS="$FORMAL_RECOVERY_ITERATIONS" \
    RECOVERY_LR="$FORMAL_RECOVERY_LR" \
    RECOVERY_R_BN="$FORMAL_RECOVERY_R_BN" VIEW_SEED=42 TEMPERATURE=20 \
    REAL_ROOT="$val_dir/imagenet-nette" VAL_DIR="$val_dir/imagenet-nette/test" \
    EXP_ROOT="$teacher_root" LOGS="$full_logs" \
    GPU0="$GPU0" GPU1="$GPU1" WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh" \
        || fail "Full pipeline failed for Teacher seed=$teacher_seed"
done

echo "[C/3] Combine the 72 post-eval cells"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds "${TEACHER_SEEDS_ARRAY[@]}" \
    --recovery-seeds "${RECOVERY_SEEDS_ARRAY[@]}" \
    --student-seeds "${STUDENT_SEEDS_ARRAY[@]}" \
    --c-values "${C_VALUES_ARRAY[@]}" \
    --output "$MASTER_ROOT/analysis/summary.json" \
    > "$MASTER_LOGS/summary.log" 2>&1

echo "Complete: $MASTER_ROOT/analysis/summary.json"
