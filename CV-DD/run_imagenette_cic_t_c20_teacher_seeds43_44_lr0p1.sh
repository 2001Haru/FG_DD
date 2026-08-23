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
readonly FORMAL_RECOVERY_ITERATIONS=4000
readonly FORMAL_RECOVERY_LR=0.1
readonly FORMAL_RECOVERY_R_BN=0.01

MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
mkdir -p "$MASTER_ROOT/analysis" "$MASTER_LOGS"

fail(){ echo "ImageNette C20 addon failed: $*" >&2; exit 1; }
wait_pair(){
    local first="$1" second="$2" status=0
    wait "$first" || status=1
    wait "$second" || status=1
    return "$status"
}
[[ "${TEACHER_SEEDS_ARRAY[*]}" == "43 44" ]] \
    || fail "formal addon requires Teacher seeds 43 44"

echo "[A/3] Train and audit C20 Teachers for seeds=${TEACHER_SEEDS_ARRAY[*]}"
teacher_stage(){
    local teacher_seed="$1" gpu="$2"
    teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    teacher_logs="$MASTER_LOGS/tseed${teacher_seed}/teachers_c20"
    TEACHER_SEED="$teacher_seed" C_VALUES=20 \
    SOURCE_ROOT="$val_dir/imagenet-nette" SOURCE_VALIDATION_SPLIT=test \
    EXP_ROOT="$teacher_root" LOGS="$teacher_logs" \
    GPU0="$gpu" GPU1="$gpu" WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_teachers_2gpu.sh"
}
teacher_stage 43 "$GPU0" & teacher43_pid=$!
teacher_stage 44 "$GPU1" & teacher44_pid=$!
wait_pair "$teacher43_pid" "$teacher44_pid" || fail "parallel C20 Teacher stage"

echo "[B/3] Run C20 x 2 Teacher seeds x 3 recovery x 3 student"
full_stage(){
    local teacher_seed="$1" gpu="$2"
    teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    full_logs="$MASTER_LOGS/tseed${teacher_seed}/full_c20"
    TEACHER_SEED="$teacher_seed" C_VALUES=20 \
    RECOVERY_SEEDS="${RECOVERY_SEEDS_ARRAY[*]}" \
    STUDENT_SEEDS="${STUDENT_SEEDS_ARRAY[*]}" \
    RECOVERY_ITERATIONS="$FORMAL_RECOVERY_ITERATIONS" \
    RECOVERY_LR="$FORMAL_RECOVERY_LR" RECOVERY_R_BN="$FORMAL_RECOVERY_R_BN" \
    VIEW_SEED=42 TEMPERATURE=20 REAL_ROOT="$val_dir/imagenet-nette" \
    VAL_DIR="$val_dir/imagenet-nette/test" EXP_ROOT="$teacher_root" LOGS="$full_logs" \
    GPU0="$gpu" GPU1="$gpu" PARALLEL_JOBS=1 WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh"
}
full_stage 43 "$GPU0" & full43_pid=$!
full_stage 44 "$GPU1" & full44_pid=$!
wait_pair "$full43_pid" "$full44_pid" || fail "parallel C20 full pipeline"

echo "[C/3] Combine C=1/2/5/10/20 across both Teacher seeds"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds "${TEACHER_SEEDS_ARRAY[@]}" \
    --recovery-seeds "${RECOVERY_SEEDS_ARRAY[@]}" \
    --student-seeds "${STUDENT_SEEDS_ARRAY[@]}" \
    --c-values 1 2 5 10 20 \
    --output "$MASTER_ROOT/analysis/summary_with_c20.json" \
    > "$MASTER_LOGS/summary_with_c20.log" 2>&1

echo "Complete: $MASTER_ROOT/analysis/summary_with_c20.json"
