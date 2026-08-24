#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
RSEEDS="${RECOVERY_SEEDS:-41 42 43}"; SSEEDS="${STUDENT_SEEDS:-42 43 44}"
fail(){ echo "High-C node B failed: $*" >&2; exit 1; }
wait_pair(){ local a="$1" b="$2" status=0; wait "$a" || status=1; wait "$b" || status=1; return "$status"; }

teacher_stage(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    TEACHER_SEED="$teacher_seed" C_VALUES="500 800" \
    SOURCE_ROOT="$val_dir/imagenet-nette" SOURCE_VALIDATION_SPLIT=test \
    EXP_ROOT="$teacher_root" LOGS="$MASTER_LOGS/tseed${teacher_seed}/teachers_c500_c800" \
    GPU0="$gpu" GPU1="$gpu" PARALLEL_JOBS=1 WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_teachers_2gpu.sh"
}

echo "Node B Teacher stage: seed43 on GPU$GPU0; seed44 on GPU$GPU1"
teacher_stage 43 "$GPU0" & teacher43_pid=$!
teacher_stage 44 "$GPU1" & teacher44_pid=$!
wait_pair "$teacher43_pid" "$teacher44_pid" || fail "parallel C500/C800 Teacher stage"

full_stage(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    TEACHER_SEED="$teacher_seed" C_VALUES="500 800" \
    RECOVERY_SEEDS="$RSEEDS" STUDENT_SEEDS="$SSEEDS" \
    RECOVERY_ITERATIONS=4000 RECOVERY_LR=0.1 RECOVERY_R_BN=0.01 \
    VIEW_SEED=42 TEMPERATURE=20 REAL_ROOT="$val_dir/imagenet-nette" \
    VAL_DIR="$val_dir/imagenet-nette/test" EXP_ROOT="$teacher_root" \
    LOGS="$MASTER_LOGS/tseed${teacher_seed}/full_c500_c800" \
    GPU0="$gpu" GPU1="$gpu" PARALLEL_JOBS=1 WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh"
}

echo "Node B full stage: seed43 on GPU$GPU0; seed44 on GPU$GPU1"
full_stage 43 "$GPU0" & full43_pid=$!
full_stage 44 "$GPU1" & full44_pid=$!
wait_pair "$full43_pid" "$full44_pid" || fail "parallel C500/C800 pipeline"

read -r -a RSEED_ARRAY <<< "$RSEEDS"; read -r -a SSED_ARRAY <<< "$SSEEDS"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds 43 44 \
    --recovery-seeds "${RSEED_ARRAY[@]}" --student-seeds "${SSED_ARRAY[@]}" \
    --c-values 1 500 800 \
    --output "$MASTER_ROOT/analysis/summary_c500_c800.json" \
    > "$MASTER_LOGS/summary_c500_c800.log" 2>&1
echo "Complete: $MASTER_ROOT/analysis/summary_c500_c800.json"
