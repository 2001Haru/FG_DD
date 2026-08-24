#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
RSEEDS="${RECOVERY_SEEDS:-41 42 43}"; SSEEDS="${STUDENT_SEEDS:-42 43 44}"
fail(){ echo "High-C node A failed: $*" >&2; exit 1; }
wait_pair(){ local a="$1" b="$2" status=0; wait "$a" || status=1; wait "$b" || status=1; return "$status"; }

for teacher_seed in 43 44; do
    for c in 100 200; do
        model="$MASTER_ROOT/tseed${teacher_seed}/models/random_c${c}_pseed42_tseed${teacher_seed}"
        [[ -f "$model/ResNet18.pth" && -f "$model/.training_complete.json" ]] \
            || fail "missing completed C=$c Teacher seed=$teacher_seed: $model"
    done
done

full_stage(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    TEACHER_SEED="$teacher_seed" C_VALUES="100 200" \
    RECOVERY_SEEDS="$RSEEDS" STUDENT_SEEDS="$SSEEDS" \
    RECOVERY_ITERATIONS=4000 RECOVERY_LR=0.1 RECOVERY_R_BN=0.01 \
    VIEW_SEED=42 TEMPERATURE=20 REAL_ROOT="$val_dir/imagenet-nette" \
    VAL_DIR="$val_dir/imagenet-nette/test" EXP_ROOT="$teacher_root" \
    LOGS="$MASTER_LOGS/tseed${teacher_seed}/full_c100_c200" \
    GPU0="$gpu" GPU1="$gpu" PARALLEL_JOBS=1 WORKERS="$WORKERS" \
    bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh"
}

echo "Node A: seed43 C100/C200 on GPU$GPU0; seed44 C100/C200 on GPU$GPU1"
full_stage 43 "$GPU0" & pid43=$!
full_stage 44 "$GPU1" & pid44=$!
wait_pair "$pid43" "$pid44" || fail "parallel C100/C200 pipeline"

read -r -a RSEED_ARRAY <<< "$RSEEDS"; read -r -a SSED_ARRAY <<< "$SSEEDS"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds 43 44 \
    --recovery-seeds "${RSEED_ARRAY[@]}" --student-seeds "${SSED_ARRAY[@]}" \
    --c-values 1 2 5 10 20 50 100 200 \
    --output "$MASTER_ROOT/analysis/summary_all_c_to200.json" \
    > "$MASTER_LOGS/summary_all_c_to200.log" 2>&1
echo "Complete: $MASTER_ROOT/analysis/summary_all_c_to200.json"
