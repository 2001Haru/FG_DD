#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_softlabel_variance_ratio/tseed${TEACHER_SEED}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_softlabel_variance_ratio/tseed${TEACHER_SEED}}"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"
run_one(){
    local c="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_softlabel_variance_ratio.py" \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
        --teacher-seed "$TEACHER_SEED" --C "$c" --output-dir "$EXP_ROOT/c${c}" \
        --batch-size 256 --workers "$WORKERS" > "$LOG_ROOT/c${c}.log" 2>&1
}
run_one 1 "$GPU0" & p1=$!
run_one 100 "$GPU1" & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1
(( status == 0 )) || { echo "variance ratio audit failed" >&2; exit 1; }
echo "Soft-label variance ratio complete: Teacher seed=$TEACHER_SEED"
