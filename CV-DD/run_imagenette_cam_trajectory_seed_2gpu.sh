#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cam_trajectories/tseed${TEACHER_SEED}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cam_trajectories/tseed${TEACHER_SEED}}"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"
fail(){ echo "CAM trajectory failed: $*" >&2; exit 1; }
wait_pair(){ local status=0; wait "$1" || status=1; wait "$2" || status=1; return "$status"; }

audit_one(){
    local c="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_cam_trajectory.py" \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
        --teacher-seed "$TEACHER_SEED" --C "$c" \
        --output-dir "$EXP_ROOT/c${c}" --batch-size 16 --workers "$WORKERS" \
        --temperature 20 --top-subheads 5 > "$LOG_ROOT/audit_c${c}.log" 2>&1
}

echo "[1/2] Numeric CAM trajectory: C1 on GPU$GPU0, C100 on GPU$GPU1"
audit_one 1 "$GPU0" & p1=$!
audit_one 100 "$GPU1" & p2=$!
wait_pair "$p1" "$p2" || fail numeric_audit
[[ -f "$EXP_ROOT/c1/summary.json" && -f "$EXP_ROOT/c100/summary.json" ]] \
    || fail missing_summary

echo "[2/2] Paired money-figure panels at epochs16/64/100/300"
if [[ "${SKIP_VISUALIZATION:-0}" == 1 ]]; then
    echo "Skipping unchanged visualization panels"
    echo "CAM trajectory complete: Teacher seed=$TEACHER_SEED"
    exit 0
fi
CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/visualize_imagenette_paired_cam_checkpoints.py" \
    --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
    --teacher-seed "$TEACHER_SEED" --output-dir "$EXP_ROOT/figures" \
    --epochs 16 64 100 300 --temperature 20 \
    > "$LOG_ROOT/visualization.log" 2>&1 || fail visualization
echo "CAM trajectory complete: Teacher seed=$TEACHER_SEED"
