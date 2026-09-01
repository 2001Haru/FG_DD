#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_best_teacher_channel_diagnostic"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_best_teacher_channel_diagnostic}"
mkdir -p "$EXP_ROOT/collected" "$EXP_ROOT/analysis" "$LOG_ROOT"

collect_one(){
    local teacher_seed="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" collect \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
        --teacher-seed "$teacher_seed" --output "$EXP_ROOT/collected/tseed${teacher_seed}.pt" \
        --device cuda:0 --batch-size 256 --workers "$WORKERS" \
        > "$LOG_ROOT/collect_tseed${teacher_seed}.log" 2>&1
}

echo "[1/2] Collect aligned soft-label matrices on two A100s"
collect_one 43 "$GPU0" & p0=$!
collect_one 44 "$GPU1" & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
(( status == 0 )) || { echo "soft-label collection failed" >&2; exit 1; }

echo "[2/2] Principal angles, CKA, CCA, class similarity, and rank agreement"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" analyze \
    --input-dir "$EXP_ROOT/collected" \
    --output "$EXP_ROOT/analysis/best_teacher_channel_diagnostic.json" \
    > "$LOG_ROOT/analyze.log" 2>&1

echo "Diagnostic complete: $EXP_ROOT/analysis/best_teacher_channel_diagnostic.json"
