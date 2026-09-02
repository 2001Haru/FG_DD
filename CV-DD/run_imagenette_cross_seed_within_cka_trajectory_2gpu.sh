#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
TEMPERATURE="${TEMPERATURE:-20}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cross_seed_within_cka_trajectory"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cross_seed_within_cka_trajectory}"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

collect_family(){
    local c="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_cross_seed_within_cka_trajectory.py" collect \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
        --C "$c" --temperature "$TEMPERATURE" --output "$EXP_ROOT/c${c}.json" \
        --device cuda:0 --batch-size 256 --workers "$WORKERS" \
        > "$LOG_ROOT/c${c}.log" 2>&1
}

echo "[1/2] Cross-seed CKA trajectory: C1 on GPU $GPU0, C100 on GPU $GPU1"
collect_family 1 "$GPU0" & p0=$!
collect_family 100 "$GPU1" & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
(( status == 0 )) || { echo "cross-seed CKA collection failed" >&2; exit 1; }

echo "[2/2] Merge C1/C100 curves"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_cross_seed_within_cka_trajectory.py" merge \
    --c1 "$EXP_ROOT/c1.json" --c100 "$EXP_ROOT/c100.json" \
    --output-json "$EXP_ROOT/cross_seed_within_cka_trajectory.json" \
    --output-csv "$EXP_ROOT/cross_seed_within_cka_trajectory.csv" \
    > "$LOG_ROOT/merge.log" 2>&1

echo "Complete: $EXP_ROOT/cross_seed_within_cka_trajectory.csv"
