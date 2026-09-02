#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_accuracy_matched_channel_diagnostic"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_accuracy_matched_channel_diagnostic}"
mkdir -p "$EXP_ROOT/collected" "$EXP_ROOT/analysis" "$LOG_ROOT"

collect_family(){
    local c="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_accuracy_matched_channels.py" collect \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" --C "$c" \
        --output "$EXP_ROOT/collected/c${c}.pt" --device cuda:0 \
        --batch-size 256 --workers "$WORKERS" > "$LOG_ROOT/collect_c${c}.log" 2>&1
}

echo "[1/2] Forward existing checkpoints: C1 on GPU $GPU0, C100 on GPU $GPU1"
collect_family 1 "$GPU0" & p0=$!
collect_family 100 "$GPU1" & p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1
(( status == 0 )) || { echo "collection failed" >&2; exit 1; }

echo "[2/2] Accuracy-matched cross-family analysis and within-family calibration"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_accuracy_matched_channels.py" analyze \
    --c1 "$EXP_ROOT/collected/c1.pt" --c100 "$EXP_ROOT/collected/c100.pt" \
    --output-json "$EXP_ROOT/analysis/accuracy_matched_channel_diagnostic.json" \
    --output-pairs-csv "$EXP_ROOT/analysis/accuracy_matched_pairs.csv" \
    --output-calibration-csv "$EXP_ROOT/analysis/within_family_calibration.csv" \
    --output-bins-csv "$EXP_ROOT/analysis/within_family_calibration_bins.csv" \
    > "$LOG_ROOT/analyze.log" 2>&1

echo "Complete: $EXP_ROOT/analysis/accuracy_matched_channel_diagnostic.json"
