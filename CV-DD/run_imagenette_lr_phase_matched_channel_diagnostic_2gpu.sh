#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
TEST_ROOT="$val_dir/imagenet-nette/test"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_lr_phase_matched_channel_diagnostic"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_lr_phase_matched_channel_diagnostic}"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

collect_one(){
    local phase="$1" epoch="$2" teacher_seed="$3" gpu="$4"
    local phase_root="$EXP_ROOT/$phase"
    mkdir -p "$phase_root/collected" "$phase_root/analysis"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" collect \
        --trajectory-root "$TRAJECTORY_ROOT" --test-root "$TEST_ROOT" \
        --teacher-seed "$teacher_seed" \
        --output "$phase_root/collected/tseed${teacher_seed}.pt" \
        --device cuda:0 --batch-size 256 --workers "$WORKERS" \
        --c1-training-epoch "$epoch" --c1-temperature 20 \
        --c100-training-epoch "$epoch" --c100-temperature 20 \
        --selection-label "LR-phase matched ${phase}: C1/C100 epoch ${epoch}, T20" \
        > "$LOG_ROOT/collect_${phase}_tseed${teacher_seed}.log" 2>&1
}

analyze_phase(){
    local phase="$1"
    local phase_root="$EXP_ROOT/$phase"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" analyze \
        --input-dir "$phase_root/collected" \
        --output "$phase_root/analysis/channel_diagnostic.json" \
        > "$LOG_ROOT/analyze_${phase}.log" 2>&1
}

run_phase(){
    local phase="$1" epoch="$2"
    echo "[${phase}] collect seed43/44 on two A100s"
    collect_one "$phase" "$epoch" 43 "$GPU0" & p0=$!
    collect_one "$phase" "$epoch" 44 "$GPU1" & p1=$!
    status=0
    wait "$p0" || status=1
    wait "$p1" || status=1
    (( status == 0 )) || { echo "collection failed: $phase" >&2; exit 1; }
    analyze_phase "$phase"
}

echo "[1/3] e100/e100 at T20"
run_phase e100 100
echo "[2/3] e300/e300 at T20"
run_phase e300 300
echo "[3/3] Merge phase-matched diagnostics"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/summarize_imagenette_lr_phase_channel_diagnostics.py" \
    --e100 "$EXP_ROOT/e100/analysis/channel_diagnostic.json" \
    --e300 "$EXP_ROOT/e300/analysis/channel_diagnostic.json" \
    --output-json "$EXP_ROOT/lr_phase_matched_channel_diagnostic.json" \
    --output-csv "$EXP_ROOT/lr_phase_matched_pair_summary.csv" \
    > "$LOG_ROOT/merge.log" 2>&1

echo "Complete: $EXP_ROOT/lr_phase_matched_channel_diagnostic.json"
