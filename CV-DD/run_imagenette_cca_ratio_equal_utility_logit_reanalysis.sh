#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
PHASE_ROOT="$Main_Data_Path/class_in_class/imagenette_lr_phase_matched_channel_diagnostic"
SOURCE_ROOT="$Main_Data_Path/class_in_class/imagenette_best_teacher_channel_diagnostic"
REAL_ROOT="$Main_Data_Path/class_in_class/imagenette_real_best_teacher_channel_diagnostic"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cca_ratio_equal_utility_logit"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cca_ratio_equal_utility_logit}"
mkdir -p "$EXP_ROOT/source_average" "$EXP_ROOT/random_real" "$EXP_ROOT/analysis" "$LOG_ROOT"

reanalyze(){
    local input_root="$1" output="$2" log="$3"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" analyze \
        --input-dir "$input_root/collected" --output "$output" > "$log" 2>&1
}

echo "[1/3] Logit reanalysis of source-average equal-utility pair"
reanalyze "$SOURCE_ROOT" "$EXP_ROOT/source_average/channel_diagnostic.json" \
    "$LOG_ROOT/source_average.log"
echo "[2/3] Logit reanalysis of Random-Real equal-utility pair"
reanalyze "$REAL_ROOT" "$EXP_ROOT/random_real/channel_diagnostic.json" \
    "$LOG_ROOT/random_real.log"
echo "[3/3] CCA component ratios, monotonicity, and static figure"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/analyze_imagenette_cca_ratio_equal_utility.py" \
    --phase-e100 "$PHASE_ROOT/e100/analysis/channel_diagnostic.json" \
    --phase-e300 "$PHASE_ROOT/e300/analysis/channel_diagnostic.json" \
    --equal-source-average "$EXP_ROOT/source_average/channel_diagnostic.json" \
    --equal-random-real "$EXP_ROOT/random_real/channel_diagnostic.json" \
    --output-dir "$EXP_ROOT/analysis" > "$LOG_ROOT/analyze.log" 2>&1

echo "Complete: $EXP_ROOT/analysis/cca_ratio_equal_utility_summary.json"
