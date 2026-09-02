#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
COLLECTED="$BASE/imagenette_accuracy_matched_channel_diagnostic/collected"
OUTPUT="$BASE/imagenette_utility_logit_R_analysis"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_utility_logit_R_analysis}"
mkdir -p "$OUTPUT" "$LOG_ROOT"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/analyze_imagenette_utility_logit_r_curve.py" \
    --c1-collected "$COLLECTED/c1.pt" --c100-collected "$COLLECTED/c100.pt" \
    --downstream-root "$BASE/imagenette_early_teacher_downstream" \
    --factorial-root "$BASE/imagenette_labeler_factorial_c100" \
    --random-root "$BASE/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
    --output-dir "$OUTPUT" > "$LOG_ROOT/analyze.log" 2>&1

echo "Complete: $OUTPUT/utility_logit_R_analysis.json"
