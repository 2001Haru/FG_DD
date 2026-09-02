#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_lr_phase_matched_channel_diagnostic"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_lr_phase_matched_channel_diagnostic}"
mkdir -p "$LOG_ROOT"

for phase in e100 e300; do
    phase_root="$EXP_ROOT/$phase"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_best_teacher_channels.py" analyze \
        --input-dir "$phase_root/collected" \
        --output "$phase_root/analysis/channel_diagnostic.json" \
        > "$LOG_ROOT/reanalyze_logit_${phase}.log" 2>&1
done

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/summarize_imagenette_lr_phase_channel_diagnostics.py" \
    --e100 "$EXP_ROOT/e100/analysis/channel_diagnostic.json" \
    --e300 "$EXP_ROOT/e300/analysis/channel_diagnostic.json" \
    --output-json "$EXP_ROOT/lr_phase_matched_channel_diagnostic.json" \
    --output-csv "$EXP_ROOT/lr_phase_matched_pair_summary.csv" \
    > "$LOG_ROOT/remerge_logit.log" 2>&1

echo "Logit-space reanalysis complete: $EXP_ROOT/lr_phase_matched_channel_diagnostic.json"
