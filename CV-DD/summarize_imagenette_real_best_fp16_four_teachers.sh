#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
SWEEP="$Main_Data_Path/class_in_class/imagenette_temperature_sweep_ipc10"
python -u "$ROOT/class_in_class/summarize_imagenette_real_best_fp16_four_teachers.py" \
    --sweep-root "$SWEEP" \
    --output "$SWEEP/analysis/real_best_fp16_four_teachers.json"
