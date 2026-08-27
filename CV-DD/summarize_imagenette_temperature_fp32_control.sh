#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
python -u "$ROOT/class_in_class/summarize_imagenette_temperature_fp32_control.py" \
    --fp16-root "$BASE/imagenette_temperature_sweep_ipc10" \
    --fp32-root "$BASE/imagenette_temperature_sweep_ipc10_fp32" \
    --output "$BASE/imagenette_temperature_sweep_ipc10_fp32/analysis/fp32_control_results.json"
