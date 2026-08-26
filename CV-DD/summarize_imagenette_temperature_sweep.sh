#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
python -u "$ROOT/class_in_class/summarize_imagenette_temperature_sweep.py" \
    --random-root "$BASE/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
    --factorial-root "$BASE/imagenette_labeler_factorial_c100" \
    --match-root "$BASE/imagenette_entropy_matched" \
    --sweep-root "$BASE/imagenette_temperature_sweep_ipc10" \
    --output "$BASE/imagenette_temperature_sweep_ipc10/analysis/temperature_sweep_results.json"
