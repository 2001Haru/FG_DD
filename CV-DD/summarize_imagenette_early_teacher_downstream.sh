#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
python -u "$ROOT/class_in_class/summarize_imagenette_early_teacher_downstream.py" \
    --experiment-root "$BASE/imagenette_early_teacher_downstream" \
    --existing-root "$BASE/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
    --factorial-root "$BASE/imagenette_labeler_factorial_c100" \
    --sweep-root "$BASE/imagenette_temperature_sweep_ipc10" \
    --output "$BASE/imagenette_early_teacher_downstream/analysis/early_teacher_downstream.json"
