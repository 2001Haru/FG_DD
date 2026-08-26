#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
python -u "$ROOT/class_in_class/summarize_imagenette_ipc_main_table.py" \
    --random-root "$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
    --factorial-root "$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100" \
    --ipc-root "$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table" \
    --teacher-seeds 43 44 --recovery-seeds 41 42 43 --student-seeds 42 43 44 \
    --output "$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table/analysis/ipc1_10_50_main_table.json"
