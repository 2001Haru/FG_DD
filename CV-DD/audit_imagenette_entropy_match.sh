#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";source "$ROOT/config.sh"
C1_MATCH_T="${C1_MATCH_T:-43.4}";R100_MATCH_T="${R100_MATCH_T:-9.2}"
python -u "$ROOT/class_in_class/audit_imagenette_entropy_match.py" --random-root "$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44" --factorial-root "$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100" --ipc-root "$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table" --match-root "$Main_Data_Path/class_in_class/imagenette_entropy_matched" --c1-temperature "$C1_MATCH_T" --r100-temperature "$R100_MATCH_T" --teacher-seeds 43 44 --recovery-seeds 41 42 43 --epoch-stride 10 --workers 8 --output "$Main_Data_Path/class_in_class/imagenette_entropy_matched/analysis/entropy_alignment.json"
