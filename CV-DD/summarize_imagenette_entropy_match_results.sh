#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";source "$ROOT/config.sh";C1_MATCH_T="${C1_MATCH_T:-43.4}";R100_MATCH_T="${R100_MATCH_T:-9.2}"
python -u "$ROOT/class_in_class/summarize_imagenette_entropy_match_results.py" --random-root "$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44" --factorial-root "$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100" --ipc-root "$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table" --match-root "$Main_Data_Path/class_in_class/imagenette_entropy_matched" --c1-temperature "$C1_MATCH_T" --r100-temperature "$R100_MATCH_T" --output "$Main_Data_Path/class_in_class/imagenette_entropy_matched/analysis/entropy_matched_results.json"
