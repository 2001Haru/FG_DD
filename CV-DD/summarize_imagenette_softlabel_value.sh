#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
python -u "$ROOT/class_in_class/analyze_imagenette_softlabel_value.py" \
    --ratio-root "$BASE/imagenette_softlabel_variance_ratio" \
    --downstream-root "$BASE/imagenette_early_teacher_downstream" \
    --factorial-root "$BASE/imagenette_labeler_factorial_c100" \
    --random-root "$BASE/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
    --output-dir "$BASE/imagenette_softlabel_variance_ratio/analysis"
