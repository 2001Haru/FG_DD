#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
RANDOM_ROOT="${RANDOM_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
CLUSTER_ROOT="${CLUSTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
MATRIX_ROOT="${MATRIX_ROOT:-$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100}"
RANK_WORKERS="${RANK_WORKERS:-8}"
python -u "$ROOT/class_in_class/summarize_imagenette_labeler_factorial.py" \
    --random-root "$RANDOM_ROOT" --cluster-root "$CLUSTER_ROOT" \
    --matrix-root "$MATRIX_ROOT" --teacher-seeds 43 44 \
    --recovery-seeds 41 42 43 --student-seeds 42 43 44 \
    --rank-epoch-stride 10 --rank-workers "$RANK_WORKERS" \
    --output "$MATRIX_ROOT/analysis/factorial_4x4_summary.json"
