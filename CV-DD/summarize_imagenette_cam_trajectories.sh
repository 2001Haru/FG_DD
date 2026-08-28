#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
python -u "$ROOT/class_in_class/summarize_imagenette_cam_trajectories.py" \
    --cam-root "$BASE/imagenette_cam_trajectories" \
    --downstream-summary "$BASE/imagenette_early_teacher_downstream/analysis/early_teacher_downstream.json" \
    --output "$BASE/imagenette_cam_trajectories/analysis/cam_trajectory_summary.json"
