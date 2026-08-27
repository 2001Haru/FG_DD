#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
python -u "$ROOT/class_in_class/summarize_imagenette_early_teacher_trajectories.py" \
    --root "$TRAJECTORY_ROOT" \
    --output "$TRAJECTORY_ROOT/analysis/teacher_trajectories.json"
