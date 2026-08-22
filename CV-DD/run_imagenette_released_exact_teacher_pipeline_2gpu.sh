#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

RUN_NAME="${RUN_NAME:-run1}"
TEACHER_ROOT="${TEACHER_ROOT:-$Main_Data_Path/class_in_class/imagenette_released_exact_teacher/$RUN_NAME}"
TEACHER="$TEACHER_ROOT/ResNet18.pth"
ASSET_ROOT="${ASSET_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_released_exact_teacher_pipeline/$RUN_NAME}"
LOGS="${LOGS:-$ROOT/logs/imagenette_released_exact_teacher_pipeline/$RUN_NAME}"

[[ -f "$TEACHER" ]] || {
    echo "missing released-exact Teacher: $TEACHER" >&2
    exit 1
}

C_VALUES=1 \
RECOVERY_SEEDS="${RECOVERY_SEEDS:-42}" \
STUDENT_SEEDS="${STUDENT_SEEDS:-42 43 44}" \
ASSET_ROOT="$ASSET_ROOT" \
EXP_ROOT="$EXP_ROOT" \
LOGS="$LOGS" \
C1_TEACHER_OVERRIDE="$TEACHER" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" WORKERS="${WORKERS:-8}" \
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}" \
bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh"
