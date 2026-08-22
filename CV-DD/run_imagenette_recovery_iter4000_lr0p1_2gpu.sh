#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

RECOVERY_ITERATIONS=4000 \
RECOVERY_LR=0.1 \
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_recovery_iter4000_lr0p1}" \
LOGS="${LOGS:-$ROOT/logs/imagenette_recovery_iter4000_lr0p1}" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" WORKERS="${WORKERS:-8}" \
RECOVERY_SEED="${RECOVERY_SEED:-42}" \
STUDENT_SEEDS="${STUDENT_SEEDS:-42 43 44}" \
VIEW_SEED="${VIEW_SEED:-42}" \
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}" \
bash "$ROOT/run_imagenette_recovery_iter2000_2gpu.sh"
