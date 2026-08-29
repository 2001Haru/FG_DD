#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=43 \
EPOCHS_ONLY=4 \
MODES_OVERRIDE="ref t8 t46 t100 t200" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_epoch4/tseed43}" \
bash "$ROOT/run_imagenette_early_teacher_fixed_temperatures_seed_2gpu.sh"
