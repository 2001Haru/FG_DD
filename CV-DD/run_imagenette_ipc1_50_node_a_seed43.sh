#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=43 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_ipc1_50_main_table/tseed43}" \
bash "$ROOT/run_imagenette_ipc1_50_teacher_seed_2gpu.sh"
