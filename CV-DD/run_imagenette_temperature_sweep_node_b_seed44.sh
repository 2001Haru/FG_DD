#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=44 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_temperature_sweep_ipc10/tseed44}" \
    bash "$ROOT/run_imagenette_temperature_sweep_teacher_seed_2gpu.sh"
