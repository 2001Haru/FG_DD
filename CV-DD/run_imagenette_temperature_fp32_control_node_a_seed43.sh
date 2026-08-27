#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=43 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_temperature_sweep_ipc10_fp32/tseed43}" \
    bash "$ROOT/run_imagenette_temperature_fp32_control_teacher_seed_2gpu.sh"
