#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=45 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_real_best_fp16/tseed45}" \
bash "$ROOT/run_imagenette_real_best_fp16_teacher_seed_2gpu.sh"
