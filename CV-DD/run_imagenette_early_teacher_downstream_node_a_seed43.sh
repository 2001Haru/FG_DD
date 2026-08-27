#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=43 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_downstream/tseed43}" \
bash "$ROOT/run_imagenette_early_teacher_downstream_seed_2gpu.sh"
