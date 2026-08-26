#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=44 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_entropy_matched/tseed44}" bash "$ROOT/run_imagenette_entropy_match_teacher_seed_2gpu.sh"
