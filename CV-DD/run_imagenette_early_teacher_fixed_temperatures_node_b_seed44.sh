#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=44 bash "$ROOT/run_imagenette_early_teacher_fixed_temperatures_seed_2gpu.sh"
