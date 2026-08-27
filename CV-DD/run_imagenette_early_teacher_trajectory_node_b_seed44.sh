#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=44 LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_trajectories/tseed44}" \
bash "$ROOT/run_imagenette_early_teacher_trajectory_seed_2gpu.sh"
