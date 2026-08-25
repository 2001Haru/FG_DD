#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROWS="random100 cluster100" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_labeler_factorial_c100/node_b_random_cluster}" \
bash "$ROOT/run_imagenette_labeler_factorial_group_2gpu.sh"
