#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROWS="real c1" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_labeler_factorial_c100/node_a_real_c1}" \
bash "$ROOT/run_imagenette_labeler_factorial_group_2gpu.sh"
