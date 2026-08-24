#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_VALUES="50 100 200 500 800" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_hard_label_eval/group_c50_to800}" \
bash "$ROOT/run_imagenette_hard_label_eval_group.sh"
