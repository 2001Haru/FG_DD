#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_VALUES="1 2 5 10 20" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_hard_label_eval/group_c1_to20}" \
bash "$ROOT/run_imagenette_hard_label_eval_group.sh"
