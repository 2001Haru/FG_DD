#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_SEED=43 bash "$ROOT/run_imagenette_softlabel_variance_ratio_seed_2gpu.sh"
