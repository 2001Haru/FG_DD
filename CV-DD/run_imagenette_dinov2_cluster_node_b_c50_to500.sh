#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_VALUES="50 100 200 500" \
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42/node_b_c50_to500}" \
bash "$ROOT/run_imagenette_dinov2_cluster_full_group_2gpu.sh"
