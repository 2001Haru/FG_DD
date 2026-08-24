#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_VALUES="2 5 10 20" \
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42/node_a_c2_to20}" \
bash "$ROOT/run_imagenette_dinov2_cluster_full_group_2gpu.sh"
