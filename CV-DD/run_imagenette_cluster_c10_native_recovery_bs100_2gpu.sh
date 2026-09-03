#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

export RECOVERY_BATCH_SIZE=100
export EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cluster_c10_native_recovery_bs100_c1_labeler"
export LOG_ROOT="$ROOT/logs/imagenette_cluster_c10_native_recovery_bs100_c1_labeler"
# Native patches depend on the partition and Teacher, not on recovery batch
# composition. Reuse the audited BS10 experiment's completed patch assets only.
export PATCH_ASSET_ROOT="$Main_Data_Path/class_in_class/imagenette_cluster_c10_native_recovery_c1_labeler"

exec bash "$ROOT/run_imagenette_cluster_c10_native_recovery_c1_labeler_2gpu.sh"
