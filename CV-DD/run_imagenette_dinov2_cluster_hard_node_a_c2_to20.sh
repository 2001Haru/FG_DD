#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
C_VALUES="2 5 10 20" \
MASTER_ROOT="$MASTER_ROOT" \
HARD_PROTOCOL="ImageNette IPC10 ResNet18; DINOv2 clustered CiC-T synthetic images; hard coarse10 labels; no FKD/relabel; 300 epochs BS10 AdamW LR5e-4 eta1" \
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42/hard_label/node_a_c2_to20}" \
bash "$ROOT/run_imagenette_hard_label_eval_group.sh"
