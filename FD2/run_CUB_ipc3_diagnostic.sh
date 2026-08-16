#!/usr/bin/env bash
set -euo pipefail

FD2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FD2_DIR/config.sh"

DATASET_NAME="CUB_imsize224"
IPC=3
EPOCHS="${EPOCHS:-400}"
GPU_RELABEL="${GPU_RELABEL:-0}"
GPU_RESNET18="${GPU_RESNET18:-0}"
GPU_RESNET50="${GPU_RESNET50:-1}"

SYN_ROOT="${Main_Data_Path}/generated_data/syn_data/SRe2Lplus_FD2_${DATASET_NAME}_09FC05_01SC4"
SOURCE_DIR="${SYN_ROOT}/rec_res18_ipc5"
TARGET_DIR="${SYN_ROOT}/rec_res18_ipc${IPC}"
FKD_DIR="${Main_Data_Path}/generated_data/new_labels/SRe2Lplus_FD2_${DATASET_NAME}_09FC05_01SC4/rec_res18_ipc${IPC}_rel_res18_bs20_ipc${IPC}"
RELABEL_SCRIPT="$FD2_DIR/relabel/scripts/CUB_imsize224_experiment/SRe2Lplus_FD2_rec_res18_ipc3_rel_res18_bs20.sh"
VALIDATE_SCRIPT="$FD2_DIR/validate/scripts/CUB_imsize224_experiments/SRe2Lplus_FD2_ipc5_validate_2gpu.sh"

python "$FD2_DIR/get_small_ipc_from_big_ipc.py" \
    --source-dir "$SOURCE_DIR" \
    --target-dir "$TARGET_DIR" \
    --target-ipc "$IPC"

epoch_count=0
batch_count=0
if [[ -d "$FKD_DIR" ]]; then
    epoch_count="$(find "$FKD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | wc -l)"
    batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
fi

expected_batches=$((EPOCHS * 30))
if (( epoch_count >= EPOCHS && batch_count >= expected_batches )); then
    echo "IPC3 FKD labels already complete: $epoch_count epochs, $batch_count batches"
else
    echo "Generating IPC3 FKD labels on GPU $GPU_RELABEL"
    GPU_ID="$GPU_RELABEL" bash "$RELABEL_SCRIPT"

    epoch_count="$(find "$FKD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | wc -l)"
    batch_count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    if (( epoch_count < EPOCHS || batch_count < expected_batches )); then
        echo "IPC3 relabel incomplete: $epoch_count epochs, $batch_count/$expected_batches batches" >&2
        exit 1
    fi
fi

echo "Starting IPC3 validation"
IPC="$IPC" \
EPOCHS="$EPOCHS" \
GPU_RESNET18="$GPU_RESNET18" \
GPU_RESNET50="$GPU_RESNET50" \
bash "$VALIDATE_SCRIPT"
