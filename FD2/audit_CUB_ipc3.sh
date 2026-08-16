#!/usr/bin/env bash
set -euo pipefail

FD2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$FD2_DIR/config.sh"

SYN_DIR="${Main_Data_Path}/generated_data/syn_data/SRe2Lplus_FD2_CUB_imsize224_09FC05_01SC4/rec_res18_ipc3"
FKD_DIR="${Main_Data_Path}/generated_data/new_labels/SRe2Lplus_FD2_CUB_imsize224_09FC05_01SC4/rec_res18_ipc3_rel_res18_bs20_ipc3"
MODEL_DIR="${Main_Data_Path}/pretrained_models/CUB_imsize224"

cd "$FD2_DIR"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python -u audit_fkd_replay.py \
    --syn-data-path "$SYN_DIR" \
    --fkd-path "$FKD_DIR" \
    --model-pool-dir "$MODEL_DIR" \
    --batch-size 20 \
    --epochs 0 1 50 399
