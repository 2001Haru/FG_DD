#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOVER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FD2_DIR="$(dirname "$RECOVER_DIR")"
source "$SCRIPT_DIR/constants.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ITERATIONS="${ITERATIONS:-10000}"
CLASS_NUM="${CLASS_NUM:-100}"
IPC_START="${IPC_START:-0}"
IPC_END="${IPC_END:-5}"
# The released scripts run IPC 4 separately, which disables SC for that IPC.
# Keep that behavior by default for baseline reproduction. Set to 0 to use
# IPCs 0--3 as SC references when recovering IPC 4 as well.
LEGACY_IPC4_NO_SC="${LEGACY_IPC4_NO_SC:-1}"
REC_NAME="rec_res18_ipc5"

SYN_DATA_ROOT="${Main_Data_Path}/generated_data/syn_data/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4"
SYN_DATA_DIR="${SYN_DATA_ROOT}/${REC_NAME}"
PATCH_DIR="${Main_Data_Path}/patches/${Dataset_Name}"
MODEL_POOL_DIR="${Main_Data_Path}/pretrained_models/${Dataset_Name}"
CHECKPOINT="${MODEL_POOL_DIR}/ResNet18_M8_5e-1cal.pth"
LOG_DIR="${SCRIPT_DIR}/logs/${REC_NAME}_2gpu"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$PATCH_DIR/2" ]]; then
    echo "Missing RDED patch directory: $PATCH_DIR/2" >&2
    exit 1
fi
patch_count="$(find "$PATCH_DIR/2" -type f -name '*.jpg' | wc -l)"
if (( patch_count != 1000 )); then
    echo "Incomplete RDED initialization: found $patch_count JPG files, expected 1000" >&2
    exit 1
fi
if (( IPC_START < 0 || IPC_END > 5 || IPC_START >= IPC_END )); then
    echo "Require 0 <= IPC_START < IPC_END <= 5" >&2
    exit 1
fi
if [[ "$LEGACY_IPC4_NO_SC" != "0" && "$LEGACY_IPC4_NO_SC" != "1" ]]; then
    echo "LEGACY_IPC4_NO_SC must be 0 or 1" >&2
    exit 1
fi
if (( IPC_START > 0 )); then
    existing_count=0
    if [[ -d "$SYN_DATA_DIR" ]]; then
        existing_count="$(find "$SYN_DATA_DIR" -type f -name '*.jpg' | wc -l)"
    fi
    required_count=$((200 * IPC_START))
    if (( existing_count < required_count )); then
        echo "Cannot start at IPC $IPC_START: found $existing_count prior images, need $required_count" >&2
        exit 1
    fi
fi

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

run_shard() {
    local gpu="$1"
    local class_start="$2"
    local class_end="$3"
    local ipc_id="$4"
    local sc_reference_start="$5"
    local log_file="$6"

    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    python -u "$RECOVER_DIR/recover_FD2.py" \
        --exp_name "$REC_NAME" \
        --apply_data_augmentation \
        --dataset_name "$Dataset_Name" \
        --class_num "$CLASS_NUM" \
        --class_start "$class_start" \
        --class_end "$class_end" \
        --subprocess_num 1 \
        --syn_data_path "$SYN_DATA_ROOT" \
        --patch_dir "$PATCH_DIR" \
        --model_pool_dir "$MODEL_POOL_DIR" \
        --pretrained_model_type offline \
        --model_choice ResNet18 \
        --M 8 \
        --cal_ratio 0.5 \
        --voter_type equal \
        --selected_size 1 \
        --lr 1e-3 \
        --iteration "$ITERATIONS" \
        --r_bn 1e-3 \
        --FC \
        --FC_ratio 0.9 \
        --IntraFC_ratio 0.5 \
        --SC \
        --SC_ratio 0.1 \
        --SC_loss_threshold 0.0 \
        --store_best_images \
        --ipc_start "$ipc_id" \
        --ipc_end "$((ipc_id + 1))" \
        --sc_reference_start "$sc_reference_start" \
        --initialisation_method Patches \
        --patch_diff 2 \
        --skip_completed > "$log_file" 2>&1
}

cd "$FD2_DIR"

for ((ipc_id = IPC_START; ipc_id < IPC_END; ipc_id++)); do
    sc_reference_start=0
    if (( LEGACY_IPC4_NO_SC == 1 && ipc_id == 4 )); then
        sc_reference_start=4
    fi
    log0="$LOG_DIR/ipc${ipc_id}_gpu${GPU0}_classes_0_100.log"
    log1="$LOG_DIR/ipc${ipc_id}_gpu${GPU1}_classes_100_200.log"
    echo "IPC $ipc_id (SC reference start=$sc_reference_start): GPU $GPU0 classes [0,100), GPU $GPU1 classes [100,200)"

    run_shard "$GPU0" 0 100 "$ipc_id" "$sc_reference_start" "$log0" &
    pid0=$!
    run_shard "$GPU1" 100 200 "$ipc_id" "$sc_reference_start" "$log1" &
    pid1=$!
    pids=("$pid0" "$pid1")

    status0=0
    status1=0
    wait "$pid0" || status0=$?
    wait "$pid1" || status1=$?
    pids=()

    if (( status0 != 0 || status1 != 0 )); then
        echo "IPC $ipc_id failed: GPU $GPU0 status=$status0, GPU $GPU1 status=$status1" >&2
        echo "Inspect logs in $LOG_DIR" >&2
        exit 1
    fi

    expected=$((200 * (ipc_id + 1)))
    actual=0
    if [[ -d "$SYN_DATA_DIR" ]]; then
        actual="$(find "$SYN_DATA_DIR" -type f -name '*.jpg' | wc -l)"
    fi
    if (( actual < expected )); then
        echo "IPC $ipc_id incomplete: found $actual images, expected at least $expected" >&2
        exit 1
    fi
    echo "IPC $ipc_id complete: $actual/$((200 * IPC_END)) total images"
done

trap - INT TERM
final_count="$(find "$SYN_DATA_DIR" -type f -name '*.jpg' | wc -l)"
echo "FD2 recover complete: $final_count images in $SYN_DATA_DIR"
