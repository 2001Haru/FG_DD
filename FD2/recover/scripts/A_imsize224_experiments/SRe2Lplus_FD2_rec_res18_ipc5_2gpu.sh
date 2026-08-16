#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOVER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
FD2_DIR="$(dirname "$RECOVER_DIR")"
source "$SCRIPT_DIR/constants.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ITERATIONS="${ITERATIONS:-4000}"
IPC_START="${IPC_START:-0}"
IPC_END="${IPC_END:-5}"
REC_NAME=rec_res18_ipc5
SYN_ROOT="${Main_Data_Path}/generated_data/syn_data/SRe2Lplus_FD2_${Dataset_Name}_09FC05_01SC4"
SYN_DIR="${SYN_ROOT}/${REC_NAME}"
PATCH_DIR="${Main_Data_Path}/patches/${Dataset_Name}"
MODEL_DIR="${Main_Data_Path}/pretrained_models/${Dataset_Name}"
CHECKPOINT="${MODEL_DIR}/ResNet18_M32_3e-1cal.pth"
LOG_DIR="${SCRIPT_DIR}/logs/${REC_NAME}_2gpu"
mkdir -p "$LOG_DIR"

[[ -f "$CHECKPOINT" ]] || { echo "Missing teacher: $CHECKPOINT" >&2; exit 1; }
patch_count="$(find "$PATCH_DIR/2" -type f -name '*.jpg' 2>/dev/null | wc -l)"
(( patch_count == 500 )) || { echo "Found $patch_count Aircraft patches; expected 500" >&2; exit 1; }

run_shard() {
    local gpu="$1" class_start="$2" class_end="$3" ipc_id="$4" sc_start="$5" log_file="$6"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$FD2_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    python -u "$RECOVER_DIR/recover_FD2.py" \
        --exp_name "$REC_NAME" --apply_data_augmentation \
        --dataset_name "$Dataset_Name" --class_num 100 \
        --class_start "$class_start" --class_end "$class_end" --subprocess_num 1 \
        --syn_data_path "$SYN_ROOT" --patch_dir "$PATCH_DIR" --model_pool_dir "$MODEL_DIR" \
        --pretrained_model_type offline --model_choice ResNet18 --M 32 --cal_ratio 0.3 \
        --voter_type equal --selected_size 1 --lr 1e-3 --iteration "$ITERATIONS" --r_bn 1e-3 \
        --FC --FC_ratio 0.9 --IntraFC_ratio 0.5 \
        --SC --SC_ratio 0.1 --SC_loss_threshold 0.0 --store_best_images \
        --ipc_start "$ipc_id" --ipc_end "$((ipc_id + 1))" --sc_reference_start "$sc_start" \
        --initialisation_method Patches --patch_diff 2 --skip_completed > "$log_file" 2>&1
}

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for ((ipc_id=IPC_START; ipc_id<IPC_END; ipc_id++)); do
    # Match the released grouping: IPC 4 is recovered independently and therefore has no earlier SC references.
    sc_start=0; (( ipc_id == 4 )) && sc_start=4
    echo "Aircraft IPC $ipc_id: GPU $GPU0 classes [0,50), GPU $GPU1 classes [50,100)"
    run_shard "$GPU0" 0 50 "$ipc_id" "$sc_start" "$LOG_DIR/ipc${ipc_id}_gpu${GPU0}.log" & pid0=$!
    run_shard "$GPU1" 50 100 "$ipc_id" "$sc_start" "$LOG_DIR/ipc${ipc_id}_gpu${GPU1}.log" & pid1=$!
    pids=("$pid0" "$pid1")
    status0=0; status1=0
    wait "$pid0" || status0=$?; wait "$pid1" || status1=$?
    pids=()
    (( status0 == 0 && status1 == 0 )) || { echo "IPC $ipc_id failed: $status0/$status1" >&2; exit 1; }
    count="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
    expected=$((100 * (ipc_id + 1)))
    (( count >= expected )) || { echo "IPC $ipc_id incomplete: $count/$expected" >&2; exit 1; }
done
trap - INT TERM
echo "Aircraft FD2 recovery complete: $(find "$SYN_DIR" -type f -name '*.jpg' | wc -l) images"
