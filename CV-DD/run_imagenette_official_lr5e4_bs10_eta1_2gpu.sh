#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
RECOVERY_SEED="${RECOVERY_SEED:-42}"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$SSEEDS_TEXT"
VIEW_SEED="${VIEW_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-20}"
readonly FKD_BATCH_SIZE=10
readonly RECOVERY_ITERATIONS=4000
readonly POST_LR=0.0005
readonly POST_ETA=1

TEACHER_DIR="${OFFICIAL_TEACHER_DIR:-$Main_Data_Path/offline_models/imagenet-nette}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
# Reuse exactly the same official-Teacher/official-patch recovery produced by
# the BS16 experiment. This makes the comparison isolate relabel/post settings.
RECOVERY_ROOT="${RECOVERY_ROOT:-$Main_Data_Path/class_in_class/imagenette_official_lr1e3_bs16_eta2}"
SYN_DIR="$RECOVERY_ROOT/synthetic/official_ipc10_rseed${RECOVERY_SEED}"

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_official_lr5e4_bs10_eta1}"
FKD_BASE="$EXP_ROOT/fkd/official_rseed${RECOVERY_SEED}"
FKD_DIR="${FKD_BASE}_bs${FKD_BATCH_SIZE}_ipc10"
POST_ROOT="$EXP_ROOT/post_eval"
PER_CLASS="$EXP_ROOT/per_class"
ANALYSIS="$EXP_ROOT/analysis"
LOGS="$ROOT/logs/imagenette_official_lr5e4_bs10_eta1"
mkdir -p "$(dirname "$FKD_BASE")" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"

fail(){ echo "ImageNette official LR5e-4/BS10/eta1 experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

[[ -f "$TEACHER_DIR/ResNet18.pth" ]] || fail "missing official ResNet18: $TEACHER_DIR"
SYN_COUNT="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
(( SYN_COUNT == 100 )) || fail "shared official recovery has $SYN_COUNT images, expected 100: $SYN_DIR"
VAL_IMAGE_COUNT="$(find "$VAL_DIR" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
VAL_CLASS_COUNT="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
(( VAL_IMAGE_COUNT == 3925 )) || fail "validation images=$VAL_IMAGE_COUNT, expected 3925: $VAL_DIR"
(( VAL_CLASS_COUNT == 10 )) || fail "validation classes=$VAL_CLASS_COUNT, expected 10: $VAL_DIR"
echo "Preflight passed: reusing official recovery seed=$RECOVERY_SEED ($SYN_COUNT images)"

relabel(){
    local count=0
    [[ -d "$FKD_DIR" ]] && count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    (( count == 0 )) || fail "partial BS10 FKD: $FKD_DIR ($count/3000)"
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$SYN_DIR" --fkd-path "$FKD_BASE" \
        --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 \
        --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" --persistent-workers \
        --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_bs10.log" 2>&1
    count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) || fail "BS10 FKD incomplete ($count/3000)"
}

echo "[1/2] Official BSSL relabel: BS10, 300 epochs, view seed=$VIEW_SEED"
relabel

validate_one(){
    local gpu="$1" sseed="$2"
    local result="$PER_CLASS/rseed${RECOVERY_SEED}_sseed${sseed}.json"
    local result_valid archive
    if [[ -f "$result" ]]; then
        result_valid="$(python -c "import json,os; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==$VAL_IMAGE_COUNT and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR'))")"
        [[ "$result_valid" == True ]] && return
        archive="${result}.invalid_val_$(date +%Y%m%d_%H%M%S)"
        mv "$result" "$archive"
        echo "Archived result with invalid validation metadata: $archive"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 10 \
        --exp-name "official_lr5e4_bs10_eta1_rseed${RECOVERY_SEED}_sseed${sseed}" \
        --original-data-path "$SYN_DIR" --fkd-path "$FKD_DIR" --output-dir "$POST_ROOT" \
        --batch-size "$FKD_BATCH_SIZE" --gradient-accumulation-steps 2 --epochs 300 \
        --dataset-name imagenet-nette --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --adamw-lr-override "$POST_LR" --eta-override "$POST_ETA" --train-seed "$sseed" \
        --persistent-workers --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOGS/validate_sseed${sseed}.log" 2>&1
}

echo "[2/2] Post-eval: LR=$POST_LR BS=$FKD_BATCH_SIZE eta=$POST_ETA, three student seeds"
pids=()
for sseed in "${SSEEDS[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]} == 1 )) && gpu="$GPU1"
    validate_one "$gpu" "$sseed" & pids+=("$!")
    if (( ${#pids[@]} == 2 )); then
        wait_jobs "${pids[@]}" || fail "post-eval"
        pids=()
    fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail "post-eval"; fi

python "$ROOT/class_in_class/summarize_imagenette_official_protocol.py" \
    --per-class-dir "$PER_CLASS" --recovery-seed "$RECOVERY_SEED" \
    --student-seeds "${SSEEDS[@]}" --output "$ANALYSIS/summary.json" \
    --batch-size "$FKD_BATCH_SIZE" --adamw-lr "$POST_LR" --eta "$POST_ETA" \
    --temperature "$TEMPERATURE" --recovery-iterations "$RECOVERY_ITERATIONS" \
    > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
