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
readonly FKD_BATCH_SIZE=16
readonly RECOVERY_ITERATIONS=4000
readonly POST_LR=0.001
readonly POST_ETA=2

TEACHER_DIR="${OFFICIAL_TEACHER_DIR:-$Main_Data_Path/offline_models/imagenet-nette}"
PATCH_DIR="${OFFICIAL_PATCH_DIR:-$Main_Data_Path/patches/imagenet-nette}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_official_lr1e3_bs16_eta2}"
SYN_BASE="$EXP_ROOT/synthetic"
SYN_DIR="$SYN_BASE/official_ipc10_rseed${RECOVERY_SEED}"
FKD_BASE="$EXP_ROOT/fkd/official_rseed${RECOVERY_SEED}"
FKD_DIR="${FKD_BASE}_bs${FKD_BATCH_SIZE}_ipc10"
POST_ROOT="$EXP_ROOT/post_eval"
PER_CLASS="$EXP_ROOT/per_class"
ANALYSIS="$EXP_ROOT/analysis"
LOGS="$ROOT/logs/imagenette_official_lr1e3_bs16_eta2"
mkdir -p "$SYN_BASE" "$(dirname "$FKD_BASE")" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"

fail(){ echo "ImageNette official LR1e-3/BS16/eta2 experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

[[ -f "$TEACHER_DIR/ResNet18.pth" ]] || fail "missing official ResNet18: $TEACHER_DIR"
PATCH_COUNT="$(find "$PATCH_DIR/medium" -type f -name '*.jpg' | wc -l)"
(( PATCH_COUNT == 500 )) || fail "official medium patches=$PATCH_COUNT, expected 500"
VAL_IMAGE_COUNT="$(find "$VAL_DIR" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
VAL_CLASS_COUNT="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
(( VAL_IMAGE_COUNT == 3925 )) || fail "validation images=$VAL_IMAGE_COUNT, expected 3925: $VAL_DIR"
(( VAL_CLASS_COUNT == 10 )) || fail "validation classes=$VAL_CLASS_COUNT, expected 10: $VAL_DIR"
echo "Preflight passed: official Teacher, 500 patches, full 3925-image test"

recover(){
    local count=0 marker="$SYN_DIR/.protocol" expected
    expected="teacher=$(sha256sum "$TEACHER_DIR/ResNet18.pth" | awk '{print $1}'):patch_dir=$(realpath "$PATCH_DIR"):rseed=$RECOVERY_SEED:iter=$RECOVERY_ITERATIONS:ipc=10"
    if [[ -d "$SYN_DIR" ]]; then
        [[ -f "$marker" ]] || fail "missing recovery protocol marker: $SYN_DIR"
        [[ "$(tr -d '[:space:]' < "$marker")" == "$expected" ]] || fail "recovery protocol mismatch: $SYN_DIR"
        count="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
        (( count == 100 )) && return
    else
        mkdir -p "$SYN_DIR"
        printf '%s\n' "$expected" > "$marker"
    fi
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "official_ipc10_rseed${RECOVERY_SEED}" \
        --apply-data-augmentation --dataset-name imagenet-nette --batch-size 10 \
        --syn-data-path "$SYN_BASE" --patch-dir "$PATCH_DIR" --model-pool-dir "$TEACHER_DIR" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --voter-type equal --selected-size 1 --lr 0.25 --iteration "$RECOVERY_ITERATIONS" \
        --r-bn 0.01 --store-best-images --ipc-start 0 --ipc-end 10 \
        --initialisation-method Patches --patch-diff medium --seed "$RECOVERY_SEED" \
        --skip-completed > "$LOGS/recover_rseed${RECOVERY_SEED}.log" 2>&1
    count="$(find "$SYN_DIR" -type f -name '*.jpg' | wc -l)"
    (( count == 100 )) || fail "recovery incomplete ($count/100)"
}

echo "[1/3] Official recovery: seed=$RECOVERY_SEED"
recover

relabel(){
    local count=0
    [[ -d "$FKD_DIR" ]] && count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 2100 )) && return
    (( count == 0 )) || fail "partial BS16 FKD: $FKD_DIR ($count/2100)"
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$SYN_DIR" --fkd-path "$FKD_BASE" \
        --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 \
        --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" --persistent-workers \
        --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_bs16.log" 2>&1
    count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 2100 )) || fail "BS16 FKD incomplete ($count/2100)"
}

echo "[2/3] Official relabel: BS16, 300 epochs, fixed views seed=$VIEW_SEED"
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
        --exp-name "official_lr1e3_bs16_eta2_rseed${RECOVERY_SEED}_sseed${sseed}" \
        --original-data-path "$SYN_DIR" --fkd-path "$FKD_DIR" --output-dir "$POST_ROOT" \
        --batch-size "$FKD_BATCH_SIZE" --gradient-accumulation-steps 2 --epochs 300 \
        --dataset-name imagenet-nette --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --adamw-lr-override "$POST_LR" --eta-override "$POST_ETA" --train-seed "$sseed" \
        --persistent-workers --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOGS/validate_sseed${sseed}.log" 2>&1
}

echo "[3/3] Post-eval: LR=$POST_LR BS=$FKD_BATCH_SIZE eta=$POST_ETA, three student seeds"
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
    > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
