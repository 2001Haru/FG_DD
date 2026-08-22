#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RSEEDS <<< "$RSEEDS_TEXT"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$SSEEDS_TEXT"
PARTITION_SEED="${PARTITION_SEED:-42}"
TEACHER_SEED="${TEACHER_SEED:-42}"
VIEW_SEED="${VIEW_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-20}"
readonly FKD_BATCH_SIZE=10

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
CONTROL_DATA="$EXP_ROOT/data/random_c1_pseed${PARTITION_SEED}"
CONTROL_TEACHER_DIR="$EXP_ROOT/models/random_c1_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
CONTROL_PATCH_DIR="$EXP_ROOT/patches/c1"
CONTROL_RESULTS="$EXP_ROOT/per_class"
OFFICIAL_TEACHER_DIR="${OFFICIAL_TEACHER_DIR:-$Main_Data_Path/offline_models/imagenet-nette}"
OFFICIAL_PATCH_DIR="${OFFICIAL_PATCH_DIR:-$Main_Data_Path/patches/imagenet-nette}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"

ABL_ROOT="${ABL_ROOT:-$EXP_ROOT/c1_teacher_patch_ablation}"
SYN_ROOT="$ABL_ROOT/synthetic"
FKD_ROOT="$ABL_ROOT/fkd"
POST_ROOT="$ABL_ROOT/post_eval"
PER_CLASS="$ABL_ROOT/per_class"
ANALYSIS="$ABL_ROOT/analysis"
LOGS="$ROOT/logs/imagenette_cic_t_official_split/c1_teacher_patch_ablation"
ARMS=(official_teacher_c1_patches c1_teacher_official_patches)

mkdir -p "$SYN_ROOT" "$FKD_ROOT" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"
fail(){ echo "ImageNette C1 Teacher/Patch ablation failed: $*" >&2; exit 1; }
wait_jobs(){
    local status=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then status=1; fi
    done
    return "$status"
}

[[ -f "$CONTROL_DATA/hierarchy.json" ]] || fail "missing controlled C1 hierarchy"
SOURCE_COUNTS="$(python -c "import json; q=json.load(open('$CONTROL_DATA/hierarchy.json')); print(q.get('source_train_images'), q.get('source_val_images'), q.get('source_validation_split'))")"
[[ "$SOURCE_COUNTS" == "9469 3925 test" ]] || fail "controlled C1 uses unsafe source split: $SOURCE_COUNTS"
[[ -f "$CONTROL_TEACHER_DIR/ResNet18.pth" ]] || fail "missing controlled C1 Teacher"
[[ -f "$OFFICIAL_TEACHER_DIR/ResNet18.pth" ]] || fail "missing official offline ResNet18"
CONTROL_PATCH_COUNT="$(find "$CONTROL_PATCH_DIR/medium" -type f -name '*.jpg' | wc -l)"
OFFICIAL_PATCH_COUNT="$(find "$OFFICIAL_PATCH_DIR/medium" -type f -name '*.jpg' | wc -l)"
(( CONTROL_PATCH_COUNT == 100 )) || fail "controlled C1 patches=$CONTROL_PATCH_COUNT, expected 100"
(( OFFICIAL_PATCH_COUNT == 500 )) || fail "official patches=$OFFICIAL_PATCH_COUNT, expected 500"
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$CONTROL_PATCH_DIR" --classes 10 --patches-per-class 10 --image-size 224 \
    > "$LOGS/patch_validate_controlled.log" 2>&1 \
    || fail "controlled C1 patch tree invalid; see $LOGS/patch_validate_controlled.log"
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$OFFICIAL_PATCH_DIR" --classes 10 --patches-per-class 50 --image-size 224 \
    > "$LOGS/patch_validate_official.log" 2>&1 \
    || fail "official patch tree invalid; see $LOGS/patch_validate_official.log"
VAL_IMAGE_COUNT="$(find "$VAL_DIR" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
VAL_CLASS_COUNT="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
(( VAL_IMAGE_COUNT == 3925 )) || fail "validation images=$VAL_IMAGE_COUNT, expected 3925: $VAL_DIR"
(( VAL_CLASS_COUNT == 10 )) || fail "validation classes=$VAL_CLASS_COUNT, expected 10: $VAL_DIR"

arm_paths(){
    local arm="$1"
    case "$arm" in
        official_teacher_c1_patches)
            ARM_TEACHER_DIR="$OFFICIAL_TEACHER_DIR"
            ARM_TEACHER="$OFFICIAL_TEACHER_DIR/ResNet18.pth"
            ARM_PATCH_DIR="$CONTROL_PATCH_DIR"
            ARM_MAPPING=""
            ;;
        c1_teacher_official_patches)
            ARM_TEACHER_DIR="$CONTROL_TEACHER_DIR"
            ARM_TEACHER="$CONTROL_TEACHER_DIR/ResNet18.pth"
            ARM_PATCH_DIR="$OFFICIAL_PATCH_DIR"
            ARM_MAPPING="$CONTROL_DATA/hierarchy.json"
            ;;
        *) fail "unknown arm: $arm" ;;
    esac
}

recover_one(){
    local gpu="$1" arm="$2" rseed="$3"
    local output="$SYN_ROOT/${arm}_ipc10_rseed${rseed}"
    local marker="$output/.protocol" count=0 expected
    local extra=()
    arm_paths "$arm"
    [[ -n "$ARM_MAPPING" ]] && extra+=(--teacher-num-classes 10 --teacher-mapping "$ARM_MAPPING")
    expected="arm=$arm:rseed=$rseed:teacher=$(sha256sum "$ARM_TEACHER" | awk '{print $1}'):patch_dir=$(realpath "$ARM_PATCH_DIR"):iter=4000"
    if [[ -d "$output" ]]; then
        [[ -f "$marker" ]] || fail "missing protocol marker: $output"
        [[ "$(tr -d '[:space:]' < "$marker")" == "$expected" ]] || fail "protocol mismatch: $output"
        count="$(find "$output" -type f -name '*.jpg' | wc -l)"
        (( count == 100 )) && return
    else
        mkdir -p "$output"
        printf '%s\n' "$expected" > "$marker"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" \
        --exp-name "${arm}_ipc10_rseed${rseed}" --apply-data-augmentation \
        --dataset-name imagenet-nette --batch-size 10 --syn-data-path "$SYN_ROOT" \
        --patch-dir "$ARM_PATCH_DIR" --model-pool-dir "$ARM_TEACHER_DIR" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        "${extra[@]}" --voter-type equal --selected-size 1 --lr 0.25 --iteration 4000 \
        --r-bn 0.01 --store-best-images --ipc-start 0 --ipc-end 10 \
        --initialisation-method Patches --patch-diff medium --seed "$rseed" --skip-completed \
        > "$LOGS/recover_${arm}_rseed${rseed}.log" 2>&1
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count == 100 )) || fail "$arm rseed=$rseed recovery incomplete ($count/100)"
}

echo "[1/3] Recovery: two crossed arms x three seeds"
pids=()
for rseed in "${RSEEDS[@]}"; do
    recover_one "$GPU0" "${ARMS[0]}" "$rseed" & pids+=("$!")
    recover_one "$GPU1" "${ARMS[1]}" "$rseed" & pids+=("$!")
    wait_jobs "${pids[@]}" || fail "recovery rseed=$rseed"
    pids=()
done

relabel_one(){
    local gpu="$1" arm="$2" rseed="$3"
    local syn="$SYN_ROOT/${arm}_ipc10_rseed${rseed}"
    local base="$FKD_ROOT/${arm}_rseed${rseed}"
    local final="${base}_bs${FKD_BATCH_SIZE}_ipc10" count=0
    local extra=()
    arm_paths "$arm"
    if [[ -n "$ARM_MAPPING" ]]; then
        extra+=(--teacher-num-classes 10 --teacher-mapping "$ARM_MAPPING" \
                --marginalize-temperature "$TEMPERATURE")
    fi
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    (( count == 0 )) || fail "partial FKD: $final ($count/3000)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$ARM_TEACHER_DIR" --teacher-model-name ResNet18 "${extra[@]}" \
        --gpu 0 --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_${arm}_rseed${rseed}.log" 2>&1
}

echo "[2/3] Relabel: matching Teacher, identical views"
pids=()
for rseed in "${RSEEDS[@]}"; do
    relabel_one "$GPU0" "${ARMS[0]}" "$rseed" & pids+=("$!")
    relabel_one "$GPU1" "${ARMS[1]}" "$rseed" & pids+=("$!")
    wait_jobs "${pids[@]}" || fail "relabel rseed=$rseed"
    pids=()
done

validate_one(){
    local gpu="$1" arm="$2" rseed="$3" sseed="$4"
    local result="$PER_CLASS/${arm}_rseed${rseed}_sseed${sseed}.json"
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
        --exp-name "imagenette_${arm}_rseed${rseed}_sseed${sseed}" \
        --original-data-path "$SYN_ROOT/${arm}_ipc10_rseed${rseed}" \
        --fkd-path "$FKD_ROOT/${arm}_rseed${rseed}_bs${FKD_BATCH_SIZE}_ipc10" \
        --output-dir "$POST_ROOT" --batch-size "$FKD_BATCH_SIZE" --epochs 300 \
        --dataset-name imagenet-nette --gradient-accumulation-steps 2 --mix-type cutmix \
        --cos --workers "$WORKERS" --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" \
        --adamw-weight-decay 0.01 --train-seed "$sseed" --persistent-workers \
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOGS/validate_${arm}_rseed${rseed}_sseed${sseed}.log" 2>&1
}

echo "[3/3] Post-eval: two arms x three recovery x three student seeds"
pids=()
for sseed in "${SSEEDS[@]}"; do
    for rseed in "${RSEEDS[@]}"; do
        validate_one "$GPU0" "${ARMS[0]}" "$rseed" "$sseed" & pids+=("$!")
        validate_one "$GPU1" "${ARMS[1]}" "$rseed" "$sseed" & pids+=("$!")
        wait_jobs "${pids[@]}" || fail "post-eval rseed=$rseed sseed=$sseed"
        pids=()
    done
done

python "$ROOT/class_in_class/summarize_imagenette_c1_teacher_patch_ablation.py" \
    --control-per-class-dir "$CONTROL_RESULTS" --ablation-per-class-dir "$PER_CLASS" \
    --recovery-seeds "${RSEEDS[@]}" --student-seeds "${SSEEDS[@]}" \
    --output "$ANALYSIS/summary.json" > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
