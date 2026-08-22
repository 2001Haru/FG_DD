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
RECOVERY_ITERATIONS="${RECOVERY_ITERATIONS:-2000}"
RECOVERY_LR="${RECOVERY_LR:-0.25}"
readonly RECOVERY_ITERATIONS RECOVERY_LR
readonly FKD_BATCH_SIZE=10

CONTROL_ROOT="${CONTROL_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
CONTROL_DATA="$CONTROL_ROOT/data/random_c1_pseed42"
CONTROL_TEACHER_DIR="$CONTROL_ROOT/models/random_c1_pseed42_tseed42"
CONTROL_TEACHER="$CONTROL_TEACHER_DIR/ResNet18.pth"
CONTROL_PATCH="$CONTROL_ROOT/patches/c1"
OFFICIAL_TEACHER_DIR="$Main_Data_Path/offline_models/imagenet-nette"
OFFICIAL_TEACHER="$OFFICIAL_TEACHER_DIR/ResNet18.pth"
OFFICIAL_PATCH="$Main_Data_Path/patches/imagenet-nette"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"

EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_recovery_iter2000}"
SYN_ROOT="$EXP_ROOT/synthetic"
FKD_ROOT="$EXP_ROOT/fkd"
POST_ROOT="$EXP_ROOT/post_eval"
PER_CLASS="$EXP_ROOT/per_class"
ANALYSIS="$EXP_ROOT/analysis"
LOGS="${LOGS:-$ROOT/logs/imagenette_recovery_iter2000}"
ARMS=(official controlled_seed42)
mkdir -p "$SYN_ROOT" "$FKD_ROOT" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"

fail(){ echo "ImageNette recovery-iter2000 experiment failed: $*" >&2; exit 1; }
wait_jobs(){
    local status=0 pid
    for pid in "$@"; do if ! wait "$pid"; then status=1; fi; done
    return "$status"
}

[[ -f "$CONTROL_DATA/hierarchy.json" ]] || fail "missing controlled hierarchy"
COUNTS="$(python -c "import json; q=json.load(open('$CONTROL_DATA/hierarchy.json')); print(q.get('source_train_images'), q.get('source_val_images'), q.get('source_validation_split'))")"
[[ "$COUNTS" == "9469 3925 test" ]] || fail "controlled data uses unsafe split: $COUNTS"
for path in "$CONTROL_TEACHER" "$OFFICIAL_TEACHER"; do
    [[ -f "$path" ]] || fail "missing Teacher: $path"
done
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$CONTROL_PATCH" --classes 10 --patches-per-class 10 --image-size 224 \
    > "$LOGS/patch_validate_controlled.log" 2>&1 \
    || fail "controlled patches invalid; see $LOGS/patch_validate_controlled.log"
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$OFFICIAL_PATCH" --classes 10 --patches-per-class 50 --image-size 224 \
    > "$LOGS/patch_validate_official.log" 2>&1 \
    || fail "official patches invalid; see $LOGS/patch_validate_official.log"
VAL_IMAGES="$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
VAL_CLASSES="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
(( VAL_IMAGES == 3925 && VAL_CLASSES == 10 )) || fail "invalid test set: images=$VAL_IMAGES classes=$VAL_CLASSES"

arm_config(){
    local arm="$1"
    case "$arm" in
        official)
            ARM_TEACHER="$OFFICIAL_TEACHER"
            ARM_TEACHER_DIR="$OFFICIAL_TEACHER_DIR"
            ARM_PATCH="$OFFICIAL_PATCH"
            ARM_MAPPING=""
            ;;
        controlled_seed42)
            ARM_TEACHER="$CONTROL_TEACHER"
            ARM_TEACHER_DIR="$CONTROL_TEACHER_DIR"
            ARM_PATCH="$CONTROL_PATCH"
            ARM_MAPPING="$CONTROL_DATA/hierarchy.json"
            ;;
        *) fail "unknown arm: $arm" ;;
    esac
}

recover_one(){
    local gpu="$1" arm="$2"
    local output="$SYN_ROOT/${arm}_ipc10_rseed${RECOVERY_SEED}"
    local marker="$output/.protocol" count=0 patch_sha expected archive
    local extra=()
    arm_config "$arm"
    [[ -n "$ARM_MAPPING" ]] && extra+=(--teacher-num-classes 10 --teacher-mapping "$ARM_MAPPING")
    patch_sha="$(find "$ARM_PATCH/medium" -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
    expected="arm=$arm:rseed=$RECOVERY_SEED:teacher=$(sha256sum "$ARM_TEACHER"|awk '{print $1}'):patch=$patch_sha:iter=$RECOVERY_ITERATIONS:lr=$RECOVERY_LR"
    if [[ -d "$output" ]]; then
        count="$(find "$output" -type f -name '*.jpg' | wc -l)"
        if [[ -f "$marker" && "$(tr -d '[:space:]' < "$marker")" == "$expected" ]]; then
            (( count == 100 )) && return
        elif (( count > 0 )); then
            archive="${output}.invalid_$(date +%Y%m%d_%H%M%S)"
            [[ ! -e "$archive" ]] || { echo "archive exists: $archive" >&2; return 1; }
            mv "$output" "$archive"
        fi
    fi
    mkdir -p "$output"; printf '%s\n' "$expected" > "$marker"
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "${arm}_ipc10_rseed${RECOVERY_SEED}" \
        --apply-data-augmentation --dataset-name imagenet-nette --batch-size 10 \
        --syn-data-path "$SYN_ROOT" --patch-dir "$ARM_PATCH" --model-pool-dir "$ARM_TEACHER_DIR" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        "${extra[@]}" --voter-type equal --selected-size 1 --lr "$RECOVERY_LR" \
        --iteration "$RECOVERY_ITERATIONS" --r-bn 0.01 --store-best-images \
        --ipc-start 0 --ipc-end 10 --initialisation-method Patches --patch-diff medium \
        --seed "$RECOVERY_SEED" --skip-completed > "$LOGS/recover_${arm}.log" 2>&1; then
        echo "$arm recovery failed; see $LOGS/recover_${arm}.log" >&2
        return 1
    fi
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count == 100 )) || { echo "$arm recovery incomplete ($count/100)" >&2; return 1; }
}

echo "[1/3] Recovery iteration=$RECOVERY_ITERATIONS lr=$RECOVERY_LR"
recover_one "$GPU0" official & pid0=$!
recover_one "$GPU1" controlled_seed42 & pid1=$!
wait_jobs "$pid0" "$pid1" || fail recovery

relabel_one(){
    local gpu="$1" arm="$2"
    local syn="$SYN_ROOT/${arm}_ipc10_rseed${RECOVERY_SEED}"
    local base="$FKD_ROOT/${arm}_rseed${RECOVERY_SEED}"
    local final="${base}_bs${FKD_BATCH_SIZE}_ipc10" count=0 archive
    local extra=()
    arm_config "$arm"
    if [[ -n "$ARM_MAPPING" ]]; then
        extra+=(--teacher-num-classes 10 --teacher-mapping "$ARM_MAPPING" \
                --marginalize-temperature "$TEMPERATURE")
    fi
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    if (( count > 0 )); then
        archive="${final}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || { echo "archive exists: $archive" >&2; return 1; }
        mv "$final" "$archive"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$ARM_TEACHER_DIR" --teacher-model-name ResNet18 "${extra[@]}" \
        --gpu 0 --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" --persistent-workers \
        --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" --min-scale-crops 0.08 \
        --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_${arm}.log" 2>&1; then
        echo "$arm relabel failed; see $LOGS/relabel_${arm}.log" >&2
        return 1
    fi
    count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) || { echo "$arm FKD incomplete ($count/3000)" >&2; return 1; }
}

echo "[2/3] BSSL relabel with identical seed=$VIEW_SEED"
relabel_one "$GPU0" official & pid0=$!
relabel_one "$GPU1" controlled_seed42 & pid1=$!
wait_jobs "$pid0" "$pid1" || fail relabel

validate_one(){
    local gpu="$1" arm="$2" sseed="$3"
    local result="$PER_CLASS/${arm}_rseed${RECOVERY_SEED}_sseed${sseed}.json"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json,os; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR'))")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 10 \
        --exp-name "iter2000_${arm}_rseed${RECOVERY_SEED}_sseed${sseed}" \
        --original-data-path "$SYN_ROOT/${arm}_ipc10_rseed${RECOVERY_SEED}" \
        --fkd-path "$FKD_ROOT/${arm}_rseed${RECOVERY_SEED}_bs${FKD_BATCH_SIZE}_ipc10" \
        --output-dir "$POST_ROOT" --batch-size "$FKD_BATCH_SIZE" --epochs 300 \
        --dataset-name imagenet-nette --gradient-accumulation-steps 2 --mix-type cutmix \
        --cos --workers "$WORKERS" --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" \
        --adamw-weight-decay 0.01 --train-seed "$sseed" --persistent-workers \
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOGS/validate_${arm}_sseed${sseed}.log" 2>&1; then
        echo "$arm sseed=$sseed post-eval failed" >&2
        return 1
    fi
    [[ -f "$result" ]] || { echo "missing result: $result" >&2; return 1; }
}

echo "[3/3] Post-eval student seeds=${SSEEDS[*]}"
for sseed in "${SSEEDS[@]}"; do
    validate_one "$GPU0" official "$sseed" & pid0=$!
    validate_one "$GPU1" controlled_seed42 "$sseed" & pid1=$!
    wait_jobs "$pid0" "$pid1" || fail "post-eval sseed=$sseed"
done

python "$ROOT/class_in_class/summarize_imagenette_iter2000.py" \
    --per-class-dir "$PER_CLASS" --recovery-seed "$RECOVERY_SEED" \
    --recovery-iterations "$RECOVERY_ITERATIONS" --recovery-lr "$RECOVERY_LR" \
    --student-seeds "${SSEEDS[@]}" --output "$ANALYSIS/summary.json" \
    > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
