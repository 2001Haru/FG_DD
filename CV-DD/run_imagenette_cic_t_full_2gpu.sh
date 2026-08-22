#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RSEEDS <<< "$RSEEDS_TEXT"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$SSEEDS_TEXT"
C_VALUES_TEXT="${C_VALUES:-1 2 5 10}"; read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
PARTITION_SEED="${PARTITION_SEED:-42}"; TEACHER_SEED="${TEACHER_SEED:-42}"
VIEW_SEED="${VIEW_SEED:-42}"; TEMPERATURE="${TEMPERATURE:-20}"
# FKD batches are serialized as a unit: saved views, CutMix metadata and soft
# labels must be loaded with exactly the same batch size.  ImageNette relabel
# uses 10; gradient accumulation below only splits this batch for forward/backward.
readonly FKD_BATCH_SIZE=10
PATCH_SCORING_BATCH="${PATCH_SCORING_BATCH:-256}"
PATCH_CROP_WORKERS="${PATCH_CROP_WORKERS:-16}"
RELABEL_PERSISTENT_WORKERS="${RELABEL_PERSISTENT_WORKERS:-1}"
REAL_ROOT="${REAL_ROOT:-$val_dir/imagenet-nette}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
ASSET_ROOT="${ASSET_ROOT:-$EXP_ROOT}"
C1_TEACHER_OVERRIDE="${C1_TEACHER_OVERRIDE:-}"
DATA_ROOT="$ASSET_ROOT/data"; MODEL_ROOT="$ASSET_ROOT/models"; PATCH_ROOT="$EXP_ROOT/patches"
SYN_ROOT="$EXP_ROOT/synthetic"; FKD_ROOT="$EXP_ROOT/fkd"; POST_ROOT="$EXP_ROOT/post_eval"
PER_CLASS="$EXP_ROOT/per_class"; ANALYSIS="$EXP_ROOT/analysis"; LOGS="${LOGS:-$ROOT/logs/imagenette_cic_t_official_split/full}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
mkdir -p "$PATCH_ROOT" "$SYN_ROOT" "$FKD_ROOT" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"
fail(){ echo "ImageNette CiC-T full experiment failed: $*" >&2; exit 1; }
wait_jobs(){
    local status=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then status=1; fi
    done
    return "$status"
}

teacher_for_c(){
    local c="$1"
    if [[ "$c" == 1 && -n "$C1_TEACHER_OVERRIDE" ]]; then
        printf '%s\n' "$C1_TEACHER_OVERRIDE"
    else
        printf '%s\n' "$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}/ResNet18.pth"
    fi
}

for c in "${C_VALUES_ARRAY[@]}"; do
    data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    model="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    teacher="$(teacher_for_c "$c")"
    [[ -f "$data/hierarchy.json" ]] || fail "missing C=$c hierarchy"
    counts="$(python -c "import json; q=json.load(open('$data/hierarchy.json')); print(q.get('source_train_images'), q.get('source_val_images'), q.get('source_validation_split'))")"
    [[ "$counts" == "9469 3925 test" ]] || fail "C=$c partition is not the official train/test split: $counts"
    if [[ "$c" == 1 && -n "$C1_TEACHER_OVERRIDE" ]]; then
        [[ -f "$teacher" && "$(basename "$teacher")" == "ResNet18.pth" ]] \
            || fail "invalid C1 Teacher override (must exist and be named ResNet18.pth): $teacher"
    else
        [[ -f "$model/.training_complete.json" && -f "$teacher" ]] \
            || fail "C=$c Teacher is not marked complete"
    fi
done
[[ -d "$VAL_DIR" ]] || fail "missing official validation directory: $VAL_DIR"
VAL_IMAGE_COUNT="$(find "$VAL_DIR" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
VAL_CLASS_COUNT="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "$VAL_IMAGE_COUNT" == 3925 ]] \
    || fail "post-eval VAL_DIR has $VAL_IMAGE_COUNT images, expected full ImageNette 3925: $VAL_DIR"
[[ "$VAL_CLASS_COUNT" == 10 ]] \
    || fail "post-eval VAL_DIR has $VAL_CLASS_COUNT class dirs, expected 10: $VAL_DIR"
echo "Post-eval validation verified: path=$VAL_DIR images=$VAL_IMAGE_COUNT classes=$VAL_CLASS_COUNT"

patch_one(){
    local gpu="$1"
    local c="$2"
    local heads=$((10*c))
    local teacher
    teacher="$(teacher_for_c "$c")"
    local patch_root="$PATCH_ROOT/c${c}"
    local output="$patch_root/medium" count=0 archive
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    if (( count == 100 )) && python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
        --patch-dir "$patch_root" --classes 10 --patches-per-class 10 --image-size 224 \
        > "$LOGS/patch_validate_c${c}.log" 2>&1; then
        return
    fi
    if [[ -d "$patch_root" ]]; then
        archive="${patch_root}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || fail "patch archive already exists: $archive"
        mv "$patch_root" "$archive"
        echo "Archived invalid C=$c patch tree: $patch_root -> $archive"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" --data-dir "$REAL_ROOT" \
        --teacher "$teacher" \
        --teacher-num-classes "$heads" \
        --teacher-mapping "$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json" \
        --teacher-architecture torchvision --num-classes 10 --patches-per-class 10 \
        --candidate-images 100 --crops-per-image 5 --image-size 224 --normalization imagenet \
        --scoring-batch-size "$PATCH_SCORING_BATCH" --crop-workers "$PATCH_CROP_WORKERS" \
        --output-dir "$output" --seed 42 \
        > "$LOGS/patch_c${c}.log" 2>&1; then
        echo "C=$c patch generation process failed; see $LOGS/patch_c${c}.log" >&2
        return 1
    fi
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count==100 )) || { echo "C=$c patches incomplete after generation ($count/100)" >&2; return 1; }
    if ! python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
        --patch-dir "$patch_root" --classes 10 --patches-per-class 10 --image-size 224 \
        > "$LOGS/patch_validate_c${c}.log" 2>&1; then
        echo "C=$c generated patch tree failed validation; see $LOGS/patch_validate_c${c}.log" >&2
        return 1
    fi
}

echo "[1/4] Teacher-specific coarse10 patches"
pids=(); for c in "${C_VALUES_ARRAY[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    patch_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail patches; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail patches; fi

recover_one(){
    local gpu="$1"
    local c="$2"
    local rseed="$3"
    local heads=$((10*c))
    local exp="cic_t_c${c}_ipc10_rseed${rseed}" output="$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}"
    local count=0 marker="$output/.protocol" patch_sha archive
    teacher="$(teacher_for_c "$c")"
    mapping="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json"
    patch_sha="$(find "$PATCH_ROOT/c${c}/medium" -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
    expected="c=$c:rseed=$rseed:teacher=$(sha256sum "$teacher"|awk '{print $1}'):mapping=$(sha256sum "$mapping"|awk '{print $1}'):patch=$patch_sha:iter=4000"
    if [[ -d "$output" ]]; then
        count="$(find "$output" -type f -name '*.jpg' | wc -l)"
        if [[ -f "$marker" && "$(tr -d '[:space:]' < "$marker")" == "$expected" ]]; then
            (( count==100 )) && return
        elif (( count > 0 )); then
            archive="${output}.invalid_$(date +%Y%m%d_%H%M%S)"
            [[ ! -e "$archive" ]] || { echo "synthetic archive already exists: $archive" >&2; return 1; }
            mv "$output" "$archive"
            echo "Archived synthetic data with stale patch protocol: $output -> $archive"
        fi
    fi
    mkdir -p "$output"; printf '%s\n' "$expected" > "$marker"
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "$exp" --apply-data-augmentation \
        --dataset-name imagenet-nette --batch-size 10 --syn-data-path "$SYN_ROOT" \
        --patch-dir "$PATCH_ROOT/c${c}" --model-pool-dir "$(dirname "$teacher")" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --teacher-num-classes "$heads" --teacher-mapping "$mapping" \
        --voter-type equal --selected-size 1 --lr 0.25 --iteration 4000 --r-bn 0.01 \
        --store-best-images --ipc-start 0 --ipc-end 10 --initialisation-method Patches \
        --patch-diff medium --seed "$rseed" --skip-completed \
        > "$LOGS/recover_c${c}_rseed${rseed}.log" 2>&1; then
        echo "C=$c rseed=$rseed recovery process failed; see $LOGS/recover_c${c}_rseed${rseed}.log" >&2
        return 1
    fi
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count == 100 )) || { echo "C=$c rseed=$rseed recovery incomplete ($count/100)" >&2; return 1; }
}

echo "[2/4] Recovery: C=${C_VALUES_ARRAY[*]}, seeds=${RSEEDS[*]}"
pids=(); for rseed in "${RSEEDS[@]}"; do for c in "${C_VALUES_ARRAY[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    recover_one "$gpu" "$c" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail recovery; pids=(); fi
done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail recovery; fi

relabel_one(){
    local gpu="$1"
    local c="$2"
    local rseed="$3"
    local heads=$((10*c))
    local teacher
    teacher="$(teacher_for_c "$c")"
    local syn="$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}"
    local base="$FKD_ROOT/cic_t_c${c}_rseed${rseed}"
    local final="${base}_bs${FKD_BATCH_SIZE}_ipc10"
    local count=0 archive
    local worker_args=()
    if [[ "$RELABEL_PERSISTENT_WORKERS" == "1" ]]; then
        worker_args+=(--persistent-workers --prefetch-factor 4)
    fi
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count==3000 )) && return
    if (( count > 0 )); then
        archive="${final}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || { echo "FKD archive already exists: $archive" >&2; return 1; }
        mv "$final" "$archive"
        echo "Archived partial FKD ($count/3000): $final -> $archive"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" \
        --teacher-model-name ResNet18 --teacher-num-classes "$heads" \
        --teacher-mapping "$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json" \
        --marginalize-temperature "$TEMPERATURE" --gpu 0 --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" \
        "${worker_args[@]}" \
        --dataset-name imagenet-nette --epochs 300 --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_c${c}_rseed${rseed}.log" 2>&1; then
        echo "C=$c rseed=$rseed relabel process failed; see $LOGS/relabel_c${c}_rseed${rseed}.log" >&2
        return 1
    fi
    count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) || { echo "C=$c rseed=$rseed FKD incomplete ($count/3000)" >&2; return 1; }
}

echo "[3/4] Relabel marg10"
pids=(); for rseed in "${RSEEDS[@]}"; do for c in "${C_VALUES_ARRAY[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    relabel_one "$gpu" "$c" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail relabel; fi

validate_one(){
    local gpu="$1" c="$2" rseed="$3" sseed="$4"
    local result="$PER_CLASS/c${c}_rseed${rseed}_sseed${sseed}.json"
    if [[ -f "$result" ]]; then
        result_valid="$(python -c "import json,os; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==$VAL_IMAGE_COUNT and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR'))")"
        if [[ "$result_valid" == "True" ]]; then
            return
        fi
        archive="${result}.invalid_val_$(date +%Y%m%d_%H%M%S)"
        mv "$result" "$archive"
        echo "Archived post-eval result with unverified validation metadata: $archive"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 10 \
        --exp-name "imagenette_cic_t_c${c}_rseed${rseed}_sseed${sseed}" \
        --original-data-path "$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}" \
        --fkd-path "$FKD_ROOT/cic_t_c${c}_rseed${rseed}_bs${FKD_BATCH_SIZE}_ipc10" \
        --output-dir "$POST_ROOT" --batch-size "$FKD_BATCH_SIZE" --epochs 300 --dataset-name imagenet-nette \
        --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --train-seed "$sseed" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$LOGS/validate_c${c}_rseed${rseed}_sseed${sseed}.log" 2>&1; then
        echo "C=$c rseed=$rseed sseed=$sseed post-eval failed; see validation log" >&2
        return 1
    fi
    [[ -f "$result" ]] || { echo "post-eval completed without result: $result" >&2; return 1; }
}

echo "[4/4] Post-eval: C=${C_VALUES_ARRAY[*]}, recovery=${RSEEDS[*]}, student=${SSEEDS[*]}"
pids=(); for sseed in "${SSEEDS[@]}"; do for rseed in "${RSEEDS[@]}"; do for c in "${C_VALUES_ARRAY[@]}"; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    validate_one "$gpu" "$c" "$rseed" "$sseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail post_eval; pids=(); fi
done; done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail post_eval; fi

SUMMARY_OUTPUT="$ANALYSIS/summary.json"
if [[ "${C_VALUES_ARRAY[*]}" != "1 2 5 10" ]]; then
    c_tag="${C_VALUES_ARRAY[*]}"; c_tag="${c_tag// /_}"
    SUMMARY_OUTPUT="$ANALYSIS/summary_c${c_tag}.json"
fi
python "$ROOT/class_in_class/summarize_imagenette_cic_t.py" --per-class-dir "$PER_CLASS" \
    --recovery-seeds "${RSEEDS[@]}" --student-seeds "${SSEEDS[@]}" \
    --c-values "${C_VALUES_ARRAY[@]}" \
    --output "$SUMMARY_OUTPUT" > "$LOGS/summary.log" 2>&1
echo "Complete: $SUMMARY_OUTPUT"
