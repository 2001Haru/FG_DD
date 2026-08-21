#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RSEEDS <<< "$RSEEDS_TEXT"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$SSEEDS_TEXT"
PARTITION_SEED="${PARTITION_SEED:-42}"; TEACHER_SEED="${TEACHER_SEED:-42}"
VIEW_SEED="${VIEW_SEED:-42}"; TEMPERATURE="${TEMPERATURE:-20}"
# FKD batches are serialized as a unit: saved views, CutMix metadata and soft
# labels must be loaded with exactly the same batch size.  ImageNette relabel
# uses 10; gradient accumulation below only splits this batch for forward/backward.
readonly FKD_BATCH_SIZE=10
PATCH_SCORING_BATCH="${PATCH_SCORING_BATCH:-256}"
PATCH_CROP_WORKERS="${PATCH_CROP_WORKERS:-16}"
RELABEL_PERSISTENT_WORKERS="${RELABEL_PERSISTENT_WORKERS:-1}"
REAL_ROOT="${REAL_ROOT:-/linxi/dataset/VLCP/ImageNette}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t}"
DATA_ROOT="$EXP_ROOT/data"; MODEL_ROOT="$EXP_ROOT/models"; PATCH_ROOT="$EXP_ROOT/patches"
SYN_ROOT="$EXP_ROOT/synthetic"; FKD_ROOT="$EXP_ROOT/fkd"; POST_ROOT="$EXP_ROOT/post_eval"
PER_CLASS="$EXP_ROOT/per_class"; ANALYSIS="$EXP_ROOT/analysis"; LOGS="$ROOT/logs/imagenette_cic_t/full"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
mkdir -p "$PATCH_ROOT" "$SYN_ROOT" "$FKD_ROOT" "$POST_ROOT" "$PER_CLASS" "$ANALYSIS" "$LOGS"
fail(){ echo "ImageNette CiC-T full experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

for c in 1 2 5 10; do
    data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    model="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    [[ -f "$data/hierarchy.json" ]] || fail "missing C=$c hierarchy"
    [[ -f "$model/.training_complete.json" && -f "$model/ResNet18.pth" ]] \
        || fail "C=$c Teacher is not marked complete"
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
    local output="$PATCH_ROOT/c${c}/medium" count=0
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count==100 )) && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" --data-dir "$REAL_ROOT" \
        --teacher "$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}/ResNet18.pth" \
        --teacher-num-classes "$heads" \
        --teacher-mapping "$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json" \
        --teacher-architecture torchvision --num-classes 10 --patches-per-class 10 \
        --candidate-images 100 --crops-per-image 5 --image-size 224 --normalization imagenet \
        --scoring-batch-size "$PATCH_SCORING_BATCH" --crop-workers "$PATCH_CROP_WORKERS" \
        --output-dir "$output" --seed 42 \
        > "$LOGS/patch_c${c}.log" 2>&1
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    (( count==100 )) || fail "C=$c patches incomplete after generation ($count/100)"
}

echo "[1/4] Teacher-specific coarse10 patches"
pids=(); for c in 1 2 5 10; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    patch_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail patches; pids=(); fi
done

recover_one(){
    local gpu="$1"
    local c="$2"
    local rseed="$3"
    local heads=$((10*c))
    local exp="cic_t_c${c}_ipc10_rseed${rseed}" output="$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}"
    local count=0 marker="$output/.protocol"
    teacher="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}/ResNet18.pth"
    mapping="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json"
    expected="c=$c:rseed=$rseed:teacher=$(sha256sum "$teacher"|awk '{print $1}'):mapping=$(sha256sum "$mapping"|awk '{print $1}'):iter=4000"
    if [[ -d "$output" ]]; then
        [[ -f "$marker" ]] || fail "missing protocol marker: $output"
        [[ "$(tr -d '[:space:]' < "$marker")" == "$expected" ]] || fail "protocol mismatch: $output"
        count="$(find "$output" -type f -name '*.jpg' | wc -l)"
        (( count==100 )) && return
    else
        mkdir -p "$output"; printf '%s\n' "$expected" > "$marker"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "$exp" --apply-data-augmentation \
        --dataset-name imagenet-nette --batch-size 10 --syn-data-path "$SYN_ROOT" \
        --patch-dir "$PATCH_ROOT/c${c}" --model-pool-dir "$(dirname "$teacher")" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --teacher-num-classes "$heads" --teacher-mapping "$mapping" \
        --voter-type equal --selected-size 1 --lr 0.25 --iteration 4000 --r-bn 0.01 \
        --store-best-images --ipc-start 0 --ipc-end 10 --initialisation-method Patches \
        --patch-diff medium --seed "$rseed" --skip-completed \
        > "$LOGS/recover_c${c}_rseed${rseed}.log" 2>&1
}

echo "[2/4] Recovery: 4 arms x 3 seeds"
pids=(); for rseed in "${RSEEDS[@]}"; do for c in 1 2 5 10; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    recover_one "$gpu" "$c" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail recovery; pids=(); fi
done; done

relabel_one(){
    local gpu="$1"
    local c="$2"
    local rseed="$3"
    local heads=$((10*c))
    local syn="$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}"
    local base="$FKD_ROOT/cic_t_c${c}_rseed${rseed}"
    local final="${base}_bs${FKD_BATCH_SIZE}_ipc10"
    local count=0
    local worker_args=()
    if [[ "$RELABEL_PERSISTENT_WORKERS" == "1" ]]; then
        worker_args+=(--persistent-workers --prefetch-factor 4)
    fi
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count==3000 )) && return; (( count==0 )) || fail "partial FKD $final ($count/3000)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" \
        --model-pool-dir "$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}" \
        --teacher-model-name ResNet18 --teacher-num-classes "$heads" \
        --teacher-mapping "$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}/hierarchy.json" \
        --marginalize-temperature "$TEMPERATURE" --gpu 0 --batch-size "$FKD_BATCH_SIZE" --workers "$WORKERS" \
        "${worker_args[@]}" \
        --dataset-name imagenet-nette --epochs 300 --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOGS/relabel_c${c}_rseed${rseed}.log" 2>&1
}

echo "[3/4] Relabel marg10"
pids=(); for rseed in "${RSEEDS[@]}"; do for c in 1 2 5 10; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    relabel_one "$gpu" "$c" "$rseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
done; done

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
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 10 \
        --exp-name "imagenette_cic_t_c${c}_rseed${rseed}_sseed${sseed}" \
        --original-data-path "$SYN_ROOT/cic_t_c${c}_ipc10_rseed${rseed}" \
        --fkd-path "$FKD_ROOT/cic_t_c${c}_rseed${rseed}_bs${FKD_BATCH_SIZE}_ipc10" \
        --output-dir "$POST_ROOT" --batch-size "$FKD_BATCH_SIZE" --epochs 300 --dataset-name imagenet-nette \
        --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature "$TEMPERATURE" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --train-seed "$sseed" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$LOGS/validate_c${c}_rseed${rseed}_sseed${sseed}.log" 2>&1
}

echo "[4/4] Post-eval: 4 x 3 x 3"
pids=(); for sseed in "${SSEEDS[@]}"; do for rseed in "${RSEEDS[@]}"; do for c in 1 2 5 10; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    validate_one "$gpu" "$c" "$rseed" "$sseed" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail post_eval; pids=(); fi
done; done; done

python "$ROOT/class_in_class/summarize_imagenette_cic_t.py" --per-class-dir "$PER_CLASS" \
    --recovery-seeds "${RSEEDS[@]}" --student-seeds "${SSEEDS[@]}" \
    --output "$ANALYSIS/summary.json" > "$LOGS/summary.log" 2>&1
echo "Complete: $ANALYSIS/summary.json"
