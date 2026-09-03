#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
TEACHER_SEEDS=(43 44)
RECOVERY_SEEDS=(41 42 43)
STUDENT_SEEDS=(42 43 44)
CLUSTER_SEED=42
C=10
NATIVE_CLASSES=100
COARSE_CLASSES=10
RECOVERY_ITERATIONS=4000
RECOVERY_LR=0.1
RECOVERY_R_BN=0.01
RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-10}"
FKD_BATCH_SIZE=10
TEMPERATURE=20

CLUSTER_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42"
C1_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cluster_c10_native_recovery_c1_labeler}"
PATCH_ASSET_ROOT="${PATCH_ASSET_ROOT:-$EXP_ROOT}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cluster_c10_native_recovery_c1_labeler}"
VAL_DIR="$val_dir/imagenet-nette/test"
mkdir -p "$EXP_ROOT/analysis" "$LOG_ROOT"

fail(){ echo "Cluster-C10 native-recovery experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

partition_dir(){
    echo "$CLUSTER_ROOT/data/dinov2_cluster_c10_seed${CLUSTER_SEED}"
}
cluster_teacher(){
    local teacher_seed="$1"
    echo "$CLUSTER_ROOT/tseed${teacher_seed}/models/dinov2_cluster_c10_seed${CLUSTER_SEED}_tseed${teacher_seed}/ResNet18.pth"
}
c1_teacher(){
    local teacher_seed="$1"
    echo "$C1_ROOT/tseed${teacher_seed}/models/random_c1_pseed42_tseed${teacher_seed}/ResNet18.pth"
}

echo "[0/6] Strict asset and protocol preflight"
[[ "$RECOVERY_BATCH_SIZE" == 10 || "$RECOVERY_BATCH_SIZE" == 100 ]] \
    || fail "RECOVERY_BATCH_SIZE must be 10 or 100"
partition="$(partition_dir)"
[[ -f "$partition/hierarchy.json" ]] || fail "missing DINO Cluster C10 hierarchy"
partition_valid="$(python -c "import json; q=json.load(open('$partition/hierarchy.json')); m=q.get('fine_to_coarse',{}); print(q.get('kind')=='imagenette_balanced_dinov2_clusters' and int(q.get('subclasses_per_coarse',-1))==10 and int(q.get('num_pseudo_classes',-1))==100 and len(m)==100 and all(int(m[str(i)])==i//10 for i in range(100)))")"
[[ "$partition_valid" == True ]] || fail "invalid DINO Cluster C10 hierarchy/mapping"
[[ "$(find "$partition/train" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 9469 ]] \
    || fail "Cluster partition train split is not the official 9469 images"
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "post-eval is not the full 3925-image test split"
for teacher_seed in "${TEACHER_SEEDS[@]}"; do
    cluster_model="$(cluster_teacher "$teacher_seed")"
    c1_model="$(c1_teacher "$teacher_seed")"
    [[ -f "$cluster_model" ]] || fail "missing Cluster C10 Teacher seed$teacher_seed"
    [[ -f "$c1_model" ]] || fail "missing C1 relabel Teacher seed$teacher_seed"
done

patch_one(){
    local teacher_seed="$1" gpu="$2"
    local teacher patch_root output count archive
    teacher="$(cluster_teacher "$teacher_seed")"
    patch_root="$PATCH_ASSET_ROOT/tseed${teacher_seed}/patches/cluster_c10_native100"
    output="$patch_root/medium"
    mkdir -p "$PATCH_ASSET_ROOT/tseed${teacher_seed}"
    count=0
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    if (( count == NATIVE_CLASSES )) && python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
        --patch-dir "$patch_root" --classes "$NATIVE_CLASSES" --patches-per-class 1 \
        --image-size 224 > "$LOG_ROOT/patch_validate_t${teacher_seed}.log" 2>&1; then
        echo "Reusing native100 patches: tseed=$teacher_seed"
        return
    fi
    if [[ -d "$patch_root" ]]; then
        archive="${patch_root}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || return 1
        mv "$patch_root" "$archive"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" \
        --data-dir "$partition" --teacher "$teacher" --teacher-num-classes 100 \
        --teacher-architecture torchvision --num-classes 100 --patches-per-class 1 \
        --candidate-images 100 --crops-per-image 5 --image-size 224 \
        --normalization imagenet --scoring-batch-size 256 --crop-workers 16 \
        --output-dir "$output" --seed 42 \
        > "$LOG_ROOT/patch_t${teacher_seed}.log" 2>&1 || return 1
    python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
        --patch-dir "$patch_root" --classes 100 --patches-per-class 1 --image-size 224 \
        > "$LOG_ROOT/patch_validate_t${teacher_seed}.log" 2>&1
}

echo "[1/6] Teacher-specific native-100 patches, one Teacher per A100"
patch_one 43 "$GPU0" & patch43=$!
patch_one 44 "$GPU1" & patch44=$!
wait_jobs "$patch43" "$patch44" || fail patches

recover_one(){
    local teacher_seed="$1" recovery_seed="$2" gpu="$3"
    local teacher patch_root seed_root output exp_name marker expected count patch_hash archive
    teacher="$(cluster_teacher "$teacher_seed")"
    patch_root="$PATCH_ASSET_ROOT/tseed${teacher_seed}/patches/cluster_c10_native100"
    seed_root="$EXP_ROOT/tseed${teacher_seed}/native_synthetic"
    exp_name="cluster_c10_native100_rseed${recovery_seed}"
    output="$seed_root/$exp_name"
    marker="$output/.protocol"
    patch_hash="$(find "$patch_root/medium" -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
    expected="native100:tseed=$teacher_seed:rseed=$recovery_seed:teacher=$(sha256sum "$teacher"|awk '{print $1}'):hierarchy=$(sha256sum "$partition/hierarchy.json"|awk '{print $1}'):patch=$patch_hash:batch=$RECOVERY_BATCH_SIZE:iter=$RECOVERY_ITERATIONS:lr=$RECOVERY_LR:r_bn=$RECOVERY_R_BN"
    count=0
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    if [[ -f "$marker" && "$(tr -d '[:space:]' < "$marker")" == "$expected" && "$count" == 100 ]]; then
        echo "Reusing native100 recovery: tseed=$teacher_seed rseed=$recovery_seed"
        return
    fi
    if [[ -d "$output" && "$count" != 0 ]]; then
        archive="${output}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || return 1
        mv "$output" "$archive"
    fi
    mkdir -p "$output"
    printf '%s\n' "$expected" > "$marker"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" \
        --exp-name "$exp_name" --apply-data-augmentation \
        --dataset-name imagenet-nette --recovery-num-classes 100 \
        --teacher-num-classes 100 --batch-size "$RECOVERY_BATCH_SIZE" --syn-data-path "$seed_root" \
        --patch-dir "$patch_root" --model-pool-dir "$(dirname "$teacher")" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --voter-type equal --selected-size 1 --lr "$RECOVERY_LR" \
        --iteration "$RECOVERY_ITERATIONS" --r-bn "$RECOVERY_R_BN" \
        --store-best-images --ipc-start 0 --ipc-end 1 --initialisation-method Patches \
        --patch-diff medium --seed "$recovery_seed" --skip-completed \
        > "$LOG_ROOT/recover_t${teacher_seed}_r${recovery_seed}.log" 2>&1 || return 1
    [[ "$(find "$output" -type f -name '*.jpg' | wc -l)" == 100 ]] || return 1
    [[ "$(find "$output" -mindepth 1 -maxdepth 1 -type d | wc -l)" == 100 ]] || return 1
}

recovery_stream(){
    local teacher_seed="$1" gpu="$2" recovery_seed
    for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
        recover_one "$teacher_seed" "$recovery_seed" "$gpu" || return 1
    done
}

echo "[2/6] Native-100 recovery: 100 targets x one image, no marginalization, BS=$RECOVERY_BATCH_SIZE"
recovery_stream 43 "$GPU0" & recovery43=$!
recovery_stream 44 "$GPU1" & recovery44=$!
wait_jobs "$recovery43" "$recovery44" || fail recovery

collapse_one(){
    local teacher_seed="$1" recovery_seed="$2"
    local input output
    input="$EXP_ROOT/tseed${teacher_seed}/native_synthetic/cluster_c10_native100_rseed${recovery_seed}"
    output="$EXP_ROOT/tseed${teacher_seed}/coarse_sources/cluster_c10_native100_rseed${recovery_seed}"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/collapse_synthetic_subclass_imagefolder.py" \
        --input-dir "$input" --hierarchy "$partition/hierarchy.json" \
        --output-dir "$output" --native-classes 100 --coarse-classes 10 \
        --images-per-native 1 \
        > "$LOG_ROOT/collapse_t${teacher_seed}_r${recovery_seed}.log" 2>&1
}

echo "[3/6] Lossless grouping of 100 native directories back to ten parents"
pids=()
for teacher_seed in "${TEACHER_SEEDS[@]}"; do
    for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
        collapse_one "$teacher_seed" "$recovery_seed" & pids+=("$!")
    done
done
wait_jobs "${pids[@]}" || fail collapse

relabel_one(){
    local teacher_seed="$1" recovery_seed="$2" gpu="$3"
    local teacher source seed_root base final count archive
    teacher="$(c1_teacher "$teacher_seed")"
    source="$EXP_ROOT/tseed${teacher_seed}/coarse_sources/cluster_c10_native100_rseed${recovery_seed}"
    seed_root="$EXP_ROOT/tseed${teacher_seed}"
    base="$seed_root/fkd/c1_soft_rseed${recovery_seed}"
    final="${base}_bs${FKD_BATCH_SIZE}_ipc10"
    mkdir -p "$seed_root/fkd"
    count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    if (( count > 0 )); then
        archive="${final}.partial_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || return 1
        mv "$final" "$archive"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$source" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 \
        --teacher-num-classes 10 --gpu 0 --batch-size "$FKD_BATCH_SIZE" \
        --workers "$WORKERS" --persistent-workers --prefetch-factor 4 \
        --dataset-name imagenet-nette --epochs 300 --fkd-seed 42 --seed 42 \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 \
        --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_c1_t${teacher_seed}_r${recovery_seed}.log" 2>&1 || return 1
    [[ "$(find "$final" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]]
}

relabel_stream(){
    local teacher_seed="$1" gpu="$2" recovery_seed
    for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
        relabel_one "$teacher_seed" "$recovery_seed" "$gpu" || return 1
    done
}

echo "[4/6] C1-only CutMix FKD relabel; Cluster C10 Teacher is not loaded here"
relabel_stream 43 "$GPU0" & relabel43=$!
relabel_stream 44 "$GPU1" & relabel44=$!
wait_jobs "$relabel43" "$relabel44" || fail relabel

post_one(){
    local label_arm="$1" teacher_seed="$2" recovery_seed="$3" student_seed="$4" gpu="$5"
    local seed_root source result run_name valid
    seed_root="$EXP_ROOT/tseed${teacher_seed}"
    source="$seed_root/coarse_sources/cluster_c10_native100_rseed${recovery_seed}"
    result="$seed_root/per_class/${label_arm}_rseed${recovery_seed}_sseed${student_seed}.json"
    run_name="cluster_c10_native100_${label_arm}_t${teacher_seed}_r${recovery_seed}_s${student_seed}"
    mkdir -p "$seed_root/per_class" "$seed_root/post_eval"
    if [[ -f "$result" ]]; then
        expected_target=hard_coarse_label
        [[ "$label_arm" == c1_soft ]] && expected_target=fkd_soft_label
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='$expected_target')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    common=(
        --model ResNet18 --ipc 10 --exp-name "$run_name"
        --original-data-path "$source" --output-dir "$seed_root/post_eval"
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette
        --gradient-accumulation-steps 2 --cos --workers "$WORKERS"
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005
        --eta-override 1 --train-seed "$student_seed" --persistent-workers
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result"
    )
    if [[ "$label_arm" == hard ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/validate/train_fkd.py" --hard-label "${common[@]}" \
            > "$LOG_ROOT/post_hard_t${teacher_seed}_r${recovery_seed}_s${student_seed}.log" 2>&1
    else
        fkd="$seed_root/fkd/c1_soft_rseed${recovery_seed}_bs10_ipc10"
        [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] || return 1
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/validate/train_fkd.py" \
            --fkd-path "$fkd" --mix-type cutmix --temperature "$TEMPERATURE" \
            "${common[@]}" \
            > "$LOG_ROOT/post_c1soft_t${teacher_seed}_r${recovery_seed}_s${student_seed}.log" 2>&1
    fi
    [[ -f "$result" ]]
}

echo "[5/6] Hard versus C1 CutMix-soft post-eval; two processes per A100"
pids=()
task=0
for label_arm in hard c1_soft; do
    for teacher_seed in "${TEACHER_SEEDS[@]}"; do
        for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
            for student_seed in "${STUDENT_SEEDS[@]}"; do
                gpu="$GPU0"
                (( task % 2 )) && gpu="$GPU1"
                task=$((task + 1))
                post_one "$label_arm" "$teacher_seed" "$recovery_seed" "$student_seed" "$gpu" &
                pids+=("$!")
                if (( ${#pids[@]} == 4 )); then
                    wait_jobs "${pids[@]}" || fail post_eval
                    pids=()
                fi
            done
        done
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post_eval

echo "[6/6] Strict paired summary"
python -u "$ROOT/class_in_class/summarize_imagenette_native100_recovery_c1_labeler.py" \
    --experiment-root "$EXP_ROOT" --recovery-batch-size "$RECOVERY_BATCH_SIZE" \
    --output "$EXP_ROOT/analysis/summary.json" \
    > "$LOG_ROOT/summarize.log" 2>&1 || fail summarize

echo "Complete: $EXP_ROOT/analysis/summary.json"
