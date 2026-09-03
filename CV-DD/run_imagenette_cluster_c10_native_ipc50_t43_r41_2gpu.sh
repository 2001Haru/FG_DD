#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
POST_PARALLEL_JOBS="${POST_PARALLEL_JOBS:-4}"
TEACHER_SEED=43
RECOVERY_SEED=41
STUDENT_SEEDS=(42 43 44)
RECOVERY_BATCH_SIZE=100
RECOVERY_IPC_PER_SUBCLASS=5
COARSE_IPC=50
RECOVERY_ITERATIONS=4000
RECOVERY_LR=0.1
RECOVERY_R_BN=0.01
FKD_BATCH_SIZE=10
TEMPERATURE=20

CLUSTER_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42"
C1_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
OLD_IPC_ROOT="$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cluster_c10_native_ipc50_t43_r41"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cluster_c10_native_ipc50_t43_r41}"
VAL_DIR="$val_dir/imagenet-nette/test"
PARTITION="$CLUSTER_ROOT/data/dinov2_cluster_c10_seed42"
CLUSTER_TEACHER="$CLUSTER_ROOT/tseed43/models/dinov2_cluster_c10_seed42_tseed43/ResNet18.pth"
C1_TEACHER="$C1_ROOT/tseed43/models/random_c1_pseed42_tseed43/ResNet18.pth"
PATCH_ROOT="$EXP_ROOT/patches/cluster_c10_native100_ipc5"
NATIVE_SYN_ROOT="$EXP_ROOT/native_synthetic"
NATIVE_SYN="$NATIVE_SYN_ROOT/cluster_c10_native100_ipc5_rseed41"
COARSE_SOURCE="$EXP_ROOT/coarse_source/cluster_c10_native100_ipc5_rseed41"
FKD_BASE="$EXP_ROOT/fkd/c1_soft_rseed41"
FKD_FINAL="${FKD_BASE}_bs10_ipc50"
POST_ROOT="$EXP_ROOT/post_eval"
RESULT_ROOT="$EXP_ROOT/per_class"
mkdir -p "$EXP_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$POST_ROOT"

fail(){ echo "Cluster-C10 native IPC50 experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

echo "[0/6] Strict protocol preflight"
[[ -f "$PARTITION/hierarchy.json" && -f "$CLUSTER_TEACHER" && -f "$C1_TEACHER" ]] \
    || fail "missing partition/Teacher assets"
partition_valid="$(python -c "import json; q=json.load(open('$PARTITION/hierarchy.json')); m=q.get('fine_to_coarse',{}); print(q.get('kind')=='imagenette_balanced_dinov2_clusters' and int(q.get('subclasses_per_coarse',-1))==10 and len(m)==100 and all(int(m[str(i)])==i//10 for i in range(100)))")"
[[ "$partition_valid" == True ]] || fail "invalid parent-major DINO Cluster C10 hierarchy"
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "post-eval is not the full ImageNette test split"

echo "[1/6] Generate/reuse five native patches per subclass"
patch_count=0
[[ -d "$PATCH_ROOT/medium" ]] && patch_count="$(find "$PATCH_ROOT/medium" -type f -name '*.jpg' | wc -l)"
if ! (( patch_count == 500 )) || ! python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$PATCH_ROOT" --classes 100 --patches-per-class 5 --image-size 224 \
    > "$LOG_ROOT/patch_validate.log" 2>&1; then
    if (( patch_count > 0 )); then
        archive="${PATCH_ROOT}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || fail "patch archive collision"
        mv "$PATCH_ROOT" "$archive"
    fi
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" \
        --data-dir "$PARTITION" --teacher "$CLUSTER_TEACHER" --teacher-num-classes 100 \
        --teacher-architecture torchvision --num-classes 100 --patches-per-class 5 \
        --candidate-images 100 --crops-per-image 5 --image-size 224 \
        --normalization imagenet --scoring-batch-size 256 --crop-workers 16 \
        --output-dir "$PATCH_ROOT/medium" --seed 42 \
        > "$LOG_ROOT/patch.log" 2>&1 || fail patches
fi
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$PATCH_ROOT" --classes 100 --patches-per-class 5 --image-size 224 \
    > "$LOG_ROOT/patch_validate.log" 2>&1 || fail patch_validation

echo "[2/6] Native-100 BS100 recovery, five images per subclass"
marker="$NATIVE_SYN/.protocol"
patch_hash="$(find "$PATCH_ROOT/medium" -type f -name '*.jpg' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
expected="native100:tseed=43:rseed=41:teacher=$(sha256sum "$CLUSTER_TEACHER"|awk '{print $1}'):hierarchy=$(sha256sum "$PARTITION/hierarchy.json"|awk '{print $1}'):patch=$patch_hash:batch=100:native_ipc=5:iter=4000:lr=0.1:r_bn=0.01"
native_count=0
[[ -d "$NATIVE_SYN" ]] && native_count="$(find "$NATIVE_SYN" -type f -name '*.jpg' | wc -l)"
if [[ ! -f "$marker" || "$(tr -d '[:space:]' < "$marker")" != "$expected" || "$native_count" != 500 ]]; then
    if (( native_count > 0 )); then
        archive="${NATIVE_SYN}.invalid_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || fail "native recovery archive collision"
        mv "$NATIVE_SYN" "$archive"
    fi
    mkdir -p "$NATIVE_SYN"
    printf '%s\n' "$expected" > "$marker"
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" \
        --exp-name "$(basename "$NATIVE_SYN")" --apply-data-augmentation \
        --dataset-name imagenet-nette --recovery-num-classes 100 --teacher-num-classes 100 \
        --batch-size 100 --syn-data-path "$NATIVE_SYN_ROOT" --patch-dir "$PATCH_ROOT" \
        --model-pool-dir "$(dirname "$CLUSTER_TEACHER")" --pretrained-model-type offline \
        --model-setting 0 --sre2l-model ResNet18 --voter-type equal --selected-size 1 \
        --lr "$RECOVERY_LR" --iteration "$RECOVERY_ITERATIONS" --r-bn "$RECOVERY_R_BN" \
        --store-best-images --ipc-start 0 --ipc-end 5 --initialisation-method Patches \
        --patch-diff medium --seed 41 --skip-completed \
        > "$LOG_ROOT/recover.log" 2>&1 || fail recovery
fi
[[ "$(find "$NATIVE_SYN" -type f -name '*.jpg' | wc -l)" == 500 ]] || fail "native recovery count"
[[ "$(find "$NATIVE_SYN" -mindepth 1 -maxdepth 1 -type d | wc -l)" == 100 ]] || fail "native directory count"

echo "[3/6] Group five images per subclass into coarse IPC50"
python -u "$ROOT/class_in_class/collapse_synthetic_subclass_imagefolder.py" \
    --input-dir "$NATIVE_SYN" --hierarchy "$PARTITION/hierarchy.json" \
    --output-dir "$COARSE_SOURCE" --native-classes 100 --coarse-classes 10 \
    --images-per-native 5 > "$LOG_ROOT/collapse.log" 2>&1 || fail collapse
[[ "$(find "$COARSE_SOURCE" -type f -name '*.jpg' | wc -l)" == 500 ]] || fail "coarse source count"

echo "[4/6] C1 CutMix FKD relabel at IPC50"
fkd_count=0
[[ -d "$FKD_FINAL" ]] && fkd_count="$(find "$FKD_FINAL" -type f -name 'batch_*.tar' | wc -l)"
if (( fkd_count != 15000 )); then
    if (( fkd_count > 0 )); then
        archive="${FKD_FINAL}.partial_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || fail "FKD archive collision"
        mv "$FKD_FINAL" "$archive"
    fi
    CUDA_VISIBLE_DEVICES="$GPU1" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$COARSE_SOURCE" --fkd-path "$FKD_BASE" \
        --model-pool-dir "$(dirname "$C1_TEACHER")" --teacher-model-name ResNet18 \
        --teacher-num-classes 10 --gpu 0 --batch-size 10 --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel.log" 2>&1 || fail relabel
fi
[[ "$(find "$FKD_FINAL" -type f -name 'batch_*.tar' | wc -l)" == 15000 ]] || fail "FKD count"

post_one(){
    local label_arm="$1" student_seed="$2" gpu="$3"
    local result expected_target valid run_name
    result="$RESULT_ROOT/${label_arm}_sseed${student_seed}.json"
    expected_target=hard_coarse_label
    [[ "$label_arm" == c1_soft ]] && expected_target=fkd_soft_label
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='$expected_target')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    run_name="cluster_c10_native_ipc50_${label_arm}_t43_r41_s${student_seed}"
    common=(
        --model ResNet18 --ipc 50 --exp-name "$run_name"
        --original-data-path "$COARSE_SOURCE" --output-dir "$POST_ROOT"
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette
        --gradient-accumulation-steps 2 --cos --workers "$WORKERS"
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005
        --eta-override 2 --train-seed "$student_seed" --persistent-workers
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result"
    )
    if [[ "$label_arm" == hard ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/validate/train_fkd.py" --hard-label "${common[@]}" \
            > "$LOG_ROOT/post_hard_s${student_seed}.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/validate/train_fkd.py" \
            --fkd-path "$FKD_FINAL" --mix-type cutmix --temperature 20 "${common[@]}" \
            > "$LOG_ROOT/post_c1soft_s${student_seed}.log" 2>&1
    fi
    [[ -f "$result" ]]
}

echo "[5/6] Hard/C1-Soft x three students, two processes per A100"
pids=()
task=0
for label_arm in hard c1_soft; do
    for student_seed in "${STUDENT_SEEDS[@]}"; do
        gpu="$GPU0"
        (( task % 2 )) && gpu="$GPU1"
        task=$((task + 1))
        post_one "$label_arm" "$student_seed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == POST_PARALLEL_JOBS )); then
            wait_jobs "${pids[@]}" || fail post_eval
            pids=()
        fi
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post_eval

echo "[6/6] Summarize and pair against prior IPC50 cells"
python -u "$ROOT/class_in_class/summarize_imagenette_cluster_c10_native_ipc50.py" \
    --experiment-root "$EXP_ROOT" --old-ipc-root "$OLD_IPC_ROOT" \
    --output "$EXP_ROOT/analysis/summary.json" \
    > "$LOG_ROOT/summarize.log" 2>&1 || fail summarize

echo "Complete: $EXP_ROOT/analysis/summary.json"
