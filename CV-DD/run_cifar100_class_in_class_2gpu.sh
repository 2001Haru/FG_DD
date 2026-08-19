#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
ITERATIONS="${ITERATIONS:-4000}"
RAW_ARCHIVE="${RAW_ARCHIVE:-/linxi/dataset/CV-DD/raw/cifar100/cifar-100-python.tar.gz}"
RAW_DIR="${RAW_DIR:-/linxi/dataset/CV-DD/raw/cifar100/cifar-100-python}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/cifar100}"

DATA="$EXP_ROOT/data"
MODELS="$EXP_ROOT/models"
PATCHES="$EXP_ROOT/patches"
SYN_PARENT="$EXP_ROOT/synthetic"
FKD_PARENT="$EXP_ROOT/fkd"
OUTPUT="$EXP_ROOT/post_eval"
LOGS="$ROOT/logs/cifar100_class_in_class"
FINE_DATA="$DATA/fine"
COARSE_DATA="$DATA/coarse"
MAPPING="$DATA/hierarchy.json"
FINE_MODELS="$MODELS/fine100"
COARSE_MODELS="$MODELS/coarse20"
FINE_PATCH_ROOT="$PATCHES/fine100"
COARSE_PATCH_ROOT="$PATCHES/coarse20"
FINE_SYN="$SYN_PARENT/oracle_fine100_ipc5"
BASE_SYN="$SYN_PARENT/baseline_coarse20_ipc25"
ORACLE_SYN="$SYN_PARENT/oracle_merged_coarse20_ipc25"
BASE_FKD_BASE="$FKD_PARENT/baseline_coarse20"
ORACLE_FKD_BASE="$FKD_PARENT/oracle_merged_coarse20"
BASE_FKD="${BASE_FKD_BASE}_bs16_ipc25"
ORACLE_FKD="${ORACLE_FKD_BASE}_bs16_ipc25"

mkdir -p "$EXP_ROOT" "$MODELS" "$PATCHES" "$SYN_PARENT" "$FKD_PARENT" "$OUTPUT" "$LOGS"
fail() { echo "Preflight failed: $*" >&2; exit 1; }

if [[ ! -f "$RAW_DIR/train" ]]; then
    [[ -f "$RAW_ARCHIVE" ]] || fail "missing official archive: $RAW_ARCHIVE"
    echo "eb9058c3a382ffc7106e4002c42a8d85  $RAW_ARCHIVE" | md5sum -c - || \
        fail "CIFAR-100 archive checksum mismatch"
    mkdir -p "$(dirname "$RAW_DIR")"
    tar -xzf "$RAW_ARCHIVE" -C "$(dirname "$RAW_DIR")"
fi

cat <<EOF
===== CV-DD SRe2L++ Class-in-Class Oracle =====
Task: 20-way CIFAR-100 coarse classification
Baseline: coarse20 x IPC25 = 500
Oracle: fine100 x IPC5, merge 5-to-1 = 500
Teacher (native CV-DD): ResNet18, Adam lr=1e-3 wd=1e-4, batch512, 200 epochs
Recovery (native CV-DD): class batch20 for both, Adam lr=0.25, r_bn=0.01, $ITERATIONS iterations
BSSL (native CV-DD): same coarse20 teacher, batch16, 300 epochs, CutMix, fp16
Post-eval (native CV-DD CIFAR-100 protocol): random CV-DD ResNet18
Post-eval: batch16, 300 epochs, AdamW lr=1e-3 wd=0.01, eta=1, T=20, grad_accum=2
Seed=$SEED
================================================
EOF

if [[ ! -f "$MAPPING" ]]; then
    echo "[1/7] Preparing fine/coarse ImageFolder datasets"
    python "$ROOT/class_in_class/prepare_cifar100_hierarchy.py" --raw-dir "$RAW_DIR" --output-dir "$DATA"
fi
[[ "$(find "$FINE_DATA/train" -type f -name '*.png' | wc -l)" == 50000 ]] || fail "fine train set incomplete"
[[ "$(find "$COARSE_DATA/test" -type f -name '*.png' | wc -l)" == 10000 ]] || fail "coarse test set incomplete"

train_teacher() {
    local gpu="$1" dataset_name="$2" data_dir="$3" save_dir="$4" log="$5"
    if [[ -f "$save_dir/ResNet18.pth" ]]; then return; fi
    mkdir -p "$save_dir"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/squeeze/squeeze.py" \
        --model-list ResNet18 --optimizer Adam --dataset-dir "$data_dir" --save-dir "$save_dir" \
        --batch-size 512 --dataset-name "$dataset_name" --epoch 200 --lr 0.001 --seed "$SEED" > "$log" 2>&1
}

echo "[2/7] Training native CV-DD fine100/coarse20 teachers"
train_teacher "$GPU0" cifar100 "$FINE_DATA" "$FINE_MODELS" "$LOGS/teacher_fine100.log" & p0=$!
train_teacher "$GPU1" cifar20 "$COARSE_DATA" "$COARSE_MODELS" "$LOGS/teacher_coarse20.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?
(( s0 == 0 && s1 == 0 )) || fail "teacher training failed: fine=$s0 coarse=$s1"

generate_patches() {
    local gpu="$1" data="$2" teacher="$3" classes="$4" ipc="$5" root="$6" log="$7"
    local directory="$root/medium" expected=$((classes * ipc)) count=0
    [[ -d "$directory" ]] && count="$(find "$directory" -type f -name '*.jpg' | wc -l)"
    if (( count == expected )); then return; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" \
        --data-dir "$data" --teacher "$teacher" --num-classes "$classes" \
        --patches-per-class "$ipc" --candidate-images 100 --crops-per-image 5 \
        --output-dir "$directory" --seed "$SEED" > "$log" 2>&1
}

echo "[3/7] Generating balanced RDED medium patches"
generate_patches "$GPU0" "$FINE_DATA" "$FINE_MODELS/ResNet18.pth" 100 5 "$FINE_PATCH_ROOT" "$LOGS/patches_fine100.log" & p0=$!
generate_patches "$GPU1" "$COARSE_DATA" "$COARSE_MODELS/ResNet18.pth" 20 25 "$COARSE_PATCH_ROOT" "$LOGS/patches_coarse20.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?
(( s0 == 0 && s1 == 0 )) || fail "patch generation failed: fine=$s0 coarse=$s1"

recover_route() {
    local gpu="$1" dataset="$2" classes="$3" ipc="$4" model_dir="$5" patch_root="$6" exp="$7" log="$8"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" \
        --exp-name "$exp" --apply-data-augmentation --dataset-name "$dataset" --batch-size 20 \
        --syn-data-path "$SYN_PARENT" --patch-dir "$patch_root" --model-pool-dir "$model_dir" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --voter-type equal --selected-size 1 --lr 0.25 --iteration "$ITERATIONS" --r-bn 0.01 \
        --store-best-images --skip-completed --ipc-start 0 --ipc-end "$ipc" \
        --initialisation-method Patches --patch-diff medium --seed "$SEED" > "$log" 2>&1
}

echo "[4/7] Running native CV-DD SRe2L++ recovery for both routes"
recover_route "$GPU0" cifar100 100 5 "$FINE_MODELS" "$FINE_PATCH_ROOT" "oracle_fine100_ipc5" "$LOGS/recover_oracle.log" & p0=$!
recover_route "$GPU1" cifar20 20 25 "$COARSE_MODELS" "$COARSE_PATCH_ROOT" "baseline_coarse20_ipc25" "$LOGS/recover_baseline.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?
(( s0 == 0 && s1 == 0 )) || fail "recovery failed: oracle=$s0 baseline=$s1"

oracle_count=0
[[ -d "$ORACLE_SYN" ]] && oracle_count="$(find "$ORACLE_SYN" -type f -name '*.jpg' | wc -l)"
if (( oracle_count == 0 )); then
    echo "[5/7] Merging oracle fine classes back into coarse classes"
    python "$ROOT/class_in_class/merge_fine_synthetic_to_coarse.py" \
        --fine-dir "$FINE_SYN" --mapping "$MAPPING" --output-dir "$ORACLE_SYN" --fine-ipc 5
elif (( oracle_count != 500 )); then
    fail "partial merged Oracle directory: $ORACLE_SYN ($oracle_count/500)"
fi
[[ "$(find "$BASE_SYN" -type f -name '*.jpg' | wc -l)" == 500 ]] || fail "baseline synthetic count is not 500"
[[ "$(find "$ORACLE_SYN" -type f -name '*.jpg' | wc -l)" == 500 ]] || fail "oracle synthetic count is not 500"

relabel_route() {
    local gpu="$1" syn="$2" base="$3" final="$4" log="$5" count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    if (( count == 9600 )); then return; fi
    if (( count != 0 )); then fail "partial FKD directory: $final ($count/9600)"; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$syn" --fkd-path "$base" --model-pool-dir "$COARSE_MODELS" \
        --teacher-model-name ResNet18 --gpu 0 --batch-size 16 --workers "$WORKERS" \
        --dataset-name cifar20 --epochs 300 --fkd-seed "$SEED" --seed "$SEED" \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$log" 2>&1
}

echo "[6/7] Generating identical coarse20 BSSL for both routes"
relabel_route "$GPU0" "$ORACLE_SYN" "$ORACLE_FKD_BASE" "$ORACLE_FKD" "$LOGS/relabel_oracle.log" & p0=$!
relabel_route "$GPU1" "$BASE_SYN" "$BASE_FKD_BASE" "$BASE_FKD" "$LOGS/relabel_baseline.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?
(( s0 == 0 && s1 == 0 )) || fail "relabel failed: oracle=$s0 baseline=$s1"

validate_route() {
    local gpu="$1" route="$2" syn="$3" fkd="$4" log="$5"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" \
        --model ResNet18 --ipc 25 --exp-name "class_in_class_${route}_seed${SEED}" \
        --original-data-path "$syn" --fkd-path "$fkd" --output-dir "$OUTPUT" \
        --batch-size 16 --epochs 300 --dataset-name cifar20 --gradient-accumulation-steps 2 \
        --mix-type cutmix --cos --workers "$WORKERS" --temperature 20 --fkd_seed "$SEED" \
        --adamw-weight-decay 0.01 \
        --train-seed "$SEED" --persistent-workers --val-dir "$COARSE_DATA/test" --disable-wandb \
        > "$log" 2>&1
}

echo "[7/7] Running native CV-DD SRe2L++ post-evaluation"
validate_route "$GPU0" oracle "$ORACLE_SYN" "$ORACLE_FKD" "$LOGS/validate_oracle.log" & p0=$!
validate_route "$GPU1" baseline "$BASE_SYN" "$BASE_FKD" "$LOGS/validate_baseline.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?
(( s0 == 0 && s1 == 0 )) || fail "post-eval failed: oracle=$s0 baseline=$s1"

for route in baseline oracle; do
    log="$LOGS/validate_${route}.log"
    best="$(grep 'TEST Iter' "$log" | sed -E 's/.*Top-1 err = ([0-9.]+).*/\1/' | \
        awk 'BEGIN{b=-1}{a=100-$1;if(a>b)b=a}END{if(b>=0)printf "%.2f",b}')"
    echo "$route best Top1: ${best:-N/A}"
    grep 'TEST Iter' "$log" | tail -n 1 || true
done

echo "Decision rule: Oracle must beat baseline under the identical 500-image budget."
