#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RECOVERY_SEEDS="${RECOVERY_SEEDS:-41 42 43}"; read -r -a SEEDS <<< "$RECOVERY_SEEDS"
(( ${#SEEDS[@]} >= 3 )) || { echo "At least three recovery seeds are required" >&2; exit 1; }
RANDOM_PARTITION_SEED="${RANDOM_PARTITION_SEED:-42}"
RANDOM_TEACHER_AUDIT_ONLY="${RANDOM_TEACHER_AUDIT_ONLY:-0}"
ITERATIONS="${ITERATIONS:-4000}"; CALIBRATION_ONLY="${CALIBRATION_ONLY:-0}"
CALIBRATION_ITERATIONS="${CALIBRATION_ITERATIONS:-400}"
BASE_RECOVERY_LR="${BASE_RECOVERY_LR:-0.25}"; ORACLE_RECOVERY_LR="${ORACLE_RECOVERY_LR:-0.25}"
BASE_R_BN="${BASE_R_BN:-0.01}"; ORACLE_R_BN="${ORACLE_R_BN:-0.01}"
RANDOM_RECOVERY_LR="${RANDOM_RECOVERY_LR:-0.25}"; RANDOM_R_BN="${RANDOM_R_BN:-0.01}"
RAW_ARCHIVE="${RAW_ARCHIVE:-/linxi/dataset/CV-DD/raw/cifar100/cifar-100-python.tar.gz}"
RAW_DIR="${RAW_DIR:-/linxi/dataset/CV-DD/raw/cifar100/cifar-100-python}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/cifar100_v2_bs100}"
[[ "$CALIBRATION_ONLY" == "1" ]] && ITERATIONS="$CALIBRATION_ITERATIONS"

DATA="$EXP_ROOT/data"; MODELS="$EXP_ROOT/models"; PATCHES="$EXP_ROOT/patches"; PLANS="$EXP_ROOT/recovery_plans"
SYN_STAGE=synthetic; [[ "$CALIBRATION_ONLY" == "1" ]] && SYN_STAGE=calibration_synthetic
SYN_PARENT="$EXP_ROOT/$SYN_STAGE"; FKD_PARENT="$EXP_ROOT/fkd"; OUTPUT="$EXP_ROOT/post_eval"
LOGS_ROOT="$ROOT/logs/cifar100_class_in_class_v2_bs100"
if [[ "$CALIBRATION_ONLY" == "1" ]]; then
    LOGS="$LOGS_ROOT/calibration_${ITERATIONS}"
else
    LOGS="$LOGS_ROOT/formal_${ITERATIONS}"
fi
FINE_DATA="$DATA/fine"; COARSE_DATA="$DATA/coarse"; MAPPING="$DATA/hierarchy.json"
RANDOM_DATA="$DATA/random100_seed${RANDOM_PARTITION_SEED}"; RANDOM_MAPPING="$RANDOM_DATA/hierarchy.json"
FINE_MODELS="$MODELS/fine100"; COARSE_MODELS="$MODELS/coarse20"
RANDOM_MODELS="$MODELS/random100_seed${RANDOM_PARTITION_SEED}"
FINE_PATCH_ROOT="$PATCHES/fine100"; COARSE_PATCH_ROOT="$PATCHES/coarse20"
RANDOM_PATCH_ROOT="$PATCHES/random100_seed${RANDOM_PARTITION_SEED}"
COARSE_TARGET_PATCH_ROOT="$PATCHES/fine100_marginalized_coarse20"
RANDOM_COARSE_TARGET_PATCH_ROOT="$PATCHES/random100_pseed${RANDOM_PARTITION_SEED}_marginalized_coarse20"
BASE_PLAN="$PLANS/baseline_coarse20_ipc25.json"; ORACLE_PLAN="$PLANS/oracle_fine100_ipc5.json"
RANDOM_PLAN="$PLANS/random_pseudo100_pseed${RANDOM_PARTITION_SEED}_ipc5.json"
mkdir -p "$EXP_ROOT" "$MODELS" "$PATCHES" "$PLANS" "$SYN_PARENT" "$FKD_PARENT" "$OUTPUT" "$LOGS"
fail() { echo "Preflight failed: $*" >&2; exit 1; }
wait_jobs() {
    local status=0 pid
    for pid in "$@"; do wait "$pid" || status=$?; done
    return "$status"
}

if [[ ! -f "$RAW_DIR/train" ]]; then
    [[ -f "$RAW_ARCHIVE" ]] || fail "missing official archive: $RAW_ARCHIVE"
    echo "eb9058c3a382ffc7106e4002c42a8d85  $RAW_ARCHIVE" | md5sum -c - || fail "archive checksum mismatch"
    mkdir -p "$(dirname "$RAW_DIR")"; tar -xzf "$RAW_ARCHIVE" -C "$(dirname "$RAW_DIR")"
fi

cat <<EOF
===== CV-DD SRe2L++ Class-in-Class v2 BS100 =====
Baseline: 5 batches × (20 coarse × 5 images) = 500
Oracle:   5 batches × (100 fine × 1 image) = 500
Both: BS100, 100% label-space coverage/batch, 5 batches, equal steps/image
Recovery seeds: ${SEEDS[*]} (paired); all other seeds fixed42
Recovery baseline: lr=$BASE_RECOVERY_LR r_bn=$BASE_R_BN iterations=$ITERATIONS
Recovery oracle:   lr=$ORACLE_RECOVERY_LR r_bn=$ORACLE_R_BN iterations=$ITERATIONS
Recovery random:   lr=$RANDOM_RECOVERY_LR r_bn=$RANDOM_R_BN iterations=$ITERATIONS; partition_seed=$RANDOM_PARTITION_SEED
Native CV-DD post-eval: R18 random, BS16, 300 epochs, AdamW 1e-3/wd0.01, eta1, T20
Fresh root: $EXP_ROOT; calibration-only=$CALIBRATION_ONLY
==================================================
EOF

if [[ ! -f "$MAPPING" ]]; then
    echo "[1/8] Preparing official hierarchy"
    python "$ROOT/class_in_class/prepare_cifar100_hierarchy.py" --raw-dir "$RAW_DIR" --output-dir "$DATA"
fi
if [[ ! -f "$BASE_PLAN" || ! -f "$ORACLE_PLAN" ]]; then
    python "$ROOT/class_in_class/build_bs100_recovery_plans.py" --mapping "$MAPPING" --output-dir "$PLANS"
fi
if [[ ! -f "$RANDOM_MAPPING" ]]; then
    echo "[1/8] Preparing balanced random pseudo classes (seed=$RANDOM_PARTITION_SEED)"
    python "$ROOT/class_in_class/prepare_random_pseudo_classes.py" --coarse-data "$COARSE_DATA" \
        --output-dir "$RANDOM_DATA" --seed "$RANDOM_PARTITION_SEED"
fi
if [[ ! -f "$RANDOM_PLAN" ]]; then
    python "$ROOT/class_in_class/build_bs100_recovery_plans.py" --mapping "$MAPPING" \
        --random-mapping "$RANDOM_MAPPING" --output-dir "$PLANS"
fi
[[ "$(find "$FINE_DATA/train" -type f -name '*.png' | wc -l)" == 50000 ]] || fail "fine train incomplete"
[[ "$(find "$COARSE_DATA/test" -type f -name '*.png' | wc -l)" == 10000 ]] || fail "coarse test incomplete"
[[ "$(find "$RANDOM_DATA/train" -type f -name '*.png' | wc -l)" == 50000 ]] || fail "random100 train incomplete"
[[ "$(find "$RANDOM_DATA/test" -type f -name '*.png' | wc -l)" == 10000 ]] || fail "random100 test incomplete"

train_teacher() {
    local gpu="$1" dataset="$2" data="$3" output="$4" log="$5"
    [[ -f "$output/ResNet18.pth" ]] && return; mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/squeeze/squeeze.py" --model-list ResNet18 --optimizer Adam --dataset-dir "$data" \
        --save-dir "$output" --batch-size 512 --dataset-name "$dataset" --epoch 200 --lr 0.001 --seed 42 \
        --workers "$WORKERS" --persistent-workers --prefetch-factor 4 > "$log" 2>&1
}
echo "[2/8] Training fresh native CV-DD teachers"
train_teacher "$GPU0" cifar100 "$FINE_DATA" "$FINE_MODELS" "$LOGS/teacher_fine100.log" & p0=$!
train_teacher "$GPU1" cifar20 "$COARSE_DATA" "$COARSE_MODELS" "$LOGS/teacher_coarse20.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?; (( s0==0 && s1==0 )) || fail "teacher failed"
train_teacher "$GPU0" cifar100 "$RANDOM_DATA" "$RANDOM_MODELS" "$LOGS/teacher_random100.log" || fail "random teacher failed"
[[ -f "$RANDOM_MODELS/model_result_info.csv" ]] || fail "random teacher metrics CSV missing"
cp "$RANDOM_MODELS/model_result_info.csv" "$LOGS/teacher_random100_metrics.csv"
CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python "$ROOT/class_in_class/audit_random_teacher_hierarchy.py" --data-dir "$RANDOM_DATA" \
    --checkpoint "$RANDOM_MODELS/ResNet18.pth" --mapping "$RANDOM_MAPPING" --workers "$WORKERS" \
    --output "$LOGS/teacher_random100_hierarchy_audit.json" \
    > "$LOGS/teacher_random100_hierarchy_audit.log" 2>&1
if [[ "$RANDOM_TEACHER_AUDIT_ONLY" == "1" ]]; then
    echo "Random Teacher audit complete; inspect $LOGS/teacher_random100_metrics.csv"
    cat "$LOGS/teacher_random100_metrics.csv"
    exit 0
fi

generate_patches() {
    local gpu="$1" data="$2" teacher="$3" classes="$4" ipc="$5" root="$6" log="$7"
    local teacher_classes="${8:-}" teacher_mapping="${9:-}"
    local extra_args=()
    [[ -n "$teacher_classes" ]] && extra_args+=(--teacher-num-classes "$teacher_classes")
    [[ -n "$teacher_mapping" ]] && extra_args+=(--teacher-mapping "$teacher_mapping")
    local expected=$((classes*ipc)) directory="$root/medium" count=0
    [[ -d "$directory" ]] && count="$(find "$directory" -type f -name '*.jpg' | wc -l)"; (( count == expected )) && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" --data-dir "$data" --teacher "$teacher" \
        --num-classes "$classes" --patches-per-class "$ipc" --candidate-images 100 --crops-per-image 5 \
        --output-dir "$directory" --seed 42 "${extra_args[@]}" > "$log" 2>&1
}
echo "[3/8] Generating fresh balanced patches"
generate_patches "$GPU0" "$FINE_DATA" "$FINE_MODELS/ResNet18.pth" 100 5 "$FINE_PATCH_ROOT" "$LOGS/patches_fine100.log" & p0=$!
generate_patches "$GPU1" "$COARSE_DATA" "$COARSE_MODELS/ResNet18.pth" 20 25 "$COARSE_PATCH_ROOT" "$LOGS/patches_coarse20.log" & p1=$!
s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?; (( s0==0 && s1==0 )) || fail "patch generation failed"
generate_patches "$GPU0" "$RANDOM_DATA" "$RANDOM_MODELS/ResNet18.pth" 100 5 "$RANDOM_PATCH_ROOT" "$LOGS/patches_random100.log" || fail "random patch generation failed"
generate_patches "$GPU0" "$COARSE_DATA" "$FINE_MODELS/ResNet18.pth" 20 25 \
    "$COARSE_TARGET_PATCH_ROOT" "$LOGS/patches_fine100_coarse_target.log" 100 "$MAPPING" \
    || fail "Fine100 marginalized coarse-target patch generation failed"
generate_patches "$GPU0" "$COARSE_DATA" "$RANDOM_MODELS/ResNet18.pth" 20 25 \
    "$RANDOM_COARSE_TARGET_PATCH_ROOT" \
    "$LOGS/patches_random100_pseed${RANDOM_PARTITION_SEED}_coarse_target.log" \
    100 "$RANDOM_MAPPING" || fail "Random100 marginalized coarse-target patch generation failed"

recover_arm() {
    local gpu="$1" plan="$2" teacher="$3" patches="$4" output="$5" seed="$6" lr="$7" rbn="$8" log="$9"
    local teacher_classes="${10:-}" teacher_mapping="${11:-}"
    local extra_args=() protocol_suffix=""
    [[ -n "$teacher_classes" ]] && extra_args+=(--teacher-num-classes "$teacher_classes")
    if [[ -n "$teacher_mapping" ]]; then
        extra_args+=(--teacher-mapping "$teacher_mapping")
        protocol_suffix=":teacher${teacher_classes}:mapping$(sha256sum "$teacher_mapping" | awk '{print $1}')"
    fi
    local plan_hash marker expected; plan_hash="$(sha256sum "$plan" | awk '{print $1}')"; marker="$output/.recovery_protocol"
    expected="$plan_hash:$seed:$lr:$rbn:$ITERATIONS$protocol_suffix"
    if [[ -d "$output" ]] && find "$output" -type f -name '*.jpg' -print -quit | grep -q .; then
        [[ -f "$marker" ]] || fail "old recovery without marker: $output"
        [[ "$(tr -d '[:space:]' < "$marker")" == "$expected" ]] || fail "protocol mismatch; archive $output"
    else mkdir -p "$output"; printf '%s\n' "$expected" > "$marker"; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/recover_from_plan.py" --plan "$plan" --teacher "$teacher" \
        --patch-root "$patches" --output-dir "$output" --iterations "$ITERATIONS" --lr "$lr" \
        --r-bn "$rbn" --first-bn-multiplier 10 --seed "$seed" \
        --diagnostics-output "$output/recovery_diagnostics.jsonl" "${extra_args[@]}" > "$log" 2>&1
}
echo "[4/8] Recovery: paired seeds, five BS100 batches per arm"
for seed in "${SEEDS[@]}"; do
    seed_root="$SYN_PARENT/seed${seed}"
    recover_arm "$GPU0" "$ORACLE_PLAN" "$FINE_MODELS/ResNet18.pth" "$FINE_PATCH_ROOT" "$seed_root/oracle_fine100_ipc5" "$seed" "$ORACLE_RECOVERY_LR" "$ORACLE_R_BN" "$LOGS/recover_oracle_seed${seed}.log" & p0=$!
    recover_arm "$GPU1" "$BASE_PLAN" "$COARSE_MODELS/ResNet18.pth" "$COARSE_PATCH_ROOT" "$seed_root/baseline_coarse20_ipc25" "$seed" "$BASE_RECOVERY_LR" "$BASE_R_BN" "$LOGS/recover_baseline_seed${seed}.log" & p1=$!
    s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?; (( s0==0 && s1==0 )) || fail "recovery seed $seed failed"
done
random_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_root="$SYN_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_pids[@]} == 1 )) && gpu="$GPU1"
    recover_arm "$gpu" "$RANDOM_PLAN" "$RANDOM_MODELS/ResNet18.pth" "$RANDOM_PATCH_ROOT" \
        "$seed_root/random_pseudo100_pseed${RANDOM_PARTITION_SEED}_ipc5" "$seed" "$RANDOM_RECOVERY_LR" "$RANDOM_R_BN" \
        "$LOGS/recover_random_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" & random_pids+=("$!")
    if (( ${#random_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_pids[@]}" || fail "random recovery batch failed"
        random_pids=()
    fi
done
coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_root="$SYN_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    recover_arm "$gpu" "$BASE_PLAN" "$FINE_MODELS/ResNet18.pth" "$COARSE_TARGET_PATCH_ROOT" \
        "$seed_root/fine100_coarse_target_ipc25" "$seed" "$BASE_RECOVERY_LR" "$BASE_R_BN" \
        "$LOGS/recover_fine100_coarse_target_seed${seed}.log" 100 "$MAPPING" \
        & coarse_target_pids+=("$!")
    if (( ${#coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${coarse_target_pids[@]}" || fail "Fine100 coarse-target recovery batch failed"
        coarse_target_pids=()
    fi
done
random_coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_root="$SYN_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    recover_arm "$gpu" "$BASE_PLAN" "$RANDOM_MODELS/ResNet18.pth" \
        "$RANDOM_COARSE_TARGET_PATCH_ROOT" \
        "$seed_root/random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_ipc25" \
        "$seed" "$BASE_RECOVERY_LR" "$BASE_R_BN" \
        "$LOGS/recover_random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" \
        100 "$RANDOM_MAPPING" & random_coarse_target_pids+=("$!")
    if (( ${#random_coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_coarse_target_pids[@]}" || fail "Random100 coarse-target recovery batch failed"
        random_coarse_target_pids=()
    fi
done
for seed in "${SEEDS[@]}"; do
    seed_root="$SYN_PARENT/seed${seed}"
    python "$ROOT/class_in_class/summarize_recovery_diagnostics.py" \
        --baseline-log "$seed_root/baseline_coarse20_ipc25/recovery_diagnostics.jsonl" \
        --oracle-log "$seed_root/oracle_fine100_ipc5/recovery_diagnostics.jsonl" \
        --random-log "$seed_root/random_pseudo100_pseed${RANDOM_PARTITION_SEED}_ipc5/recovery_diagnostics.jsonl" \
        --coarse-target-log "$seed_root/fine100_coarse_target_ipc25/recovery_diagnostics.jsonl" \
        --random-coarse-target-log "$seed_root/random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_ipc25/recovery_diagnostics.jsonl" \
        > "$LOGS/recovery_balance_seed${seed}.txt"
done
{
    for seed in "${SEEDS[@]}"; do
        echo "######## recovery seed $seed ########"
        cat "$LOGS/recovery_balance_seed${seed}.txt"
    done
} > "$LOGS/recovery_balance_all_seeds.txt"
python "$ROOT/class_in_class/analyze_recovery_ce.py" --synthetic-parent "$SYN_PARENT" --seeds "${SEEDS[@]}" \
    --random-partition-seed "$RANDOM_PARTITION_SEED" \
    --output-dir "$LOGS/recovery_ce_analysis" > "$LOGS/recovery_ce_analysis.log" 2>&1
if [[ "$CALIBRATION_ONLY" == "1" ]]; then echo "[5/8] Calibration complete: $LOGS/recovery_balance_all_seeds.txt"; exit 0; fi

echo "[5/8] Merging Oracle and Random seeds into coarse20"
for seed in "${SEEDS[@]}"; do
    source="$SYN_PARENT/seed${seed}/oracle_fine100_ipc5"; merged="$SYN_PARENT/seed${seed}/oracle_merged_coarse20_ipc25"
    count=0; [[ -d "$merged" ]] && count="$(find "$merged" -type f -name '*.jpg' | wc -l)"
    if (( count==0 )); then python "$ROOT/class_in_class/merge_fine_synthetic_to_coarse.py" --fine-dir "$source" --mapping "$MAPPING" --output-dir "$merged" --fine-ipc 5
    elif (( count!=500 )); then fail "partial merged directory: $merged"; fi
    source="$SYN_PARENT/seed${seed}/random_pseudo100_pseed${RANDOM_PARTITION_SEED}_ipc5"; merged="$SYN_PARENT/seed${seed}/random_merged_coarse20_pseed${RANDOM_PARTITION_SEED}_ipc25"
    count=0; [[ -d "$merged" ]] && count="$(find "$merged" -type f -name '*.jpg' | wc -l)"
    if (( count==0 )); then python "$ROOT/class_in_class/merge_fine_synthetic_to_coarse.py" --fine-dir "$source" --mapping "$RANDOM_MAPPING" --output-dir "$merged" --fine-ipc 5
    elif (( count!=500 )); then fail "partial random merged directory: $merged"; fi
done
relabel_arm() {
    local gpu="$1" syn="$2" base="$3" final="$4" log="$5" count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"; (( count==9600 )) && return; (( count==0 )) || fail "partial FKD: $final"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$syn" --fkd-path "$base" --model-pool-dir "$COARSE_MODELS" \
        --teacher-model-name ResNet18 --gpu 0 --batch-size 16 --workers "$WORKERS" --dataset-name cifar20 \
        --epochs 300 --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$log" 2>&1
}
echo "[6/8] Coarse20 BSSL for paired recovery seeds"
for seed in "${SEEDS[@]}"; do
    seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    relabel_arm "$GPU0" "$seed_syn/oracle_merged_coarse20_ipc25" "$seed_fkd/oracle" "$seed_fkd/oracle_bs16_ipc25" "$LOGS/relabel_oracle_seed${seed}.log" & p0=$!
    relabel_arm "$GPU1" "$seed_syn/baseline_coarse20_ipc25" "$seed_fkd/baseline" "$seed_fkd/baseline_bs16_ipc25" "$LOGS/relabel_baseline_seed${seed}.log" & p1=$!
    s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?; (( s0==0 && s1==0 )) || fail "relabel seed $seed failed"
done
random_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_pids[@]} == 1 )) && gpu="$GPU1"
    relabel_arm "$gpu" "$seed_syn/random_merged_coarse20_pseed${RANDOM_PARTITION_SEED}_ipc25" \
        "$seed_fkd/random_pseed${RANDOM_PARTITION_SEED}" \
        "$seed_fkd/random_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25" \
        "$LOGS/relabel_random_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" & random_pids+=("$!")
    if (( ${#random_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_pids[@]}" || fail "random relabel batch failed"
        random_pids=()
    fi
done
coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    relabel_arm "$gpu" "$seed_syn/fine100_coarse_target_ipc25" "$seed_fkd/coarse_target" \
        "$seed_fkd/coarse_target_bs16_ipc25" "$LOGS/relabel_coarse_target_seed${seed}.log" \
        & coarse_target_pids+=("$!")
    if (( ${#coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${coarse_target_pids[@]}" || fail "coarse-target relabel batch failed"
        coarse_target_pids=()
    fi
done
random_coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    relabel_arm "$gpu" \
        "$seed_syn/random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_ipc25" \
        "$seed_fkd/random_coarse_target_pseed${RANDOM_PARTITION_SEED}" \
        "$seed_fkd/random_coarse_target_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25" \
        "$LOGS/relabel_random_coarse_target_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" \
        & random_coarse_target_pids+=("$!")
    if (( ${#random_coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_coarse_target_pids[@]}" || fail "random coarse-target relabel batch failed"
        random_coarse_target_pids=()
    fi
done

validate_arm() {
    local gpu="$1" arm="$2" rseed="$3" syn="$4" fkd="$5" log="$6" per_class="$7"
    [[ -f "$per_class" ]] && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" --model ResNet18 --ipc 25 --exp-name "class_in_class_${arm}_rseed${rseed}" \
        --original-data-path "$syn" --fkd-path "$fkd" --output-dir "$OUTPUT" --batch-size 16 --epochs 300 \
        --dataset-name cifar20 --gradient-accumulation-steps 2 --mix-type cutmix --cos --workers "$WORKERS" \
        --temperature 20 --fkd_seed 42 --adamw-weight-decay 0.01 --train-seed 42 --persistent-workers \
        --val-dir "$COARSE_DATA/test" --disable-wandb --per-class-output "$per_class" > "$log" 2>&1
}
echo "[7/8] Native CV-DD post-eval with fixed student seed42"
PER_CLASS="$EXP_ROOT/per_class"; mkdir -p "$PER_CLASS"
for seed in "${SEEDS[@]}"; do
    seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    validate_arm "$GPU0" oracle "$seed" "$seed_syn/oracle_merged_coarse20_ipc25" "$seed_fkd/oracle_bs16_ipc25" "$LOGS/validate_oracle_seed${seed}.log" "$PER_CLASS/oracle_seed${seed}.json" & p0=$!
    validate_arm "$GPU1" baseline "$seed" "$seed_syn/baseline_coarse20_ipc25" "$seed_fkd/baseline_bs16_ipc25" "$LOGS/validate_baseline_seed${seed}.log" "$PER_CLASS/baseline_seed${seed}.json" & p1=$!
    s0=0; s1=0; wait "$p0" || s0=$?; wait "$p1" || s1=$?; (( s0==0 && s1==0 )) || fail "post-eval seed $seed failed"
done
random_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_pids[@]} == 1 )) && gpu="$GPU1"
    validate_arm "$gpu" "random_pseed${RANDOM_PARTITION_SEED}" "$seed" \
        "$seed_syn/random_merged_coarse20_pseed${RANDOM_PARTITION_SEED}_ipc25" \
        "$seed_fkd/random_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25" \
        "$LOGS/validate_random_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" \
        "$PER_CLASS/random_pseed${RANDOM_PARTITION_SEED}_seed${seed}.json" & random_pids+=("$!")
    if (( ${#random_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_pids[@]}" || fail "random post-eval batch failed"
        random_pids=()
    fi
done
coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    validate_arm "$gpu" coarse_target "$seed" "$seed_syn/fine100_coarse_target_ipc25" \
        "$seed_fkd/coarse_target_bs16_ipc25" "$LOGS/validate_coarse_target_seed${seed}.log" \
        "$PER_CLASS/coarse_target_seed${seed}.json" & coarse_target_pids+=("$!")
    if (( ${#coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${coarse_target_pids[@]}" || fail "coarse-target post-eval batch failed"
        coarse_target_pids=()
    fi
done
random_coarse_target_pids=()
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"; seed_syn="$SYN_PARENT/seed${seed}"; seed_fkd="$FKD_PARENT/seed${seed}"
    gpu="$GPU0"; (( ${#random_coarse_target_pids[@]} == 1 )) && gpu="$GPU1"
    validate_arm "$gpu" "random_coarse_target_pseed${RANDOM_PARTITION_SEED}" "$seed" \
        "$seed_syn/random100_coarse_target_pseed${RANDOM_PARTITION_SEED}_ipc25" \
        "$seed_fkd/random_coarse_target_pseed${RANDOM_PARTITION_SEED}_bs16_ipc25" \
        "$LOGS/validate_random_coarse_target_pseed${RANDOM_PARTITION_SEED}_seed${seed}.log" \
        "$PER_CLASS/random_coarse_target_pseed${RANDOM_PARTITION_SEED}_seed${seed}.json" \
        & random_coarse_target_pids+=("$!")
    if (( ${#random_coarse_target_pids[@]} == 2 || index == ${#SEEDS[@]} - 1 )); then
        wait_jobs "${random_coarse_target_pids[@]}" || fail "random coarse-target post-eval batch failed"
        random_coarse_target_pids=()
    fi
done

echo "[8/8] Superclass distance/gain analysis"
python "$ROOT/class_in_class/analyze_superclass_gain.py" --fine-data "$FINE_DATA" --fine-teacher "$FINE_MODELS/ResNet18.pth" \
    --mapping "$MAPPING" --per-class-dir "$PER_CLASS" --recovery-seeds "${SEEDS[@]}" \
    --random-partition-seed "$RANDOM_PARTITION_SEED" --output-dir "$EXP_ROOT/analysis"
echo "Complete: $EXP_ROOT/analysis and $LOGS"
