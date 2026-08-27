#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 45 or 46}"
[[ "$TEACHER_SEED" == 45 || "$TEACHER_SEED" == 46 ]] || {
    echo "This extension is reserved for Teacher seeds 45/46" >&2
    exit 1
}
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
RSEEDS=(41 42)
SSEEDS=(42 43)

MASTER_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
TEACHER_ROOT="$MASTER_ROOT/tseed${TEACHER_SEED}"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
SWEEP_ROOT="$Main_Data_Path/class_in_class/imagenette_temperature_sweep_ipc10"
REAL_TRAIN="$val_dir/imagenet-nette/train"
VAL_DIR="$val_dir/imagenet-nette/test"
TEACHER_C1="$TEACHER_ROOT/models/random_c1_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
TEACHER_R100="$TEACHER_ROOT/models/random_c100_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
MAP_C1="$TEACHER_ROOT/data/random_c1_pseed42/hierarchy.json"
MAP_R100="$TEACHER_ROOT/data/random_c100_pseed42/hierarchy.json"
FKD_ROOT="$SWEEP_ROOT/tseed${TEACHER_SEED}/fkd"
RESULT_ROOT="$SWEEP_ROOT/tseed${TEACHER_SEED}/per_class"
POST_ROOT="$SWEEP_ROOT/tseed${TEACHER_SEED}/post_eval"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_real_best_fp16/tseed${TEACHER_SEED}}"
mkdir -p "$FKD_ROOT" "$RESULT_ROOT" "$POST_ROOT" "$LOG_ROOT"

fail() { echo "ImageNette Real best-T FP16 extension failed: $*" >&2; exit 1; }
wait_jobs() {
    local status=0 pid
    for pid in "$@"; do wait "$pid" || status=1; done
    return "$status"
}
temp_for() { [[ "$1" == c1 ]] && echo 800 || echo 200; }
heads_for() { [[ "$1" == c1 ]] && echo 10 || echo 1000; }
teacher_for() { [[ "$1" == c1 ]] && echo "$TEACHER_C1" || echo "$TEACHER_R100"; }
mapping_for() { [[ "$1" == c1 ]] && echo "$MAP_C1" || echo "$MAP_R100"; }
source_for() { echo "$FACTORIAL_ROOT/real_sets/tseed${TEACHER_SEED}_rseed$1"; }

echo "[1/4] Train/reuse C1 and Random C100 Teachers for seed=$TEACHER_SEED"
TEACHER_SEED="$TEACHER_SEED" C_VALUES="1 100" PARTITION_SEED=42 \
SOURCE_ROOT="$val_dir/imagenet-nette" SOURCE_VALIDATION_SPLIT=test \
EXP_ROOT="$TEACHER_ROOT" LOGS="$LOG_ROOT/teachers" \
GPU0="$GPU0" GPU1="$GPU1" WORKERS="$WORKERS" PARALLEL_JOBS=2 \
bash "$ROOT/run_imagenette_cic_t_teachers_2gpu.sh" || fail teacher_training
[[ -f "$TEACHER_C1" && -f "$TEACHER_R100" && -f "$MAP_C1" && -f "$MAP_R100" ]] \
    || fail "Teacher assets missing after training"

echo "[2/4] Prepare independently sampled IPC10 Real sources"
for rseed in "${RSEEDS[@]}"; do
    subset="$(source_for "$rseed")"
    subset_seed=$((42000000 + TEACHER_SEED * 100003 + rseed * 1009))
    python "$ROOT/class_in_class/prepare_imagenette_random_real_ipc10.py" \
        --source-train "$REAL_TRAIN" --output-dir "$subset" \
        --seed "$subset_seed" --images-per-class 10 --repair-invalid-output \
        > "$LOG_ROOT/prepare_real_r${rseed}.log" 2>&1 || fail "real subset r=$rseed"
    images="$(find "$subset" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    classes="$(find "$subset" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "$images" == 100 && "$classes" == 10 ]] || fail "invalid real subset r=$rseed"
done
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "validation split is not full ImageNette test"

assert_fp16_fkd() {
    local fkd="$1"
    python - "$fkd" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1]) / "epoch_0" / "batch_0.tar"
dtype = torch.load(path, map_location="cpu", weights_only=False)[5].dtype
if dtype != torch.float16:
    raise SystemExit(f"expected native FKD torch.float16 logits, got {dtype}: {path}")
PY
}

relabel_one() {
    local col="$1" rseed="$2" gpu="$3"
    local temp heads teacher mapping source tag base final count
    temp="$(temp_for "$col")"; heads="$(heads_for "$col")"
    teacher="$(teacher_for "$col")"; mapping="$(mapping_for "$col")"
    source="$(source_for "$rseed")"; tag="${col}_T${temp}"
    base="$FKD_ROOT/real__${tag}_rseed${rseed}"; final="${base}_bs10_ipc10"
    count=0; [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    if (( count == 3000 )); then assert_fp16_fkd "$final"; return; fi
    (( count > 0 )) && mv "$final" "${final}.partial_$(date +%Y%m%d_%H%M%S)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$source" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 \
        --teacher-num-classes "$heads" --teacher-mapping "$mapping" \
        --marginalize-temperature "$temp" --gpu 0 --batch-size 10 --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette \
        --epochs 300 --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_real__${tag}_r${rseed}.log" 2>&1 || return 1
    [[ "$(find "$final" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] || return 1
    assert_fp16_fkd "$final"
}

echo "[3/4] FP16 relabel at C1 T800 and Random C100 T200"
pids=(); task=0
for rseed in "${RSEEDS[@]}"; do
    for col in c1 random100; do
        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
        relabel_one "$col" "$rseed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == 2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
    done
done

post_one() {
    local col="$1" rseed="$2" sseed="$3" gpu="$4"
    local temp source tag fkd result
    temp="$(temp_for "$col")"; source="$(source_for "$rseed")"; tag="${col}_T${temp}"
    fkd="$FKD_ROOT/real__${tag}_rseed${rseed}_bs10_ipc10"
    result="$RESULT_ROOT/real__${tag}_rseed${rseed}_sseed${sseed}.json"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='fkd_soft_label')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    assert_fp16_fkd "$fkd"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" \
        --fkd-path "$fkd" --mix-type cutmix --temperature "$temp" \
        --model ResNet18 --ipc 10 \
        --exp-name "real_best_fp16_${tag}_t${TEACHER_SEED}_r${rseed}_s${sseed}" \
        --original-data-path "$source" --output-dir "$POST_ROOT" --batch-size 10 \
        --epochs 300 --dataset-name imagenet-nette --gradient-accumulation-steps 2 \
        --cos --workers "$WORKERS" --fkd_seed 42 --adamw-weight-decay 0.01 \
        --adamw-lr-override 0.0005 --eta-override 1 --train-seed "$sseed" \
        --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" \
        > "$LOG_ROOT/post_real__${tag}_r${rseed}_s${sseed}.log" 2>&1
}

echo "[4/4] Post-eval: two processes per A100"
pids=(); task=0
for rseed in "${RSEEDS[@]}"; do
    for col in c1 random100; do
        for sseed in "${SSEEDS[@]}"; do
            gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
            post_one "$col" "$rseed" "$sseed" "$gpu" & pids+=("$!")
            if (( ${#pids[@]} == 4 )); then wait_jobs "${pids[@]}" || fail post; pids=(); fi
        done
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post
echo "Real best-T native-FP16 extension complete: Teacher seed=$TEACHER_SEED"
