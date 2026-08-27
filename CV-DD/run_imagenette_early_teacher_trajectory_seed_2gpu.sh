#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
[[ "$TEACHER_SEED" == 43 || "$TEACHER_SEED" == 44 ]] || {
    echo "Early trajectory protocol uses Teacher seeds43/44" >&2
    exit 1
}
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories/tseed${TEACHER_SEED}"
DATA_ROOT="$EXP_ROOT/data"
MODEL_ROOT="$EXP_ROOT/models"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_trajectories/tseed${TEACHER_SEED}}"
SOURCE_ROOT="$val_dir/imagenet-nette"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$LOG_ROOT"
fail(){ echo "Early Teacher trajectory failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

echo "[1/2] Prepare deterministic pseed42 C1/C100 partitions"
for c in 1 100; do
    partition="$DATA_ROOT/random_c${c}_pseed42"
    python "$ROOT/class_in_class/prepare_imagenette_random_subclasses.py" \
        --source-root "$SOURCE_ROOT" --output-dir "$partition" \
        --source-validation-split test --subclasses "$c" --seed 42 \
        --repair-invalid-output > "$LOG_ROOT/partition_c${c}.log" 2>&1 \
        || fail "partition C=$c"
    counts="$(python -c "import json; q=json.load(open('$partition/hierarchy.json')); print(q['source_train_images'],q['source_val_images'],q['num_pseudo_classes'])")"
    [[ "$counts" == "9469 3925 $((10*c))" ]] || fail "invalid partition C=$c: $counts"
done

train_one(){
    # Do not derive classes in the same `local` declaration as c. Bash expands
    # the arithmetic expression before the new local c is assigned, which can
    # accidentally read the outer partition-loop value (C=100) and launch the
    # C=1 job with 1000 heads.
    local c="$1" gpu="$2"
    local classes=$((10 * c))
    local data="$DATA_ROOT/random_c${c}_pseed42"
    local output="$MODEL_ROOT/c${c}_tseed${TEACHER_SEED}"
    if [[ -d "$output" && ! -f "$output/.training_complete.json" ]]; then
        if find "$output/checkpoints" -type f -name 'epoch_*.pth' -print -quit 2>/dev/null | grep -q .; then
            archive="${output}.partial_$(date +%Y%m%d_%H%M%S)"
            mv "$output" "$archive"
            echo "Archived incomplete deterministic trajectory: $archive"
        fi
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/train_imagenette_early_teacher_trajectory.py" \
        --data-dir "$data" --hierarchy "$data/hierarchy.json" \
        --output-dir "$output" --classes "$classes" --batch-size 64 \
        --epochs 300 --workers "$WORKERS" --seed "$TEACHER_SEED" --temperature 20 \
        > "$LOG_ROOT/train_c${c}.log" 2>&1
}

echo "[2/2] Train C1 and C100 trajectories concurrently"
train_one 1 "$GPU0" & p1=$!
train_one 100 "$GPU1" & p2=$!
wait_jobs "$p1" "$p2" || fail teacher_training
for c in 1 100; do
    output="$MODEL_ROOT/c${c}_tseed${TEACHER_SEED}"
    [[ -f "$output/.training_complete.json" ]] || fail "missing completion marker C=$c"
    [[ "$(find "$output/checkpoints" -type f -name 'epoch_*.pth' | wc -l)" == 300 ]] \
        || fail "C=$c does not contain 300 checkpoints"
    [[ "$(python -c "import json; print(len(json.load(open('$output/metrics.json'))))")" == 300 ]] \
        || fail "C=$c does not contain 300 metric rows"
done
echo "Early Teacher trajectories complete: seed=$TEACHER_SEED"
