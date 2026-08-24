#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
[[ "$PARALLEL_JOBS" == 1 || "$PARALLEL_JOBS" == 2 ]] || {
    echo "PARALLEL_JOBS must be 1 or 2" >&2; exit 1;
}
SOURCE_ROOT="${SOURCE_ROOT:-$val_dir/imagenet-nette}"
SOURCE_VALIDATION_SPLIT="${SOURCE_VALIDATION_SPLIT:-test}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
PARTITION_SEED="${PARTITION_SEED:-42}"; TEACHER_SEED="${TEACHER_SEED:-42}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-300}"
C_VALUES_TEXT="${C_VALUES:-1 2 5 10}"; read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
DATA_ROOT="$EXP_ROOT/data"; MODEL_ROOT="$EXP_ROOT/models"; AUDIT_ROOT="$EXP_ROOT/audits"
LOGS="${LOGS:-$ROOT/logs/imagenette_cic_t_official_split/teachers}"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$AUDIT_ROOT" "$LOGS"
fail(){ echo "ImageNette CiC-T Teacher stage failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

for split in train "$SOURCE_VALIDATION_SPLIT"; do
    [[ -d "$SOURCE_ROOT/$split" ]] || fail "missing source split: $SOURCE_ROOT/$split"
    classes="$(find "$SOURCE_ROOT/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "$classes" == 10 ]] || fail "$split contains $classes class directories, expected 10"
done
MIN_VALIDATION_PARENT_IMAGES="$(find "$SOURCE_ROOT/$SOURCE_VALIDATION_SPLIT" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' directory; do find "$directory" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l; done | sort -n | head -n1)"
[[ "$MIN_VALIDATION_PARENT_IMAGES" =~ ^[0-9]+$ ]] || fail "cannot determine minimum validation parent size"
echo "Teacher pseudo-validation is defined only for C <= $MIN_VALIDATION_PARENT_IMAGES"

echo "[1/3] Preparing random subclass ImageFolders"
for c in "${C_VALUES_ARRAY[@]}"; do
    output="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    python "$ROOT/class_in_class/prepare_imagenette_random_subclasses.py" \
        --source-root "$SOURCE_ROOT" --output-dir "$output" \
        --source-validation-split "$SOURCE_VALIDATION_SPLIT" \
        --subclasses "$c" --seed "$PARTITION_SEED" --repair-invalid-output \
        > "$LOGS/partition_c${c}.log" 2>&1
    counts="$(python -c "import json; q=json.load(open('$output/hierarchy.json')); print(q['source_train_images'], q['source_val_images'], q.get('source_validation_split'))")"
    [[ "$counts" == "9469 3925 test" ]] || fail "C=$c partition has unsafe source counts/split: $counts"
done

train_one(){
    local gpu="$1"
    local c="$2"
    local classes=$((10*c))
    local data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    local model_dir="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    local validation_enabled=True
    local validation_args=()
    if (( c > MIN_VALIDATION_PARENT_IMAGES )); then
        validation_enabled=False
        validation_args+=(--skip-validation)
    fi
    manifest_hash="$(sha256sum "$data/hierarchy.json" | awk '{print $1}')"
    if [[ -f "$model_dir/.training_complete.json" && -f "$model_dir/ResNet18.pth" ]]; then
        marker_valid="$(python -c "import json; q=json.load(open('$model_dir/.training_complete.json')); print(int(q.get('epochs',-1))==$TEACHER_EPOCHS and int(q.get('classes',-1))==$classes and int(q.get('seed',-1))==$TEACHER_SEED and q.get('data_manifest_sha256')=='$manifest_hash' and bool(q.get('validation_enabled',True)) is $validation_enabled)")"
        [[ "$marker_valid" == "True" ]] && return
    fi
    if [[ -f "$model_dir/ResNet18.pth" && -f "$model_dir/training_history.json" ]]; then
        completed_epochs="$(python -c "import json; print(len(json.load(open('$model_dir/training_history.json'))))")"
        if [[ "$completed_epochs" == "$TEACHER_EPOCHS" ]]; then
            python -c "import json; json.dump({'epochs': $TEACHER_EPOCHS, 'classes': $classes, 'seed': $TEACHER_SEED, 'checkpoint': 'ResNet18.pth', 'data_manifest_sha256': '$manifest_hash', 'validation_enabled': $validation_enabled}, open('$model_dir/.training_complete.json','w'), indent=2)"
            return
        fi
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/train_imagenette_subclass_teacher.py" \
        --data-dir "$data" --output-dir "$model_dir" --classes "$classes" \
        --batch-size 64 --epochs "$TEACHER_EPOCHS" --workers "$WORKERS" --seed "$TEACHER_SEED" \
        "${validation_args[@]}" \
        > "$LOGS/train_c${c}.log" 2>&1
    [[ -f "$model_dir/ResNet18.pth" && -f "$model_dir/.training_complete.json" ]] || {
        echo "C=$c Teacher process ended without checkpoint/completion marker: $model_dir" >&2
        return 1
    }
}

echo "[2/3] Training Teachers: C=${C_VALUES_ARRAY[*]}"
pids=()
for c in "${C_VALUES_ARRAY[@]}"; do
    gpu="$GPU0"; (( PARALLEL_JOBS == 2 && ${#pids[@]}==1 )) && gpu="$GPU1"
    train_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==PARALLEL_JOBS )); then wait_jobs "${pids[@]}" || fail teacher_training; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail teacher_training; fi
for c in "${C_VALUES_ARRAY[@]}"; do
    model_dir="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    [[ -f "$model_dir/ResNet18.pth" && -f "$model_dir/.training_complete.json" ]] \
        || fail "C=$c Teacher missing after training stage: $model_dir"
done

audit_one(){
    local gpu="$1" c="$2"
    local data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    local model_dir="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    local output="$AUDIT_ROOT/random_c${c}_teacher_audit.json"
    [[ -f "$model_dir/ResNet18.pth" ]] || {
        echo "C=$c audit blocked: missing checkpoint $model_dir/ResNet18.pth" >&2
        return 1
    }
    if [[ -f "$output" ]]; then
        schema="$(python -c "import json; print(json.load(open('$output')).get('audit_schema_version', 0))")"
        [[ "$schema" == "2" ]] && return
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/class_in_class/audit_imagenette_subclass_teacher.py" \
        --data-dir "$data" --checkpoint "$model_dir/ResNet18.pth" \
        --mapping "$data/hierarchy.json" --output "$output" --workers "$WORKERS" \
        > "$LOGS/audit_c${c}.log" 2>&1
}

echo "[3/3] Auditing memorization and hierarchy collapse"
pids=()
for c in "${C_VALUES_ARRAY[@]}"; do
    if (( c > MIN_VALIDATION_PARENT_IMAGES )); then
        echo "Skipping pseudo-label Teacher audit for C=$c (> $MIN_VALIDATION_PARENT_IMAGES validation images in the smallest parent)"
        continue
    fi
    gpu="$GPU0"; (( PARALLEL_JOBS == 2 && ${#pids[@]}==1 )) && gpu="$GPU1"
    audit_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==PARALLEL_JOBS )); then wait_jobs "${pids[@]}" || fail teacher_audit; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail teacher_audit; fi

all_ready=1
for c in 1 2 5 10; do
    [[ -f "$AUDIT_ROOT/random_c${c}_teacher_audit.json" ]] || all_ready=0
done
if (( all_ready )); then
    python "$ROOT/class_in_class/summarize_imagenette_teacher_audits.py" \
        --audit-dir "$AUDIT_ROOT" --output "$AUDIT_ROOT/summary.json"
    echo "Teacher-only stage complete: $AUDIT_ROOT/summary.json"
else
    echo "Selected Teacher/audit stage complete for C=${C_VALUES_ARRAY[*]}; full summary waits for C=1,2,5,10"
fi
