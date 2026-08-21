#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
SOURCE_ROOT="${SOURCE_ROOT:-/linxi/dataset/VLCP/ImageNette}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t}"
PARTITION_SEED="${PARTITION_SEED:-42}"; TEACHER_SEED="${TEACHER_SEED:-42}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-300}"
DATA_ROOT="$EXP_ROOT/data"; MODEL_ROOT="$EXP_ROOT/models"; AUDIT_ROOT="$EXP_ROOT/audits"
LOGS="$ROOT/logs/imagenette_cic_t/teachers"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$AUDIT_ROOT" "$LOGS"
fail(){ echo "ImageNette CiC-T Teacher stage failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=$?; done; return "$status"; }

for split in train val; do
    [[ -d "$SOURCE_ROOT/$split" ]] || fail "missing source split: $SOURCE_ROOT/$split"
    classes="$(find "$SOURCE_ROOT/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "$classes" == 10 ]] || fail "$split contains $classes class directories, expected 10"
done

echo "[1/3] Preparing random subclass ImageFolders"
for c in 1 2 5 10; do
    output="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    python "$ROOT/class_in_class/prepare_imagenette_random_subclasses.py" \
        --source-root "$SOURCE_ROOT" --output-dir "$output" \
        --subclasses "$c" --seed "$PARTITION_SEED" --repair-invalid-output \
        > "$LOGS/partition_c${c}.log" 2>&1
done

train_one(){
    local gpu="$1"
    local c="$2"
    local classes=$((10*c))
    local data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    local model_dir="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    manifest_hash="$(sha256sum "$data/hierarchy.json" | awk '{print $1}')"
    if [[ -f "$model_dir/.training_complete.json" ]]; then
        marker_valid="$(python -c "import json; q=json.load(open('$model_dir/.training_complete.json')); print(int(q.get('epochs',-1))==$TEACHER_EPOCHS and int(q.get('classes',-1))==$classes and int(q.get('seed',-1))==$TEACHER_SEED and q.get('data_manifest_sha256')=='$manifest_hash')")"
        [[ "$marker_valid" == "True" ]] && return
    fi
    if [[ -f "$model_dir/ResNet18.pth" && -f "$model_dir/training_history.json" ]]; then
        completed_epochs="$(python -c "import json; print(len(json.load(open('$model_dir/training_history.json'))))")"
        if [[ "$completed_epochs" == "$TEACHER_EPOCHS" ]]; then
            python -c "import json; json.dump({'epochs': $TEACHER_EPOCHS, 'classes': $classes, 'seed': $TEACHER_SEED, 'checkpoint': 'ResNet18.pth', 'data_manifest_sha256': '$manifest_hash'}, open('$model_dir/.training_complete.json','w'), indent=2)"
            return
        fi
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/train_imagenette_subclass_teacher.py" \
        --data-dir "$data" --output-dir "$model_dir" --classes "$classes" \
        --batch-size 64 --epochs "$TEACHER_EPOCHS" --workers "$WORKERS" --seed "$TEACHER_SEED" \
        > "$LOGS/train_c${c}.log" 2>&1
}

echo "[2/3] Training C=2/5, then C=1/10 Teachers"
pids=()
for c in 2 5; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    train_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail teacher_training; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail teacher_training; fi
pids=()
train_one "$GPU0" 1 & pids+=("$!")
train_one "$GPU1" 10 & pids+=("$!")
wait_jobs "${pids[@]}" || fail teacher_training

audit_one(){
    local gpu="$1" c="$2"
    local data="$DATA_ROOT/random_c${c}_pseed${PARTITION_SEED}"
    local model_dir="$MODEL_ROOT/random_c${c}_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}"
    local output="$AUDIT_ROOT/random_c${c}_teacher_audit.json"
    [[ -f "$output" ]] && return
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/class_in_class/audit_imagenette_subclass_teacher.py" \
        --data-dir "$data" --checkpoint "$model_dir/ResNet18.pth" \
        --mapping "$data/hierarchy.json" --output "$output" --workers "$WORKERS" \
        > "$LOGS/audit_c${c}.log" 2>&1
}

echo "[3/3] Auditing memorization and hierarchy collapse"
pids=()
for c in 1 2 5 10; do
    gpu="$GPU0"; (( ${#pids[@]}==1 )) && gpu="$GPU1"
    audit_one "$gpu" "$c" & pids+=("$!")
    if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail teacher_audit; pids=(); fi
done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail teacher_audit; fi

python "$ROOT/class_in_class/summarize_imagenette_teacher_audits.py" \
    --audit-dir "$AUDIT_ROOT" --output "$AUDIT_ROOT/summary.json"
echo "Teacher-only stage complete: $AUDIT_ROOT/summary.json"
