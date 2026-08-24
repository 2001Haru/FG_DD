#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED}"
C_VALUES_TEXT="${C_VALUES:?set C_VALUES}"
read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
CLUSTER_SEED="${CLUSTER_SEED:-42}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-300}"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
DATA_ROOT="$MASTER_ROOT/data"
TEACHER_ROOT="$MASTER_ROOT/tseed${TEACHER_SEED}"
MODEL_ROOT="$TEACHER_ROOT/models"
AUDIT_ROOT="$TEACHER_ROOT/audits"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42/tseed${TEACHER_SEED}/teachers}"
mkdir -p "$MODEL_ROOT" "$AUDIT_ROOT" "$LOG_ROOT"

fail(){ echo "DINO cluster Teacher stage failed: $*" >&2; exit 1; }

partition_for_c(){
    printf '%s\n' "$DATA_ROOT/dinov2_cluster_c${1}_seed${CLUSTER_SEED}"
}

model_for_c(){
    printf '%s\n' "$MODEL_ROOT/dinov2_cluster_c${1}_seed${CLUSTER_SEED}_tseed${TEACHER_SEED}"
}

for c in "${C_VALUES_ARRAY[@]}"; do
    data="$(partition_for_c "$c")"
    [[ -f "$data/hierarchy.json" ]] || fail "missing clustered hierarchy: $data/hierarchy.json"
    counts="$(python -c "import json; q=json.load(open('$data/hierarchy.json')); print(q.get('kind'),q.get('source_train_images'),q.get('source_val_images'),q.get('source_validation_split'),q.get('subclasses_per_coarse'))")"
    [[ "$counts" == "imagenette_balanced_dinov2_clusters 9469 3925 test $c" ]] \
        || fail "invalid C=$c clustered partition: $counts"
done

train_one(){
    local c="$1" classes=$((10*c))
    local data="$(partition_for_c "$c")"
    local model_dir="$(model_for_c "$c")"
    local marker="$model_dir/.training_complete.json"
    local checkpoint="$model_dir/ResNet18.pth"
    local manifest_hash marker_valid
    manifest_hash="$(sha256sum "$data/hierarchy.json" | awk '{print $1}')"

    if [[ -f "$checkpoint" && -f "$marker" ]]; then
        marker_valid="$(python -c "import json; q=json.load(open('$marker')); print(int(q.get('epochs',-1))==$TEACHER_EPOCHS and int(q.get('classes',-1))==$classes and int(q.get('seed',-1))==$TEACHER_SEED and q.get('data_manifest_sha256')=='$manifest_hash' and bool(q.get('validation_enabled',False)))")"
        if [[ "$marker_valid" == "True" ]]; then
            echo "Reusing completed Teacher: tseed=$TEACHER_SEED C=$c"
            return
        fi
    fi

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/train_imagenette_subclass_teacher.py" \
        --data-dir "$data" --output-dir "$model_dir" --classes "$classes" \
        --batch-size 64 --epochs "$TEACHER_EPOCHS" --workers "$WORKERS" \
        --seed "$TEACHER_SEED" \
        > "$LOG_ROOT/train_c${c}.log" 2>&1

    [[ -f "$checkpoint" && -f "$marker" ]] \
        || { echo "C=$c training ended without checkpoint/marker" >&2; return 1; }
    marker_valid="$(python -c "import json; q=json.load(open('$marker')); print(int(q.get('epochs',-1))==$TEACHER_EPOCHS and int(q.get('classes',-1))==$classes and int(q.get('seed',-1))==$TEACHER_SEED and q.get('data_manifest_sha256')=='$manifest_hash' and bool(q.get('validation_enabled',False)))")"
    [[ "$marker_valid" == "True" ]] \
        || { echo "C=$c completion marker failed strict validation" >&2; return 1; }
}

audit_one(){
    local c="$1"
    local data="$(partition_for_c "$c")"
    local model_dir="$(model_for_c "$c")"
    local checkpoint="$model_dir/ResNet18.pth"
    local marker="$model_dir/.training_complete.json"
    local output="$AUDIT_ROOT/dinov2_cluster_c${c}_teacher_audit.json"

    # Critical ordering guard: C500 audit must never start before Teacher completion.
    [[ -f "$checkpoint" && -f "$marker" ]] || {
        echo "C=$c audit blocked: Teacher checkpoint/marker missing" >&2
        return 1
    }
    if [[ -f "$output" ]]; then
        valid="$(python -c "import json; q=json.load(open('$output')); print(q.get('audit_scope')=='train_and_validation' and int(q.get('subclasses_per_coarse',-1))==$c and int(q.get('train',{}).get('images',-1))==9469 and int(q.get('val',{}).get('images',-1))==3925)")"
        [[ "$valid" == "True" ]] && { echo "Reusing Teacher audit: tseed=$TEACHER_SEED C=$c"; return; }
    fi
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_subclass_teacher.py" \
        --data-dir "$data" --checkpoint "$checkpoint" \
        --mapping "$data/hierarchy.json" --output "$output" --workers "$WORKERS" \
        > "$LOG_ROOT/audit_c${c}.log" 2>&1
    [[ -f "$output" ]] || { echo "C=$c audit produced no JSON" >&2; return 1; }
    valid="$(python -c "import json; q=json.load(open('$output')); print(q.get('audit_scope')=='train_and_validation' and int(q.get('subclasses_per_coarse',-1))==$c and int(q.get('train',{}).get('images',-1))==9469 and int(q.get('val',{}).get('images',-1))==3925)")"
    [[ "$valid" == "True" ]] || { echo "C=$c audit JSON failed validation" >&2; return 1; }
}

echo "Teacher seed=$TEACHER_SEED on GPU$GPU; sequential C=${C_VALUES_ARRAY[*]}"
for c in "${C_VALUES_ARRAY[@]}"; do
    echo "[Teacher] tseed=$TEACHER_SEED C=$c"
    train_one "$c" || fail "Teacher C=$c"
    echo "[Audit] tseed=$TEACHER_SEED C=$c"
    audit_one "$c" || fail "audit C=$c"
done

python "$ROOT/class_in_class/summarize_imagenette_teacher_audits.py" \
    --audit-dir "$AUDIT_ROOT" --c-values "${C_VALUES_ARRAY[@]}" \
    --file-template 'dinov2_cluster_c{c}_teacher_audit.json' \
    --partition-description 'balanced within-parent DINOv2 spherical clusters; test assigned to nearest train centroid' \
    --output "$AUDIT_ROOT/summary_c${C_VALUES_TEXT// /_}.json" \
    > "$LOG_ROOT/summary.log" 2>&1
echo "Teacher/audit complete: tseed=$TEACHER_SEED C=${C_VALUES_ARRAY[*]}"
