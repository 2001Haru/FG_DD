#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
C_VALUES_TEXT="${C_VALUES:?set C_VALUES}"
read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
CLUSTER_SEED="${CLUSTER_SEED:-42}"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
MASTER_LOGS="${MASTER_LOGS:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42}"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"
read -r -a RSEEDS <<< "$RSEEDS_TEXT"
read -r -a SSEEDS <<< "$SSEEDS_TEXT"
mkdir -p "$MASTER_LOGS"

fail(){ echo "DINO cluster full group failed: $*" >&2; exit 1; }
wait_pair(){
    local first="$1" second="$2" status=0
    wait "$first" || status=1
    wait "$second" || status=1
    return "$status"
}

teacher_stream(){
    local teacher_seed="$1" gpu="$2"
    TEACHER_SEED="$teacher_seed" GPU="$gpu" WORKERS="$WORKERS" \
    C_VALUES="$C_VALUES_TEXT" CLUSTER_SEED="$CLUSTER_SEED" \
    MASTER_ROOT="$MASTER_ROOT" \
    LOG_ROOT="$MASTER_LOGS/tseed${teacher_seed}/teachers_c${C_VALUES_TEXT// /_}" \
    bash "$ROOT/run_imagenette_dinov2_cluster_teachers.sh"
}

echo "[A/3] Teacher+audit streams: seed43 GPU$GPU0; seed44 GPU$GPU1; C=${C_VALUES_ARRAY[*]}"
teacher_stream 43 "$GPU0" & teacher43=$!
teacher_stream 44 "$GPU1" & teacher44=$!
wait_pair "$teacher43" "$teacher44" || fail "Teacher/audit stage"

# Global barrier and strict C500-safe preflight before any downstream stage.
for teacher_seed in 43 44; do
    for c in "${C_VALUES_ARRAY[@]}"; do
        model="$MASTER_ROOT/tseed${teacher_seed}/models/dinov2_cluster_c${c}_seed${CLUSTER_SEED}_tseed${teacher_seed}"
        audit="$MASTER_ROOT/tseed${teacher_seed}/audits/dinov2_cluster_c${c}_teacher_audit.json"
        [[ -f "$model/ResNet18.pth" && -f "$model/.training_complete.json" ]] \
            || fail "missing completed Teacher after barrier: tseed=$teacher_seed C=$c"
        hierarchy="$MASTER_ROOT/data/dinov2_cluster_c${c}_seed${CLUSTER_SEED}/hierarchy.json"
        hierarchy_hash="$(sha256sum "$hierarchy" | awk '{print $1}')"
        marker_valid="$(python -c "import json; q=json.load(open('$model/.training_complete.json')); print(int(q.get('classes',-1))==10*$c and int(q.get('seed',-1))==$teacher_seed and q.get('data_manifest_sha256')=='$hierarchy_hash' and bool(q.get('validation_enabled',False)))")"
        [[ "$marker_valid" == "True" ]] \
            || fail "Teacher marker/hierarchy provenance mismatch: tseed=$teacher_seed C=$c"
        [[ -f "$audit" ]] || fail "missing Teacher audit after barrier: tseed=$teacher_seed C=$c"
        valid="$(python -c "import json; q=json.load(open('$audit')); print(q.get('audit_scope')=='train_and_validation' and int(q.get('subclasses_per_coarse',-1))==$c and int(q.get('train',{}).get('images',-1))==9469 and int(q.get('val',{}).get('images',-1))==3925)")"
        [[ "$valid" == "True" ]] || fail "invalid audit after barrier: tseed=$teacher_seed C=$c"
    done
done

full_stream(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    TEACHER_SEED="$teacher_seed" C_VALUES="$C_VALUES_TEXT" \
    RECOVERY_SEEDS="$RSEEDS_TEXT" STUDENT_SEEDS="$SSEEDS_TEXT" \
    PARTITION_SEED="$CLUSTER_SEED" PARTITION_PREFIX=dinov2_cluster PARTITION_SEED_TOKEN=seed \
    EXPECTED_PARTITION_KIND=imagenette_balanced_dinov2_clusters \
    DATA_ROOT_OVERRIDE="$MASTER_ROOT/data" MODEL_ROOT_OVERRIDE="$teacher_root/models" \
    EXP_ROOT="$teacher_root" REAL_ROOT="$val_dir/imagenet-nette" \
    VAL_DIR="$val_dir/imagenet-nette/test" VIEW_SEED=42 TEMPERATURE=20 \
    RECOVERY_ITERATIONS=4000 RECOVERY_LR=0.1 RECOVERY_R_BN=0.01 \
    GPU0="$gpu" GPU1="$gpu" PARALLEL_JOBS=1 WORKERS="$WORKERS" \
    LOGS="$MASTER_LOGS/tseed${teacher_seed}/full_c${C_VALUES_TEXT// /_}" \
    bash "$ROOT/run_imagenette_cic_t_full_2gpu.sh"
}

echo "[B/3] Recovery/Relabel/Post-eval: seed43 GPU$GPU0; seed44 GPU$GPU1"
full_stream 43 "$GPU0" & full43=$!
full_stream 44 "$GPU1" & full44=$!
wait_pair "$full43" "$full44" || fail "downstream full stage"

echo "[C/3] Cross-Teacher summary"
c_tag="${C_VALUES_TEXT// /_}"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds 43 44 \
    --recovery-seeds "${RSEEDS[@]}" --student-seeds "${SSEEDS[@]}" \
    --c-values "${C_VALUES_ARRAY[@]}" \
    --protocol "ImageNette IPC10 ResNet18 DINOv2 clustered CiC-T, official split, balanced within-parent clusters, Teacher seeds43/44, recovery iter4000 LR0.1 r_bn0.01, marg10 T20" \
    --output "$MASTER_ROOT/analysis/dinov2_cluster_summary_c${c_tag}.json" \
    > "$MASTER_LOGS/summary_c${c_tag}.log" 2>&1
echo "Complete: $MASTER_ROOT/analysis/dinov2_cluster_summary_c${c_tag}.json"
