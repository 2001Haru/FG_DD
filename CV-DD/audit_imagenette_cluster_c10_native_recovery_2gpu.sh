#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
CLUSTER_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42"
C1_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_cluster_c10_native_recovery_c1_labeler"
OUTPUT_DIR="$EXP_ROOT/analysis/native_recovery_audit"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cluster_c10_native_recovery_audit}"
HIERARCHY="$CLUSTER_ROOT/data/dinov2_cluster_c10_seed42/hierarchy.json"
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

fail(){ echo "Native-recovery audit failed: $*" >&2; exit 1; }

preflight_seed(){
    local teacher_seed="$1" recovery_seed raw collapsed fkd
    [[ -f "$CLUSTER_ROOT/tseed${teacher_seed}/models/dinov2_cluster_c10_seed42_tseed${teacher_seed}/ResNet18.pth" ]] \
        || fail "missing Cluster Teacher seed$teacher_seed"
    [[ -f "$C1_ROOT/tseed${teacher_seed}/models/random_c1_pseed42_tseed${teacher_seed}/ResNet18.pth" ]] \
        || fail "missing C1 Teacher seed$teacher_seed"
    for recovery_seed in 41 42 43; do
        raw="$EXP_ROOT/tseed${teacher_seed}/native_synthetic/cluster_c10_native100_rseed${recovery_seed}"
        collapsed="$EXP_ROOT/tseed${teacher_seed}/coarse_sources/cluster_c10_native100_rseed${recovery_seed}"
        fkd="$EXP_ROOT/tseed${teacher_seed}/fkd/c1_soft_rseed${recovery_seed}_bs10_ipc10"
        [[ "$(find "$raw" -type f -name '*.jpg' | wc -l)" == 100 ]] \
            || fail "incomplete raw source: tseed=$teacher_seed rseed=$recovery_seed"
        [[ "$(find "$raw" -mindepth 1 -maxdepth 1 -type d | wc -l)" == 100 ]] \
            || fail "raw source does not have 100 native directories"
        [[ "$(find "$collapsed" -type f -name '*.jpg' | wc -l)" == 100 ]] \
            || fail "incomplete collapsed source"
        [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] \
            || fail "incomplete C1 FKD labels"
    done
}

audit_seed(){
    local teacher_seed="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/class_in_class:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/audit_imagenette_native100_recovery_outputs.py" \
        --teacher-seed "$teacher_seed" --experiment-root "$EXP_ROOT" \
        --cluster-checkpoint "$CLUSTER_ROOT/tseed${teacher_seed}/models/dinov2_cluster_c10_seed42_tseed${teacher_seed}/ResNet18.pth" \
        --c1-checkpoint "$C1_ROOT/tseed${teacher_seed}/models/random_c1_pseed42_tseed${teacher_seed}/ResNet18.pth" \
        --hierarchy "$HIERARCHY" --recovery-seeds 41 42 43 \
        --workers "$WORKERS" --fkd-epoch-stride 10 \
        --output "$OUTPUT_DIR/tseed${teacher_seed}.json" \
        > "$LOG_ROOT/tseed${teacher_seed}.log" 2>&1
}

echo "[1/3] Strict audit asset preflight"
[[ -f "$HIERARCHY" ]] || fail "missing hierarchy"
preflight_seed 43
preflight_seed 44

echo "[2/3] Exact-pixel Teacher audit and consumed-FKD audit, one seed per A100"
audit_seed 43 "$GPU0" & pid43=$!
audit_seed 44 "$GPU1" & pid44=$!
status=0
wait "$pid43" || status=1
wait "$pid44" || status=1
(( status == 0 )) || fail "one or both Teacher-seed audits"

echo "[3/3] Combine six Teacher/recovery roots"
PYTHONPATH="$ROOT/class_in_class:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/summarize_imagenette_native100_recovery_audit.py" \
    --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/summary.json" \
    > "$LOG_ROOT/summary.log" 2>&1 || fail summarize

echo "Complete: $OUTPUT_DIR/summary.json"
