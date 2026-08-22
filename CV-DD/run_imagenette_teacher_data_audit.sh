#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${WORKERS:-16}"
PARTITION_SEED="${PARTITION_SEED:-42}"
TEACHER_SEED="${TEACHER_SEED:-42}"
VLCP_ROOT="${VLCP_ROOT:-/linxi/dataset/VLCP/ImageNette}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$Main_Data_Path/test_data/imagenet-nette}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t}"
CONTROL_DATA="$EXP_ROOT/data/random_c1_pseed${PARTITION_SEED}"
CONTROL_MODEL="$EXP_ROOT/models/random_c1_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}/ResNet18.pth"
OFFICIAL_MODEL="$Main_Data_Path/offline_models/imagenet-nette/ResNet18.pth"
OUTPUT="$EXP_ROOT/audits/teacher_data_forensics"
LOGS="$ROOT/logs/imagenette_cic_t/teacher_data_forensics"
mkdir -p "$OUTPUT" "$LOGS"

fail(){ echo "ImageNette Teacher/data audit failed: $*" >&2; exit 1; }
[[ -f "$CONTROL_MODEL" ]] || fail "missing controlled Teacher: $CONTROL_MODEL"
[[ -f "$OFFICIAL_MODEL" ]] || fail "missing official Teacher: $OFFICIAL_MODEL"
[[ -f "$CONTROL_DATA/hierarchy.json" ]] || fail "missing C1 hierarchy"

python -u "$ROOT/class_in_class/audit_imagenette_split_overlap.py" \
    --vlcp-root "$VLCP_ROOT" --official-root "$OFFICIAL_ROOT" --workers "$WORKERS" \
    --dhash-threshold 4 --output "$OUTPUT/split_overlap.json" \
    > "$LOGS/split_overlap.log" 2>&1 & overlap_pid=$!

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_teachers.py" \
    --official-root "$OFFICIAL_ROOT" --official-checkpoint "$OFFICIAL_MODEL" \
    --controlled-checkpoint "$CONTROL_MODEL" --controlled-hierarchy "$CONTROL_DATA/hierarchy.json" \
    --workers "$WORKERS" --output "$OUTPUT/teacher_comparison.json" \
    > "$LOGS/teacher_comparison.log" 2>&1 & teacher_pid=$!

overlap_status=0; teacher_status=0
wait "$overlap_pid" || overlap_status=$?
wait "$teacher_pid" || teacher_status=$?
(( overlap_status == 0 )) || fail "split-overlap audit failed; see $LOGS/split_overlap.log"
(( teacher_status == 0 )) || fail "Teacher audit failed; see $LOGS/teacher_comparison.log"
echo "Complete: $OUTPUT/split_overlap.json"
echo "Complete: $OUTPUT/teacher_comparison.json"
