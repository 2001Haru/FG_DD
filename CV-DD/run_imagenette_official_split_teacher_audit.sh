#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
PARTITION_SEED="${PARTITION_SEED:-42}"
TEACHER_SEED="${TEACHER_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-20}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split}"
DATA="$EXP_ROOT/data/random_c1_pseed${PARTITION_SEED}"
CONTROLLED="$EXP_ROOT/models/random_c1_pseed${PARTITION_SEED}_tseed${TEACHER_SEED}/ResNet18.pth"
OFFICIAL="$Main_Data_Path/offline_models/imagenet-nette/ResNet18.pth"
OFFICIAL_ROOT="$Main_Data_Path/test_data/imagenet-nette"
OUTPUT="$EXP_ROOT/audits/official_vs_retrained_c1_extended.json"
LOG="$ROOT/logs/imagenette_cic_t_official_split/teacher_extended_audit.log"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

[[ -f "$CONTROLLED" ]] || { echo "missing retrained C1 Teacher: $CONTROLLED" >&2; exit 1; }
[[ -f "$OFFICIAL" ]] || { echo "missing official Teacher: $OFFICIAL" >&2; exit 1; }
[[ -f "$DATA/hierarchy.json" ]] || { echo "missing hierarchy: $DATA" >&2; exit 1; }

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_teachers.py" \
    --official-root "$OFFICIAL_ROOT" --official-checkpoint "$OFFICIAL" \
    --controlled-checkpoint "$CONTROLLED" --controlled-hierarchy "$DATA/hierarchy.json" \
    --workers "$WORKERS" --temperature "$TEMPERATURE" --output "$OUTPUT" \
    > "$LOG" 2>&1

echo "Complete: $OUTPUT"
