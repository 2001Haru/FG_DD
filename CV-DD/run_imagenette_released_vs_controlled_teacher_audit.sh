#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-run1}"
OFFICIAL_SPLIT_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split"
SEED42="$OFFICIAL_SPLIT_ROOT/models/random_c1_pseed42_tseed42/ResNet18.pth"
RELEASED="$Main_Data_Path/class_in_class/imagenette_released_exact_teacher/$RUN_NAME/ResNet18.pth"
HIERARCHY="$OFFICIAL_SPLIT_ROOT/data/random_c1_pseed42/hierarchy.json"
OUTPUT="$Main_Data_Path/class_in_class/imagenette_released_exact_teacher/$RUN_NAME/vs_seed42_controlled.json"
LOG="$ROOT/logs/imagenette_released_exact_teacher/$RUN_NAME/vs_seed42_controlled.log"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

for path in "$SEED42" "$RELEASED" "$HIERARCHY"; do
    [[ -f "$path" ]] || { echo "missing audit input: $path" >&2; exit 1; }
done

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_teachers.py" \
    --official-root "$Main_Data_Path/test_data/imagenet-nette" \
    --official-checkpoint "$SEED42" --official-label seed42_controlled \
    --controlled-checkpoint "$RELEASED" --controlled-label released_exact_run1 \
    --controlled-hierarchy "$HIERARCHY" --workers "$WORKERS" --temperature 20 \
    --output "$OUTPUT" > "$LOG" 2>&1

echo "Complete: $OUTPUT"
