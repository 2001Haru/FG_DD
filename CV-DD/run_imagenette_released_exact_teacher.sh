#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${AUDIT_WORKERS:-8}"
RUN_NAME="${RUN_NAME:-run1}"
DATA_ROOT="${DATA_ROOT:-$Main_Data_Path/test_data/imagenet-nette}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_released_exact_teacher}"
OUTPUT="$EXP_ROOT/$RUN_NAME"
CHECKPOINT="$OUTPUT/resnet18.pth"
AUDIT="$OUTPUT/official_comparison.json"
LOGS="$ROOT/logs/imagenette_released_exact_teacher/$RUN_NAME"
mkdir -p "$OUTPUT" "$LOGS"

fail(){ echo "Released-exact ImageNette Teacher run failed: $*" >&2; exit 1; }
[[ ! -e "$CHECKPOINT" ]] || fail "checkpoint already exists; choose another RUN_NAME: $CHECKPOINT"
TRAIN_COUNT="$(find "$DATA_ROOT/train" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
TEST_COUNT="$(find "$DATA_ROOT/test" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
(( TRAIN_COUNT == 9469 )) || fail "train images=$TRAIN_COUNT, expected 9469"
(( TEST_COUNT == 3925 )) || fail "test images=$TEST_COUNT, expected 3925"

python - <<'PY' > "$OUTPUT/environment.txt"
import platform
import torch
import torchvision
print("python", platform.python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("cudnn_deterministic_default", torch.backends.cudnn.deterministic)
print("cudnn_benchmark_default", torch.backends.cudnn.benchmark)
PY

echo "Training released-exact unseeded ResNet18: output=$OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/train_imagenette_teacher_released_exact.py" \
    --data-dir "$DATA_ROOT" --save-dir "$OUTPUT" \
    > "$LOGS/train.log" 2>&1
[[ -f "$CHECKPOINT" ]] || fail "training ended without checkpoint"
cp "$CHECKPOINT" "$OUTPUT/ResNet18.pth"

# Reuse the official-split C1 hierarchy only for its verified class order.
HIERARCHY="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split/data/random_c1_pseed42/hierarchy.json"
[[ -f "$HIERARCHY" ]] || fail "missing official-split C1 hierarchy: $HIERARCHY"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_teachers.py" \
    --official-root "$DATA_ROOT" \
    --official-checkpoint "$Main_Data_Path/offline_models/imagenet-nette/ResNet18.pth" \
    --controlled-checkpoint "$CHECKPOINT" --controlled-hierarchy "$HIERARCHY" \
    --workers "$WORKERS" --temperature 20 --output "$AUDIT" \
    > "$LOGS/audit.log" 2>&1

echo "Complete: $CHECKPOINT"
echo "Complete: $AUDIT"
