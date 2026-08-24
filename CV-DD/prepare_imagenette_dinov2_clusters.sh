#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
CLUSTER_SEED="${CLUSTER_SEED:-42}"
C_VALUES_TEXT="${C_VALUES:-2 5 10 20 50 100 200 500}"
read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
SOURCE_ROOT="${SOURCE_ROOT:-$val_dir/imagenet-nette}"
DINOV2_MODEL="${DINOV2_MODEL:-/linxi/models/DINOv2/dinov2-base}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
FEATURE_CACHE="${FEATURE_CACHE:-$EXP_ROOT/features/dinov2_base_official_imagenette.pt}"
DATA_ROOT="$EXP_ROOT/data"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cic_t_dinov2_cluster_seed42/prepare}"
mkdir -p "$(dirname "$FEATURE_CACHE")" "$DATA_ROOT" "$LOG_ROOT"

fail(){ echo "ImageNette DINO cluster preparation failed: $*" >&2; exit 1; }

for split in train test; do
    [[ -d "$SOURCE_ROOT/$split" ]] || fail "missing source split: $SOURCE_ROOT/$split"
done
[[ -d "$DINOV2_MODEL" ]] || fail "missing local DINOv2 model: $DINOV2_MODEL"

echo "[1/2] Encoding official ImageNette once with DINOv2-Base on GPU$GPU"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/encode_imagenette_dinov2.py" \
    --data-root "$SOURCE_ROOT" --model-dir "$DINOV2_MODEL" \
    --output "$FEATURE_CACHE" --splits train test \
    --batch-size "$ENCODE_BATCH_SIZE" --workers "$WORKERS" --device cuda \
    > "$LOG_ROOT/encode.log" 2>&1 || fail "DINO encoding; see $LOG_ROOT/encode.log"

echo "[2/2] Balanced within-class spherical clustering: C=${C_VALUES_ARRAY[*]}"
for c in "${C_VALUES_ARRAY[@]}"; do
    output="$DATA_ROOT/dinov2_cluster_c${c}_seed${CLUSTER_SEED}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/prepare_imagenette_dinov2_clusters.py" \
        --source-root "$SOURCE_ROOT" --source-validation-split test \
        --feature-cache "$FEATURE_CACHE" --output-dir "$output" \
        --subclasses "$c" --seed "$CLUSTER_SEED" --cluster-device cuda \
        --kmeans-iterations 25 --kmeans-restarts 3 --balance-refinements 2 \
        --repair-invalid-output \
        > "$LOG_ROOT/cluster_c${c}.log" 2>&1 \
        || fail "C=$c clustering; see $LOG_ROOT/cluster_c${c}.log"
done

python - "$EXP_ROOT" "$CLUSTER_SEED" "${C_VALUES_ARRAY[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
rows = []
for c_text in sys.argv[3:]:
    c = int(c_text)
    path = root / "data" / f"dinov2_cluster_c{c}_seed{seed}" / "hierarchy.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    sizes = [
        (item["minimum_cluster_size"], item["maximum_cluster_size"])
        for item in record["cluster_stats"].values()
    ]
    rows.append({
        "C": c,
        "heads": record["num_pseudo_classes"],
        "train_images": record["source_train_images"],
        "test_images": record["source_val_images"],
        "minimum_train_subclass_size": min(value[0] for value in sizes),
        "maximum_train_subclass_size": max(value[1] for value in sizes),
        "partition_dir": str(path.parent),
    })
output = root / "analysis" / "dinov2_cluster_preparation_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rows": rows}, indent=2))
print(f"Saved: {output}")
PY
