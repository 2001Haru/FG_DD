#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
PARTITION_SEED="${PARTITION_SEED:-42}"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_cic_t_official_split_lr0p1_tseeds43_44/high_c_train_audit}"
mkdir -p "$LOG_ROOT"

fail(){ echo "C500/C800 train-only Teacher audit failed: $*" >&2; exit 1; }

audit_teacher_seed(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    local c data model output log
    for c in 500 800; do
        data="$teacher_root/data/random_c${c}_pseed${PARTITION_SEED}"
        model="$teacher_root/models/random_c${c}_pseed${PARTITION_SEED}_tseed${teacher_seed}/ResNet18.pth"
        output="$teacher_root/audits/random_c${c}_teacher_train_only_audit.json"
        log="$LOG_ROOT/tseed${teacher_seed}_c${c}.log"

        [[ -f "$data/hierarchy.json" ]] || {
            echo "missing hierarchy: $data/hierarchy.json" >&2; return 1;
        }
        [[ -f "$model" ]] || {
            echo "missing Teacher checkpoint: $model" >&2; return 1;
        }

        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/class_in_class/audit_imagenette_subclass_teacher.py" \
            --data-dir "$data" --checkpoint "$model" \
            --mapping "$data/hierarchy.json" --output "$output" \
            --workers "$WORKERS" --train-only \
            > "$log" 2>&1

        python - "$output" <<'PY'
import json, sys
path = sys.argv[1]
record = json.load(open(path, encoding="utf-8"))
assert record.get("audit_scope") == "train_only", record.get("audit_scope")
assert "val" not in record
train = record["train"]
print(
    f"{path}: C={record['subclasses_per_coarse']}, "
    f"images={train['images']}, native_top1={train['native_subclass_top1']:.6f}, "
    f"coarse_top1={train['collapsed_coarse10_top1']:.6f}, "
    f"within_parent_entropy={train['within_parent_entropy']:.8f}"
)
PY
    done
}

echo "Train-only audit: Teacher seed43 on GPU$GPU0; seed44 on GPU$GPU1"
audit_teacher_seed 43 "$GPU0" & pid43=$!
audit_teacher_seed 44 "$GPU1" & pid44=$!
status=0
wait "$pid43" || status=1
wait "$pid44" || status=1
(( status == 0 )) || fail "one or more audit streams"

python - "$MASTER_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for teacher_seed in (43, 44):
    for c in (500, 800):
        path = root / f"tseed{teacher_seed}" / "audits" / f"random_c{c}_teacher_train_only_audit.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        train = record["train"]
        rows.append({
            "teacher_seed": teacher_seed,
            "C": c,
            "heads": record["num_pseudo_classes"],
            "train_images": train["images"],
            "train_native_top1": train["native_subclass_top1"],
            "train_coarse_top1": train["collapsed_coarse10_top1"],
            "train_within_parent_entropy": train["within_parent_entropy"],
        })
result = {
    "definition": "Clean deterministic train-split evaluation only; no validation/test images are loaded.",
    "rows": rows,
}
output = root / "analysis" / "teacher_train_only_audit_c500_c800.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"Saved: {output}")
PY
