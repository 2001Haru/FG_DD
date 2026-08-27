#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
BASE="$Main_Data_Path/class_in_class"
mkdir -p "$ROOT/logs/imagenette_early_teacher_downstream"
for seed in 43 44; do
    python -u "$ROOT/class_in_class/select_imagenette_early_teacher_checkpoints.py" \
        --trajectory-root "$BASE/imagenette_early_teacher_trajectories" \
        --existing-teacher-root "$BASE/imagenette_cic_t_official_split_lr0p1_tseeds43_44" \
        --teacher-seed "$seed" \
        --output-root "$BASE/imagenette_early_teacher_downstream" \
        > "$ROOT/logs/imagenette_early_teacher_downstream/selection_seed${seed}.log" 2>&1
done
python - "$BASE/imagenette_early_teacher_downstream" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("seed C label training_epoch checkpoint_index train_acc val_acc sd_z predicted_T")
for seed in (43, 44):
    payload = json.loads((root / f"tseed{seed}" / "selection.json").read_text())
    for row in payload["selections"]:
        print(
            seed, row["C"], row["label"], row["training_epoch"], row["epoch"],
            f'{row["actual_train_accuracy"]:.3f}',
            f'{row["actual_val_accuracy"]:.3f}',
            f'{row["sd_z"]:.6f}',
            f'{row["predicted_temperature"]:.6f}',
        )
PY
