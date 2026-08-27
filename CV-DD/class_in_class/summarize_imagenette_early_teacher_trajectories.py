import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    trajectories = {}
    for teacher_seed in (43, 44):
        for c in (1, 100):
            directory = root / f"tseed{teacher_seed}" / "models" / f"c{c}_tseed{teacher_seed}"
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            checkpoints = sorted((directory / "checkpoints").glob("epoch_*.pth"))
            if len(metrics) != 300 or len(checkpoints) != 300:
                raise ValueError(f"incomplete trajectory: {directory}")
            key = f"c{c}_tseed{teacher_seed}"
            trajectories[key] = {
                "directory": str(directory),
                "checkpoints": len(checkpoints),
                "epochs": metrics,
            }
            for record in metrics:
                rows.append({
                    "teacher_seed": teacher_seed,
                    "C": c,
                    "epoch": record["epoch"],
                    "checkpoint": record["checkpoint"],
                    "lr": record["lr"],
                    "train_acc": record["train_acc"],
                    "val_acc": record["val_acc"],
                    "val_native_accuracy": record["val_native_accuracy"],
                    "sd_z": record["sd_z"],
                    "marg_label_entropy_T20": record["marg_label_entropy_T20"],
                    "participation_rank": record["participation_rank"],
                })
    result = {
        "protocol": (
            "ImageNette C1/C100 random-subclass early Teacher trajectories; "
            "per-epoch checkpoints and clean train/test geometry metrics"
        ),
        "teacher_seeds": [43, 44],
        "c_values": [1, 100],
        "epochs": 300,
        "trajectories": trajectories,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {output} and {csv_path}")


if __name__ == "__main__":
    main()
