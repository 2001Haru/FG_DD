import argparse
import json
import os
from pathlib import Path

import torch


TARGET_LABELS = ("v45", "v60", "v66", "v72", "v79")
BASE_TARGETS = (45.0, 60.0, 66.0, 72.0, 79.0)
BASE_OPTIMAL_TEMPERATURE = {1: 800.0, 100: 200.0}


def ordered_nearest(records, targets):
    """Minimum total absolute error subject to strictly increasing epochs."""
    candidates = records[:-1]
    count = len(targets)
    infinity = float("inf")
    dp = [[infinity] * len(candidates) for _ in range(count)]
    parent = [[None] * len(candidates) for _ in range(count)]
    for index, record in enumerate(candidates):
        dp[0][index] = abs(float(record["val_acc"]) - targets[0])
    for target_index in range(1, count):
        best_cost, best_index = infinity, None
        for index, record in enumerate(candidates):
            previous = index - 1
            if previous >= 0 and dp[target_index - 1][previous] < best_cost:
                best_cost = dp[target_index - 1][previous]
                best_index = previous
            if best_index is not None:
                dp[target_index][index] = (
                    best_cost + abs(float(record["val_acc"]) - targets[target_index])
                )
                parent[target_index][index] = best_index
    end = min(range(len(candidates)), key=lambda index: dp[-1][index])
    selected = [end]
    for target_index in range(count - 1, 0, -1):
        end = parent[target_index][end]
        if end is None:
            raise RuntimeError("cannot select strictly ordered early checkpoints")
        selected.append(end)
    selected.reverse()
    return [candidates[index] for index in selected]


def exact_state_dict_match(left_path, right_path):
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    if isinstance(left, dict) and "state_dict" in left:
        left = left["state_dict"]
    if isinstance(right, dict) and "state_dict" in right:
        right = right["state_dict"]
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def materialize_teacher_view(checkpoint, directory):
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / "ResNet18.pth"
    if link.exists() or link.is_symlink():
        if link.resolve() == checkpoint.resolve():
            return
        link.unlink()
    os.symlink(checkpoint.resolve(), link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--existing-teacher-root", required=True)
    parser.add_argument("--teacher-seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    trajectory_root = Path(args.trajectory_root) / f"tseed{args.teacher_seed}"
    existing_root = Path(args.existing_teacher_root) / f"tseed{args.teacher_seed}"
    output_root = Path(args.output_root) / f"tseed{args.teacher_seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_by_c = {}
    for c in (1, 100):
        directory = trajectory_root / "models" / f"c{c}_tseed{args.teacher_seed}"
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if len(metrics) != 300:
            raise ValueError(f"expected 300 epochs: {directory}")
        metrics_by_c[c] = metrics

    c100_final_val = float(metrics_by_c[100][-1]["val_acc"])
    selections = []
    endpoint_audit = {}
    for c in (1, 100):
        metrics = metrics_by_c[c]
        targets = list(BASE_TARGETS)
        if c == 1:
            targets[-1] = c100_final_val
        selected = ordered_nearest(metrics, targets)
        final_sd = float(metrics[-1]["sd_z"])
        if final_sd <= 0:
            raise ValueError(f"invalid final sd(z): C={c}")
        trajectory_dir = trajectory_root / "models" / f"c{c}_tseed{args.teacher_seed}"
        existing = (
            existing_root / "models"
            / f"random_c{c}_pseed42_tseed{args.teacher_seed}" / "ResNet18.pth"
        )
        trajectory_final = trajectory_dir / "ResNet18.pth"
        match = exact_state_dict_match(trajectory_final, existing)
        endpoint_audit[f"c{c}"] = {
            "trajectory_final": str(trajectory_final),
            "existing_final": str(existing),
            "state_dict_exact_match": match,
        }
        if not match:
            raise RuntimeError(
                f"trajectory final checkpoint does not exactly match reusable Teacher: C={c}"
            )
        for label, target, record in zip(TARGET_LABELS, targets, selected):
            epoch = int(record["epoch"])
            checkpoint = trajectory_dir / "checkpoints" / record["checkpoint"]
            predicted_temperature = (
                BASE_OPTIMAL_TEMPERATURE[c]
                * float(record["sd_z"]) / final_sd
            )
            view = output_root / "teacher_views" / f"c{c}_{label}_e{epoch:03d}"
            materialize_teacher_view(checkpoint, view)
            selections.append({
                "teacher_seed": args.teacher_seed,
                "C": c,
                "label": label,
                "requested_val_accuracy": target,
                "epoch": epoch,
                "actual_train_accuracy": float(record["train_acc"]),
                "actual_val_accuracy": float(record["val_acc"]),
                "sd_z": float(record["sd_z"]),
                "final_sd_z": final_sd,
                "predicted_temperature": predicted_temperature,
                "checkpoint": str(checkpoint),
                "teacher_view": str(view),
            })
        final = metrics[-1]
        selections.append({
            "teacher_seed": args.teacher_seed,
            "C": c,
            "label": "final",
            "requested_val_accuracy": float(final["val_acc"]),
            "epoch": int(final["epoch"]),
            "actual_train_accuracy": float(final["train_acc"]),
            "actual_val_accuracy": float(final["val_acc"]),
            "sd_z": float(final["sd_z"]),
            "final_sd_z": final_sd,
            "predicted_temperature": BASE_OPTIMAL_TEMPERATURE[c],
            "checkpoint": str(trajectory_final),
            "teacher_view": str(existing.parent),
            "reused_existing_downstream_results": True,
        })

    result = {
        "protocol": (
            "ImageNette early Teacher checkpoint selection by collapsed coarse val "
            "accuracy; predicted T = final optimal T * sd(z_e)/sd(z_final)"
        ),
        "teacher_seed": args.teacher_seed,
        "targets": list(TARGET_LABELS),
        "c100_final_val_accuracy_target_for_c1_v79": c100_final_val,
        "base_optimal_temperature": {
            "C1": BASE_OPTIMAL_TEMPERATURE[1],
            "C100": BASE_OPTIMAL_TEMPERATURE[100],
        },
        "endpoint_reuse_audit": endpoint_audit,
        "selections": selections,
    }
    output = output_root / "selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tsv = output_root / "selection_early.tsv"
    with tsv.open("w", encoding="utf-8") as handle:
        for row in selections:
            if row["label"] == "final":
                continue
            handle.write("\t".join(map(str, (
                row["C"], row["label"], row["epoch"],
                row["actual_val_accuracy"], row["sd_z"],
                row["predicted_temperature"], row["teacher_view"],
            ))) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
