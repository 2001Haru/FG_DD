import argparse
import json
import os
from pathlib import Path

import torch


TRAINING_EPOCHS = (4, 8, 16, 32, 64, 100, 150, 200, 250, 300)
BASE_OPTIMAL_TEMPERATURE = {1: 800.0, 100: 200.0}


def compare_state_dicts(left_path, right_path):
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    if isinstance(left, dict) and "state_dict" in left:
        left = left["state_dict"]
    if isinstance(right, dict) and "state_dict" in right:
        right = right["state_dict"]
    if left.keys() != right.keys():
        return {"keys_exact_match": False, "exact_match": False}
    squared_difference = squared_left = squared_right = dot = 0.0
    maximum_absolute_difference = 0.0
    unequal_tensors = 0
    for key in left:
        left_tensor = left[key].double().reshape(-1)
        right_tensor = right[key].double().reshape(-1)
        difference = left_tensor - right_tensor
        if not torch.equal(left[key], right[key]):
            unequal_tensors += 1
        squared_difference += difference.square().sum().item()
        squared_left += left_tensor.square().sum().item()
        squared_right += right_tensor.square().sum().item()
        dot += (left_tensor * right_tensor).sum().item()
        if difference.numel():
            maximum_absolute_difference = max(
                maximum_absolute_difference, difference.abs().max().item()
            )
    return {
        "keys_exact_match": True,
        "exact_match": unequal_tensors == 0,
        "unequal_tensor_count": unequal_tensors,
        "global_relative_l2_trajectory_denominator": (
            (squared_difference / max(squared_left, 1e-30)) ** 0.5
        ),
        "global_cosine": dot / max((squared_left * squared_right) ** 0.5, 1e-30),
        "maximum_absolute_parameter_difference": maximum_absolute_difference,
    }


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

    selections = []
    endpoint_audit = {}
    for c in (1, 100):
        metrics = metrics_by_c[c]
        final_sd = float(metrics[-1]["sd_z"])
        if final_sd <= 0:
            raise ValueError(f"invalid final sd(z): C={c}")
        trajectory_dir = trajectory_root / "models" / f"c{c}_tseed{args.teacher_seed}"
        existing = (
            existing_root / "models"
            / f"random_c{c}_pseed42_tseed{args.teacher_seed}" / "ResNet18.pth"
        )
        trajectory_final = trajectory_dir / "ResNet18.pth"
        comparison = compare_state_dicts(trajectory_final, existing)
        match = comparison["exact_match"]
        endpoint_audit[f"c{c}"] = {
            "trajectory_final": str(trajectory_final),
            "existing_final": str(existing),
            "state_dict_exact_match": match,
            "state_dict_comparison": comparison,
        }
        for training_epoch in TRAINING_EPOCHS:
            record = metrics[training_epoch - 1]
            label = f"e{training_epoch:03d}"
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
                "training_epoch": training_epoch,
                "checkpoint_epoch_index": epoch,
                "epoch": epoch,
                "actual_train_accuracy": float(record["train_acc"]),
                "actual_val_accuracy": float(record["val_acc"]),
                "sd_z": float(record["sd_z"]),
                "marg_label_entropy_T20": float(record["marg_label_entropy_T20"]),
                "participation_rank": float(record["participation_rank"]),
                "lr": float(record["lr"]),
                "val_sd_z": float(record["val_sd_z"]),
                "val_marg_entropy_T20": float(record["val_marg_entropy_T20"]),
                "val_participation_rank": float(record["val_participation_rank"]),
                "final_sd_z": final_sd,
                "predicted_temperature": predicted_temperature,
                "checkpoint": str(checkpoint),
                "teacher_view": str(view),
            })

    result = {
        "protocol": (
            "ImageNette early Teacher checkpoint selection by collapsed coarse val "
            "accuracy; predicted T = final optimal T * sd(z_e)/sd(z_final)"
        ),
        "teacher_seed": args.teacher_seed,
        "training_epochs": list(TRAINING_EPOCHS),
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
            handle.write("\t".join(map(str, (
                row["C"], row["label"], row["epoch"],
                row["actual_val_accuracy"], row["sd_z"],
                row["predicted_temperature"], row["teacher_view"],
            ))) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
