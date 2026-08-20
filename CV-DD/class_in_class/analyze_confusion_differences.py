import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import ResNet18  # noqa: E402


ARMS = ("baseline", "random", "coarse_target", "random_coarse_target", "oracle")
COMPARISONS = {
    "oracle_minus_baseline": ("oracle", "baseline"),
    "random_minus_baseline": ("random", "baseline"),
    "oracle_minus_random": ("oracle", "random"),
    "coarse_target_minus_baseline": ("coarse_target", "baseline"),
    "oracle_minus_coarse_target": ("oracle", "coarse_target"),
    "random_minus_random_coarse_target": ("random", "random_coarse_target"),
}


def rank(values):
    order = np.argsort(values)
    result = np.empty_like(order, dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def checkpoint_path(root, arm, seed, partition_seed):
    if arm == "random":
        experiment = f"class_in_class_random_pseed{partition_seed}_rseed{seed}"
    elif arm == "random_coarse_target":
        experiment = f"class_in_class_random_coarse_target_pseed{partition_seed}_rseed{seed}"
    else:
        experiment = f"class_in_class_{arm}_rseed{seed}"
    return root / "cifar20" / experiment / "model_best.pth.tar"


def confusion_for_checkpoint(path, loader):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet18(20)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    confusion = torch.zeros(20, 20, dtype=torch.long)
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.cuda(non_blocking=True)).argmax(dim=1).cpu()
            indices = targets * 20 + predictions
            confusion += torch.bincount(indices, minlength=400).reshape(20, 20)
    return confusion.numpy(), float(checkpoint["best_acc1"])


def metrics(confusion):
    diagonal = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(diagonal), where=(precision + recall) > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def write_matrix(path, matrix, names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [float(value) for value in row])


def correlations(x, y):
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rank(x), rank(y))[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser("Best-checkpoint confusion/F1 analysis for three-arm experiment")
    parser.add_argument("--post-eval-root", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--superclass-results", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-partition-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hierarchy = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    names = hierarchy["coarse_names"]

    dataset = datasets.ImageFolder(args.val_dir, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ]))
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    confusions, normalized, arm_metrics = {}, {}, {}
    metric_rows = []
    best_top1 = {}
    for arm in ARMS:
        for seed in args.recovery_seeds:
            key = (arm, seed)
            path = checkpoint_path(Path(args.post_eval_root), arm, seed,
                                   args.random_partition_seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            confusion, top1 = confusion_for_checkpoint(path, loader)
            confusions[key] = confusion
            normalized[key] = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)
            arm_metrics[key] = metrics(confusion)
            best_top1[key] = top1
            write_matrix(output / f"confusion_{arm}_seed{seed}_counts.csv", confusion, names)
            write_matrix(output / f"confusion_{arm}_seed{seed}_row_normalized.csv",
                         normalized[key], names)
            for class_id, name in enumerate(names):
                metric_rows.append({
                    "arm": arm, "seed": seed, "class_id": class_id, "class_name": name,
                    "precision": arm_metrics[key]["precision"][class_id],
                    "recall": arm_metrics[key]["recall"][class_id],
                    "f1": arm_metrics[key]["f1"][class_id],
                    "support": arm_metrics[key]["support"][class_id],
                })

    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader(); writer.writerows(metric_rows)

    feature_rows = {}
    with Path(args.superclass_results).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            feature_rows[int(row["coarse_id"])] = row

    difference_rows, boundary_rows = [], []
    correlations_output = {"best_top1": {}}
    comparison_f1 = {}
    for comparison, (positive, negative) in COMPARISONS.items():
        deltas = np.stack([
            normalized[(positive, seed)] - normalized[(negative, seed)]
            for seed in args.recovery_seeds
        ])
        mean_delta, std_delta = deltas.mean(axis=0), deltas.std(axis=0, ddof=1)
        write_matrix(output / f"delta_confusion_{comparison}_mean.csv", mean_delta, names)
        write_matrix(output / f"delta_confusion_{comparison}_std.csv", std_delta, names)

        f1_differences = np.stack([
            arm_metrics[(positive, seed)]["f1"] - arm_metrics[(negative, seed)]["f1"]
            for seed in args.recovery_seeds
        ])
        recall_differences = np.stack([
            arm_metrics[(positive, seed)]["recall"] - arm_metrics[(negative, seed)]["recall"]
            for seed in args.recovery_seeds
        ])
        precision_differences = np.stack([
            arm_metrics[(positive, seed)]["precision"] - arm_metrics[(negative, seed)]["precision"]
            for seed in args.recovery_seeds
        ])
        comparison_f1[comparison] = (f1_differences.mean(axis=0),
                                     f1_differences.std(axis=0, ddof=1))
        for class_id, name in enumerate(names):
            difference_rows.append({
                "comparison": comparison, "class_id": class_id, "class_name": name,
                "precision_gain_mean": precision_differences[:, class_id].mean(),
                "precision_gain_std": precision_differences[:, class_id].std(ddof=1),
                "recall_gain_mean": recall_differences[:, class_id].mean(),
                "recall_gain_std": recall_differences[:, class_id].std(ddof=1),
                "f1_gain_mean": f1_differences[:, class_id].mean(),
                "f1_gain_std": f1_differences[:, class_id].std(ddof=1),
            })
        for left in range(20):
            for right in range(left + 1, 20):
                left_to_right = mean_delta[left, right]
                right_to_left = mean_delta[right, left]
                boundary_rows.append({
                    "comparison": comparison,
                    "left_id": left, "left_name": names[left],
                    "right_id": right, "right_name": names[right],
                    "delta_left_to_right": left_to_right,
                    "delta_right_to_left": right_to_left,
                    "antisymmetric_shift_favoring_left": 0.5 * (right_to_left - left_to_right),
                    "symmetric_pair_error_change": 0.5 * (right_to_left + left_to_right),
                })

        x_abs = np.array([float(feature_rows[i]["fine_centroid_cosine_distance"])
                          for i in range(20)])
        x_rho = np.array([float(feature_rows[i]["rho_inter_over_intra"])
                          for i in range(20)])
        y_f1 = f1_differences.mean(axis=0)
        correlations_output[comparison] = {
            "absolute_dispersion_vs_f1_gain": correlations(x_abs, y_f1),
            "relative_margin_rho_vs_f1_gain": correlations(x_rho, y_f1),
        }
        correlations_output["best_top1"][comparison] = [
            best_top1[(positive, seed)] - best_top1[(negative, seed)]
            for seed in args.recovery_seeds
        ]

    with (output / "paired_per_class_metric_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=difference_rows[0].keys())
        writer.writeheader(); writer.writerows(difference_rows)
    with (output / "pairwise_boundary_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=boundary_rows[0].keys())
        writer.writeheader(); writer.writerows(boundary_rows)

    mutual_pairs = []
    for left in range(20):
        right = int(feature_rows[left]["nearest_other_superclass_id"])
        if left < right and int(feature_rows[right]["nearest_other_superclass_id"]) == left:
            for comparison in COMPARISONS:
                pair_row = next(row for row in boundary_rows
                                if row["comparison"] == comparison
                                and row["left_id"] == left and row["right_id"] == right)
                f1_mean, f1_std = comparison_f1[comparison]
                mutual_pairs.append({
                    **pair_row,
                    "pair_f1_gain_mean": 0.5 * (f1_mean[left] + f1_mean[right]),
                    "pair_f1_gain_std_rms": float(np.sqrt(
                        0.5 * (f1_std[left] ** 2 + f1_std[right] ** 2)
                    )),
                })
    if mutual_pairs:
        with (output / "mutual_nearest_pair_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=mutual_pairs[0].keys())
            writer.writeheader(); writer.writerows(mutual_pairs)

    rho = np.array([float(feature_rows[i]["rho_inter_over_intra"]) for i in range(20)])
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(6 * len(COMPARISONS), 5.5),
                             sharex=True)
    for axis, comparison in zip(axes, COMPARISONS):
        f1_mean, f1_std = comparison_f1[comparison]
        axis.errorbar(rho, 100 * f1_mean, yerr=100 * f1_std, fmt="o", capsize=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(comparison.replace("_", " "))
        axis.set_xlabel("relative margin rho")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Paired per-class F1 gain (points)")
    fig.tight_layout()
    fig.savefig(output / "relative_margin_vs_f1_gains.png", dpi=200)
    plt.close(fig)

    with (output / "confusion_f1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(correlations_output, handle, indent=2)
    print(json.dumps(correlations_output, indent=2))
    print(f"Confusion/F1 analysis written to {output}")


if __name__ == "__main__":
    main()
