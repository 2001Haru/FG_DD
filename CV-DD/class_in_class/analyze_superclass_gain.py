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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import ResNet18  # noqa: E402


def rank(values):
    order = np.argsort(values)
    result = np.empty_like(order, dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def fine_centroid_distances(data_dir, checkpoint, hierarchy, workers):
    dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ]))
    model = ResNet18(100)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    captured = {}

    def hook(module, inputs, output):
        captured["feature"] = inputs[0].detach()

    handle = model.linear.register_forward_hook(hook)
    sums = torch.zeros(100, 512, dtype=torch.float64)
    counts = torch.zeros(100, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    try:
        with torch.no_grad():
            for images, targets in loader:
                model(images.cuda(non_blocking=True))
                features = captured["feature"].cpu().double()
                sums.index_add_(0, targets, features)
                counts.index_add_(0, targets, torch.ones_like(targets))
    finally:
        handle.remove()
    centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)

    superclass_centroids = F.normalize(torch.stack([
        centroids[hierarchy["coarse_to_fine"][str(coarse)]].mean(dim=0)
        for coarse in range(20)
    ]), dim=1)
    superclass_cosine = (1.0 - superclass_centroids @ superclass_centroids.T).clamp_min(0.0)

    distances = {}
    for coarse in range(20):
        fine_ids = hierarchy["coarse_to_fine"][str(coarse)]
        values = centroids[fine_ids]
        cosine = 1.0 - values @ values.T
        upper = cosine[torch.triu_indices(5, 5, offset=1).unbind()]
        intra = upper.mean().item()
        other_distances = superclass_cosine[coarse].clone()
        other_distances[coarse] = float("inf")
        nearest_distance, nearest_id = other_distances.min(dim=0)
        inter = nearest_distance.item()
        distances[coarse] = {
            "cosine_centroid_distance": intra,
            "cosine_centroid_distance_std": upper.std(unbiased=True).item(),
            "nearest_other_superclass_id": nearest_id.item(),
            "nearest_other_superclass_cosine_distance": inter,
            "rho_inter_over_intra": inter / max(intra, 1e-12),
            "fragmentation_over_margin": intra / max(inter, 1e-12),
        }
    return distances


def load_accuracy(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["best_top1"], {item["class_id"]: item["accuracy"] for item in payload["per_class"]}


def main():
    parser = argparse.ArgumentParser("Analyze superclass structure vs class-in-class gain")
    parser.add_argument("--fine-data", required=True)
    parser.add_argument("--fine-teacher", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--per-class-dir", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-partition-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.mapping).open(encoding="utf-8") as handle:
        hierarchy = json.load(handle)
    distances = fine_centroid_distances(args.fine_data, args.fine_teacher, hierarchy, args.workers)

    baseline_by_seed, oracle_by_seed, random_by_seed = {}, {}, {}
    coarse_target_by_seed, random_coarse_target_by_seed, overall = {}, {}, []
    for seed in args.recovery_seeds:
        baseline_top1, baseline = load_accuracy(Path(args.per_class_dir) / f"baseline_seed{seed}.json")
        oracle_top1, oracle = load_accuracy(Path(args.per_class_dir) / f"oracle_seed{seed}.json")
        random_top1, random_arm = load_accuracy(
            Path(args.per_class_dir) / f"random_pseed{args.random_partition_seed}_seed{seed}.json"
        )
        coarse_target_top1, coarse_target = load_accuracy(
            Path(args.per_class_dir) / f"coarse_target_seed{seed}.json"
        )
        random_coarse_target_top1, random_coarse_target = load_accuracy(
            Path(args.per_class_dir)
            / f"random_coarse_target_pseed{args.random_partition_seed}_seed{seed}.json"
        )
        baseline_by_seed[seed], oracle_by_seed[seed] = baseline, oracle
        random_by_seed[seed] = random_arm
        coarse_target_by_seed[seed] = coarse_target
        random_coarse_target_by_seed[seed] = random_coarse_target
        oracle_gain = oracle_top1 - baseline_top1
        random_gain = random_top1 - baseline_top1
        overall.append({
            "seed": seed,
            "baseline_top1": baseline_top1,
            "random_top1": random_top1,
            "coarse_target_top1": coarse_target_top1,
            "random_coarse_target_top1": random_coarse_target_top1,
            "oracle_top1": oracle_top1,
            "random_gain": random_gain,
            "oracle_gain": oracle_gain,
            "oracle_vs_random": oracle_top1 - random_top1,
            "coarse_target_gain": coarse_target_top1 - baseline_top1,
            "oracle_vs_coarse_target": oracle_top1 - coarse_target_top1,
            "random_vs_random_coarse_target": random_top1 - random_coarse_target_top1,
            "fine_vs_random_teacher_at_coarse_target": (
                coarse_target_top1 - random_coarse_target_top1
            ),
            "difference_in_differences": (
                (oracle_top1 - coarse_target_top1)
                - (random_top1 - random_coarse_target_top1)
            ),
            "gain": oracle_gain,
        })

    rows = []
    for coarse in range(20):
        baseline_values = [baseline_by_seed[seed][coarse] for seed in args.recovery_seeds]
        random_values = [random_by_seed[seed][coarse] for seed in args.recovery_seeds]
        coarse_target_values = [coarse_target_by_seed[seed][coarse] for seed in args.recovery_seeds]
        random_coarse_target_values = [
            random_coarse_target_by_seed[seed][coarse] for seed in args.recovery_seeds
        ]
        oracle_values = [oracle_by_seed[seed][coarse] for seed in args.recovery_seeds]
        oracle_gains = [oracle - baseline for oracle, baseline in zip(oracle_values, baseline_values)]
        random_gains = [random_arm - baseline for random_arm, baseline
                        in zip(random_values, baseline_values)]
        oracle_vs_random = [oracle - random_arm for oracle, random_arm
                            in zip(oracle_values, random_values)]
        coarse_target_gains = [value - baseline for value, baseline
                               in zip(coarse_target_values, baseline_values)]
        oracle_vs_coarse_target = [oracle - value for oracle, value
                                   in zip(oracle_values, coarse_target_values)]
        random_vs_random_coarse_target = [random_arm - value for random_arm, value
                                          in zip(random_values, random_coarse_target_values)]
        did_values = [fine_delta - random_delta for fine_delta, random_delta
                      in zip(oracle_vs_coarse_target, random_vs_random_coarse_target)]
        rows.append({
            "coarse_id": coarse,
            "coarse_name": hierarchy["coarse_names"][coarse],
            "fine_names": "|".join(hierarchy["fine_names"][fine]
                                   for fine in hierarchy["coarse_to_fine"][str(coarse)]),
            "fine_centroid_cosine_distance": distances[coarse]["cosine_centroid_distance"],
            "nearest_other_superclass_id": distances[coarse]["nearest_other_superclass_id"],
            "nearest_other_superclass_name": hierarchy["coarse_names"][
                distances[coarse]["nearest_other_superclass_id"]
            ],
            "nearest_other_superclass_cosine_distance": distances[coarse][
                "nearest_other_superclass_cosine_distance"
            ],
            "rho_inter_over_intra": distances[coarse]["rho_inter_over_intra"],
            "fragmentation_over_margin": distances[coarse]["fragmentation_over_margin"],
            "baseline_mean": float(np.mean(baseline_values)),
            "baseline_std": float(np.std(baseline_values, ddof=1)),
            "random_mean": float(np.mean(random_values)),
            "random_std": float(np.std(random_values, ddof=1)),
            "coarse_target_mean": float(np.mean(coarse_target_values)),
            "coarse_target_std": float(np.std(coarse_target_values, ddof=1)),
            "random_coarse_target_mean": float(np.mean(random_coarse_target_values)),
            "random_coarse_target_std": float(np.std(random_coarse_target_values, ddof=1)),
            "oracle_mean": float(np.mean(oracle_values)),
            "oracle_std": float(np.std(oracle_values, ddof=1)),
            "random_paired_gain_mean": float(np.mean(random_gains)),
            "random_paired_gain_std": float(np.std(random_gains, ddof=1)),
            "paired_gain_mean": float(np.mean(oracle_gains)),
            "paired_gain_std": float(np.std(oracle_gains, ddof=1)),
            "oracle_vs_random_mean": float(np.mean(oracle_vs_random)),
            "oracle_vs_random_std": float(np.std(oracle_vs_random, ddof=1)),
            "coarse_target_paired_gain_mean": float(np.mean(coarse_target_gains)),
            "coarse_target_paired_gain_std": float(np.std(coarse_target_gains, ddof=1)),
            "oracle_vs_coarse_target_mean": float(np.mean(oracle_vs_coarse_target)),
            "oracle_vs_coarse_target_std": float(np.std(oracle_vs_coarse_target, ddof=1)),
            "random_vs_random_coarse_target_mean": float(np.mean(random_vs_random_coarse_target)),
            "random_vs_random_coarse_target_std": float(np.std(random_vs_random_coarse_target, ddof=1)),
            "difference_in_differences_mean": float(np.mean(did_values)),
            "difference_in_differences_std": float(np.std(did_values, ddof=1)),
        })

    with (output / "superclass_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (output / "overall_paired_results.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    x = np.array([row["fine_centroid_cosine_distance"] for row in rows])
    y = np.array([row["paired_gain_mean"] for row in rows])
    yerr = np.array([row["paired_gain_std"] for row in rows])
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank(x), rank(y))[0, 1])
    random_y = np.array([row["random_paired_gain_mean"] for row in rows])
    random_yerr = np.array([row["random_paired_gain_std"] for row in rows])
    random_pearson = float(np.corrcoef(x, random_y)[0, 1])
    random_spearman = float(np.corrcoef(rank(x), rank(random_y))[0, 1])
    semantic_y = np.array([row["oracle_vs_random_mean"] for row in rows])
    semantic_yerr = np.array([row["oracle_vs_random_std"] for row in rows])
    semantic_pearson = float(np.corrcoef(x, semantic_y)[0, 1])
    semantic_spearman = float(np.corrcoef(rank(x), rank(semantic_y))[0, 1])
    rho = np.array([row["rho_inter_over_intra"] for row in rows])
    rho_oracle_pearson = float(np.corrcoef(rho, y)[0, 1])
    rho_oracle_spearman = float(np.corrcoef(rank(rho), rank(y))[0, 1])
    rho_random_pearson = float(np.corrcoef(rho, random_y)[0, 1])
    rho_random_spearman = float(np.corrcoef(rank(rho), rank(random_y))[0, 1])
    rho_semantic_pearson = float(np.corrcoef(rho, semantic_y)[0, 1])
    rho_semantic_spearman = float(np.corrcoef(rank(rho), rank(semantic_y))[0, 1])
    slope, intercept = np.polyfit(x, y, 1)

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, alpha=0.8)
    line_x = np.linspace(x.min(), x.max(), 100)
    axis.plot(line_x, slope * line_x + intercept, linestyle="--", color="tab:red")
    for row, x_value, y_value in zip(rows, x, y):
        axis.annotate(row["coarse_name"], (x_value, y_value), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean pairwise cosine distance among five fine-class centroids")
    axis.set_ylabel("Oracle class-in-class gain over coarse baseline (Top1 points)")
    axis.set_title(f"Superclass structure vs Oracle gain (Pearson={pearson:.3f}, Spearman={spearman:.3f})")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "fine_distance_vs_oracle_gain.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.errorbar(x - 0.001, y, yerr=yerr, fmt="o", capsize=3, alpha=0.75,
                  label="Oracle fine labels - Baseline")
    axis.errorbar(x + 0.001, random_y, yerr=random_yerr, fmt="s", capsize=3, alpha=0.75,
                  label="Random pseudo labels - Baseline")
    oracle_slope, oracle_intercept = np.polyfit(x, y, 1)
    random_slope, random_intercept = np.polyfit(x, random_y, 1)
    axis.plot(line_x, oracle_slope * line_x + oracle_intercept, linestyle="--", alpha=0.8)
    axis.plot(line_x, random_slope * line_x + random_intercept, linestyle="--", alpha=0.8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean pairwise cosine distance among five official fine-class centroids")
    axis.set_ylabel("Gain over coarse Baseline (Top1 points)")
    axis.set_title("Semantic Oracle versus random target decorrelation")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "fine_distance_vs_three_arm_gain.png", dpi=200)
    plt.close(fig)

    semantic_slope, semantic_intercept = np.polyfit(x, semantic_y, 1)
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.errorbar(x, semantic_y, yerr=semantic_yerr, fmt="o", capsize=3, alpha=0.8)
    axis.plot(line_x, semantic_slope * line_x + semantic_intercept,
              linestyle="--", color="tab:purple")
    axis.axhline(0, color="black", linewidth=0.8)
    for row, x_value, y_value in zip(rows, x, semantic_y):
        axis.annotate(row["coarse_name"], (x_value, y_value), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    axis.set_xlabel("Mean pairwise cosine distance among five official fine-class centroids")
    axis.set_ylabel("Oracle semantic residual over Random (Top1 points)")
    axis.set_title(
        f"Fine structure vs semantic residual (Pearson={semantic_pearson:.3f}, "
        f"Spearman={semantic_spearman:.3f})"
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "fine_distance_vs_semantic_residual.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].errorbar(rho, y, yerr=yerr, fmt="o", capsize=3, alpha=0.75,
                     label="Oracle - Baseline")
    axes[0].errorbar(rho, random_y, yerr=random_yerr, fmt="s", capsize=3, alpha=0.75,
                     label="Random - Baseline")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$\rho$: nearest inter-superclass distance / within-superclass dispersion")
    axes[0].set_ylabel("Gain over Baseline (Top1 points)")
    axes[0].set_title(
        f"Total gains vs relative margin\n"
        f"Oracle Pearson={rho_oracle_pearson:.3f}; Random Pearson={rho_random_pearson:.3f}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].errorbar(rho, semantic_y, yerr=semantic_yerr, fmt="o", capsize=3, alpha=0.8)
    rho_line = np.linspace(rho.min(), rho.max(), 100)
    rho_slope, rho_intercept = np.polyfit(rho, semantic_y, 1)
    axes[1].plot(rho_line, rho_slope * rho_line + rho_intercept,
                 linestyle="--", color="tab:purple")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for row, x_value, y_value in zip(rows, rho, semantic_y):
        axes[1].annotate(row["coarse_name"], (x_value, y_value), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
    axes[1].set_xlabel(r"$\rho$: nearest inter-superclass distance / within-superclass dispersion")
    axes[1].set_ylabel("Oracle - Random semantic residual (Top1 points)")
    axes[1].set_title(
        f"Semantic residual vs relative margin\n"
        f"Pearson={rho_semantic_pearson:.3f}, Spearman={rho_semantic_spearman:.3f}"
    )
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "relative_margin_vs_gains.png", dpi=200)
    plt.close(fig)

    summary = {
        "recovery_seeds": args.recovery_seeds,
        "random_partition_seed": args.random_partition_seed,
        "overall": overall,
        "overall_gain_mean": float(np.mean([item["gain"] for item in overall])),
        "overall_gain_std": float(np.std([item["gain"] for item in overall], ddof=1)),
        "overall_random_gain_mean": float(np.mean([item["random_gain"] for item in overall])),
        "overall_random_gain_std": float(np.std([item["random_gain"] for item in overall], ddof=1)),
        "overall_oracle_vs_random_mean": float(np.mean([item["oracle_vs_random"] for item in overall])),
        "overall_oracle_vs_random_std": float(np.std([item["oracle_vs_random"] for item in overall], ddof=1)),
        "overall_coarse_target_gain_mean": float(np.mean([item["coarse_target_gain"] for item in overall])),
        "overall_coarse_target_gain_std": float(np.std([item["coarse_target_gain"] for item in overall], ddof=1)),
        "overall_oracle_vs_coarse_target_mean": float(np.mean([item["oracle_vs_coarse_target"] for item in overall])),
        "overall_oracle_vs_coarse_target_std": float(np.std([item["oracle_vs_coarse_target"] for item in overall], ddof=1)),
        "overall_random_vs_random_coarse_target_mean": float(np.mean([
            item["random_vs_random_coarse_target"] for item in overall
        ])),
        "overall_random_vs_random_coarse_target_std": float(np.std([
            item["random_vs_random_coarse_target"] for item in overall
        ], ddof=1)),
        "overall_fine_vs_random_teacher_at_coarse_target_mean": float(np.mean([
            item["fine_vs_random_teacher_at_coarse_target"] for item in overall
        ])),
        "overall_fine_vs_random_teacher_at_coarse_target_std": float(np.std([
            item["fine_vs_random_teacher_at_coarse_target"] for item in overall
        ], ddof=1)),
        "overall_difference_in_differences_mean": float(np.mean([
            item["difference_in_differences"] for item in overall
        ])),
        "overall_difference_in_differences_std": float(np.std([
            item["difference_in_differences"] for item in overall
        ], ddof=1)),
        "pearson_distance_gain": pearson,
        "spearman_distance_gain": spearman,
        "pearson_distance_random_gain": random_pearson,
        "spearman_distance_random_gain": random_spearman,
        "pearson_distance_oracle_vs_random": semantic_pearson,
        "spearman_distance_oracle_vs_random": semantic_spearman,
        "relative_margin_definition": (
            "rho = nearest cosine distance to another normalized superclass centroid / "
            "mean pairwise cosine distance among the five normalized fine-class centroids"
        ),
        "pearson_rho_oracle_gain": rho_oracle_pearson,
        "spearman_rho_oracle_gain": rho_oracle_spearman,
        "pearson_rho_random_gain": rho_random_pearson,
        "spearman_rho_random_gain": rho_random_spearman,
        "pearson_rho_oracle_vs_random": rho_semantic_pearson,
        "spearman_rho_oracle_vs_random": rho_semantic_spearman,
        "feature_distance": "mean pairwise cosine distance among five normalized fine-class centroids",
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Scatter: {output / 'fine_distance_vs_oracle_gain.png'}")


if __name__ == "__main__":
    main()
