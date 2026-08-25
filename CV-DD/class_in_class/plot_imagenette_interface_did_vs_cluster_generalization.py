import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        "Plot interface DiD against Cluster Teacher excess native accuracy"
    )
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--interface-summary", required=True)
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    root = Path(args.cluster_root)
    interface = json.loads(Path(args.interface_summary).read_text(encoding="utf-8"))
    points = []
    for c in args.c_values:
        teacher_x = []
        teacher_native = []
        for teacher_seed in args.teacher_seeds:
            path = (
                root / f"tseed{teacher_seed}" / "audits"
                / f"dinov2_cluster_c{c}_teacher_audit.json"
            )
            audit = json.loads(path.read_text(encoding="utf-8"))
            if int(audit["subclasses_per_coarse"]) != c:
                raise ValueError(f"C mismatch: {path}")
            if int(audit["val"]["images"]) != 3925:
                raise ValueError(f"test split mismatch: {path}")
            conditional = float(
                audit["val"]["conditional_native_given_coarse_correct"]
            )
            teacher_native.append(conditional)
            teacher_x.append(100.0 * (conditional - 1.0 / c))

        effect = interface["comparisons"][f"C{c}"][
            "interface_difference_in_differences"
        ]
        points.append({
            "C": c,
            "expected_chance_conditional_native": 1.0 / c,
            "cluster_conditional_native_mean": float(np.mean(teacher_native)),
            "cluster_conditional_native_teacher_sd": sample_sd(teacher_native),
            "x_excess_native_percentage_points": float(np.mean(teacher_x)),
            "x_teacher_seed_sd_percentage_points": sample_sd(teacher_x),
            "y_interface_did_top1_points": float(effect["grand_mean"]),
            "y_cell_sd": float(effect["sample_sd_across_cells_descriptive"]),
            "y_student_sd": float(
                effect["pooled_within_teacher_recovery_student_seed_sd"]
            ),
            "y_recovery_sd": float(
                effect["pooled_within_teacher_recovery_seed_sd_of_student_means"]
            ),
            "y_teacher_seed_sd": float(
                effect["teacher_seed_sd_of_recovery_student_means"]
            ),
            "teacher_x_excess_percentage_points": teacher_x,
        })

    x = np.array([point["x_excess_native_percentage_points"] for point in points])
    y = np.array([point["y_interface_did_top1_points"] for point in points])
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    pearson_r, pearson_p = float(pearson[0]), float(pearson[1])
    spearman_r, spearman_p = float(spearman[0]), float(spearman[1])
    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(x, y)
    order = np.argsort(x)
    ordered_y = y[order]
    # Prediction: larger x -> smaller y. Positive increments are violations.
    monotonic_violations = int(np.sum(np.diff(ordered_y) > 0))

    result = {
        "definition": {
            "x": (
                "100 * (Cluster Teacher P(native correct | coarse correct) - 1/C); "
                "mean across Teacher seeds"
            ),
            "y": (
                "interface DiD = (Cluster Soft - Cluster Hard) - "
                "(Random Soft - Random Hard)"
            ),
            "x_error": "sample SD across Cluster Teacher seeds",
            "y_error": "Teacher-seed SD of paired interface DiD means",
        },
        "points": points,
        "statistics": {
            "n_C": len(points),
            "pearson_r": pearson_r,
            "pearson_two_sided_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_two_sided_p": spearman_p,
            "ols_slope_y_per_x_percentage_point": float(slope),
            "ols_intercept": float(intercept),
            "ols_r_squared": float(r_value ** 2),
            "ols_two_sided_p": float(p_value),
            "ols_slope_standard_error": float(slope_stderr),
            "monotonic_decrease_adjacent_violations_after_sorting_x": monotonic_violations,
        },
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "C", "x_excess_native_percentage_points",
            "x_teacher_seed_sd_percentage_points", "y_interface_did_top1_points",
            "y_teacher_seed_sd", "cluster_conditional_native_mean",
            "expected_chance_conditional_native",
        ])
        writer.writeheader()
        for point in points:
            writer.writerow({key: point[key] for key in writer.fieldnames})

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    xerr = np.array([point["x_teacher_seed_sd_percentage_points"] for point in points])
    yerr = np.array([point["y_teacher_seed_sd"] for point in points])
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=7,
        capsize=3, linewidth=1.2, label="C-level mean ± Teacher-seed SD",
    )
    grid = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(
        grid, intercept + slope * grid, linewidth=1.5,
        label=f"OLS: slope={slope:.3f}, $R^2$={r_value ** 2:.3f}",
    )
    for point in points:
        ax.annotate(
            f"C={point['C']}",
            (point["x_excess_native_percentage_points"], point["y_interface_did_top1_points"]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )
    ax.axhline(0, color="0.45", linewidth=1, linestyle="--")
    ax.axvline(0, color="0.45", linewidth=1, linestyle=":")
    ax.set_xlabel("Cluster Teacher excess native accuracy over 1/C (percentage points)")
    ax.set_ylabel("Interface DiD (Top-1 points)")
    ax.set_title(
        "Subclass generalization vs Cluster–Random interface DiD\n"
        f"Pearson r={pearson_r:.3f}; Spearman ρ={spearman_r:.3f}"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(f"Saved: {prefix}.json/.csv/.png/.pdf")


if __name__ == "__main__":
    main()
