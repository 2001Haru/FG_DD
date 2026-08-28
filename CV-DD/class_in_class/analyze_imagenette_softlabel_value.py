import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


SEEDS = (43, 44)
RSEEDS = (41, 42)
SSEEDS = (42, 43)
EPOCHS = (8, 16, 32, 64, 100, 150, 200, 250, 300)


def load_best(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation set: {path}")
    return float(payload["best_top1"])


def fit_ols(rows, predictors):
    y = np.asarray([row["dd_utility"] for row in rows], dtype=float)
    x = np.column_stack([
        np.ones(len(rows)),
        *[np.asarray([row[name] for row in rows], dtype=float) for name in predictors],
    ])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    rank = np.linalg.matrix_rank(x)
    df = len(y) - rank
    sigma2 = float(residual @ residual / df)
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.diag(covariance))
    statistic = beta / standard_error
    p_values = 2.0 * student_t.sf(np.abs(statistic), df)
    total = float(((y - y.mean()) ** 2).sum())
    residual_sum = float((residual ** 2).sum())
    r2 = 1.0 - residual_sum / total
    names = ["intercept", *predictors]
    return {
        "n": len(y), "rank": int(rank), "df_residual": int(df),
        "r_squared": r2,
        "adjusted_r_squared": 1.0 - (1.0 - r2) * (len(y) - 1) / df,
        "residual_sd": math.sqrt(sigma2),
        "coefficients": {
            name: {
                "estimate": float(beta[index]),
                "standard_error": float(standard_error[index]),
                "t": float(statistic[index]),
                "two_sided_p": float(p_values[index]),
            }
            for index, name in enumerate(names)
        },
        "fitted": fitted.tolist(),
        "residuals": residual.tolist(),
    }


def residualize(values, covariate):
    design = np.column_stack([np.ones(len(covariate)), covariate])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio-root", required=True)
    parser.add_argument("--downstream-root", required=True)
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ratio_root = Path(args.ratio_root)
    downstream_root = Path(args.downstream_root)
    factorial_root = Path(args.factorial_root)
    random_root = Path(args.random_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ratio = {}
    for seed in SEEDS:
        for c in (1, 100):
            payload = json.loads(
                (ratio_root / f"tseed{seed}" / f"c{c}" / "summary.json").read_text(encoding="utf-8")
            )
            for row in payload["rows"]:
                ratio[(seed, c, int(row["training_epoch"]))] = row

    def soft_path(seed, c, epoch, source, mode, recovery, student):
        checkpoint_index = epoch - 1
        return (
            downstream_root / f"tseed{seed}" / "per_class"
            / f"{source}__c{c}_e{epoch:03d}_e{checkpoint_index:03d}_{mode}_rseed{recovery}_sseed{student}.json"
        )

    def hard_path(seed, source, recovery, student):
        if source == "real":
            return (
                factorial_root / f"tseed{seed}" / "per_class"
                / f"real__hard_rseed{recovery}_sseed{student}.json"
            )
        return (
            random_root / f"tseed{seed}" / "hard_per_class"
            / f"c1_rseed{recovery}_sseed{student}.json"
        )

    utilities = {}
    utility_summaries = {}
    aggregate_rows, seed_rows = [], []
    for c in (1, 100):
        for epoch in EPOCHS:
            for mode in ("ref", "pred"):
                source_values = {"real": {}, "c1": {}}
                averaged = {}
                for seed in SEEDS:
                    for recovery in RSEEDS:
                        for student in SSEEDS:
                            key = (seed, recovery, student)
                            for source in ("real", "c1"):
                                soft = load_best(soft_path(
                                    seed, c, epoch, source, mode, recovery, student
                                ))
                                hard = load_best(hard_path(
                                    seed, source, recovery, student
                                ))
                                source_values[source][key] = soft - hard
                            averaged[key] = 0.5 * (
                                source_values["real"][key] + source_values["c1"][key]
                            )
                name = f"c{c}_epoch{epoch:03d}_{mode}"
                utilities[(c, epoch, mode)] = averaged
                utility_summaries[name] = {
                    "real": three_level_summary(
                        source_values["real"], SEEDS, RSEEDS, SSEEDS
                    ),
                    "c1_synthetic": three_level_summary(
                        source_values["c1"], SEEDS, RSEEDS, SSEEDS
                    ),
                    "equal_source_average": three_level_summary(
                        averaged, SEEDS, RSEEDS, SSEEDS
                    ),
                }

            t20_rows = [ratio[(seed, c, epoch)] for seed in SEEDS]
            r_values = [
                row["temperatures"]["T20"]["probability_vectors"][
                    "R_within_over_between"
                ] for row in t20_rows
            ]
            w_values = [
                row["temperatures"]["T20"]["probability_vectors"][
                    "within_trace"
                ] for row in t20_rows
            ]
            b_values = [
                row["temperatures"]["T20"]["probability_vectors"][
                    "between_trace"
                ] for row in t20_rows
            ]
            total_values = [
                row["temperatures"]["T20"]["probability_vectors"][
                    "total_trace"
                ] for row in t20_rows
            ]
            r_pred_values = [
                row["temperatures"]["predicted"]["probability_vectors"][
                    "R_within_over_between"
                ] for row in t20_rows
            ]
            w_pred_values = [
                row["temperatures"]["predicted"]["probability_vectors"][
                    "within_trace"
                ] for row in t20_rows
            ]
            b_pred_values = [
                row["temperatures"]["predicted"]["probability_vectors"][
                    "between_trace"
                ] for row in t20_rows
            ]
            r_logit_values = [
                row["temperatures"]["T20"]["centered_equivalent_logits"][
                    "R_within_over_between"
                ] for row in t20_rows
            ]
            accuracy_values = [row["trajectory_val_accuracy"] for row in t20_rows]
            utility = utility_summaries[f"c{c}_epoch{epoch:03d}_ref"][
                "equal_source_average"
            ]["grand_mean"]
            mean_w = float(np.mean(w_values))
            mean_b = float(np.mean(b_values))
            mean_w_pred = float(np.mean(w_pred_values))
            mean_b_pred = float(np.mean(b_pred_values))
            aggregate_r = mean_w / max(mean_b, 1e-30)
            aggregate_r_pred = mean_w_pred / max(mean_b_pred, 1e-30)
            aggregate_rows.append({
                "family": "C1" if c == 1 else "C100",
                "family_indicator": 0 if c == 1 else 1,
                "C": c, "epoch": epoch,
                "dd_utility": utility,
                "A_val_accuracy": float(np.mean(accuracy_values)),
                "R_T20": aggregate_r,
                "log_R_T20": math.log(max(aggregate_r, 1e-30)),
                "mean_of_seed_level_R_T20": float(np.mean(r_values)),
                "W_T20": mean_w,
                "B_T20": mean_b,
                "total_trace_T20": float(np.mean(total_values)),
                "log_W_T20": math.log(max(float(np.mean(w_values)), 1e-30)),
                "log_B_T20": math.log(max(float(np.mean(b_values)), 1e-30)),
                "R_predicted_temperature": aggregate_r_pred,
                "mean_of_seed_level_R_predicted_temperature": float(np.mean(r_pred_values)),
                "W_predicted_temperature": mean_w_pred,
                "B_predicted_temperature": mean_b_pred,
                "R_centered_logits_T20": float(np.mean(r_logit_values)),
                "relative_R_temperature_change": (
                    aggregate_r_pred / max(aggregate_r, 1e-30) - 1.0
                ),
            })
            for seed, ratio_row in zip(SEEDS, t20_rows):
                teacher_utilities = [
                    utilities[(c, epoch, "ref")][(seed, recovery, student)]
                    for recovery in RSEEDS for student in SSEEDS
                ]
                r_seed = ratio_row["temperatures"]["T20"]["probability_vectors"][
                    "R_within_over_between"
                ]
                seed_rows.append({
                    "teacher_seed": seed,
                    "family": "C1" if c == 1 else "C100",
                    "family_indicator": 0 if c == 1 else 1,
                    "C": c, "epoch": epoch,
                    "dd_utility": float(np.mean(teacher_utilities)),
                    "A_val_accuracy": ratio_row["trajectory_val_accuracy"],
                    "R_T20": r_seed,
                    "log_R_T20": math.log(max(r_seed, 1e-30)),
                    "W_T20": ratio_row["temperatures"]["T20"][
                        "probability_vectors"
                    ]["within_trace"],
                    "B_T20": ratio_row["temperatures"]["T20"][
                        "probability_vectors"
                    ]["between_trace"],
                    "log_W_T20": math.log(max(
                        ratio_row["temperatures"]["T20"]["probability_vectors"][
                            "within_trace"
                        ], 1e-30
                    )),
                    "log_B_T20": math.log(max(
                        ratio_row["temperatures"]["T20"]["probability_vectors"][
                            "between_trace"
                        ], 1e-30
                    )),
                })

    primary = fit_ols(aggregate_rows, ["A_val_accuracy", "log_R_T20"])
    family_adjusted = fit_ols(
        aggregate_rows, ["A_val_accuracy", "log_R_T20", "family_indicator"]
    )
    decomposed = fit_ols(
        aggregate_rows, ["A_val_accuracy", "log_B_T20", "log_W_T20"]
    )
    decomposed_family_adjusted = fit_ols(
        aggregate_rows,
        ["A_val_accuracy", "log_B_T20", "log_W_T20", "family_indicator"],
    )
    seed_fits = {
        str(seed): fit_ols(
            [row for row in seed_rows if row["teacher_seed"] == seed],
            ["A_val_accuracy", "log_R_T20"],
        ) for seed in SEEDS
    }
    correlation_A_logR = float(np.corrcoef(
        [row["A_val_accuracy"] for row in aggregate_rows],
        [row["log_R_T20"] for row in aggregate_rows],
    )[0, 1])

    y = np.asarray([row["dd_utility"] for row in aggregate_rows])
    a = np.asarray([row["A_val_accuracy"] for row in aggregate_rows])
    log_r = np.asarray([row["log_R_T20"] for row in aggregate_rows])
    partial_y = residualize(y, a)
    partial_r = residualize(log_r, a)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    colors = ["#31688e" if row["C"] == 1 else "#d1495b" for row in aggregate_rows]
    axes[0].scatter(partial_r, partial_y, c=colors, s=55, alpha=0.9)
    slope, intercept = np.polyfit(partial_r, partial_y, 1)
    grid = np.linspace(partial_r.min(), partial_r.max(), 100)
    axes[0].plot(grid, intercept + slope * grid, color="black", linestyle="--")
    for x_value, y_value, row in zip(partial_r, partial_y, aggregate_rows):
        axes[0].annotate(
            f'{row["family"]}-e{row["epoch"]}', (x_value, y_value),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("log R residual after removing coarse val accuracy A")
    axes[0].set_ylabel("DD utility residual after removing A")
    axes[0].set_title("Partial regression: DD utility vs log R | A")

    fitted = np.asarray(primary["fitted"])
    axes[1].scatter(y, fitted, c=colors, s=55, alpha=0.9)
    bounds = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    axes[1].plot(bounds, bounds, color="black", linestyle="--")
    axes[1].set_xlabel("Observed DD utility: Soft − Hard")
    axes[1].set_ylabel("Fitted DD utility")
    axes[1].set_title(f'Primary model, $R^2$={primary["r_squared"]:.3f}')
    figure.tight_layout()
    figure.savefig(output / "dd_utility_vs_variance_ratio.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    utility_values = [row["dd_utility"] for row in aggregate_rows]
    normalization = plt.Normalize(min(utility_values), max(utility_values))
    for family, color, marker in (
        ("C1", "#31688e", "o"), ("C100", "#d1495b", "s")
    ):
        selected = [row for row in aggregate_rows if row["family"] == family]
        axis.plot(
            [row["B_T20"] for row in selected],
            [row["W_T20"] for row in selected],
            color=color, linewidth=1.4, alpha=0.75,
        )
        scatter = axis.scatter(
            [row["B_T20"] for row in selected],
            [row["W_T20"] for row in selected],
            c=[row["dd_utility"] for row in selected], cmap="viridis",
            norm=normalization,
            marker=marker, s=75, edgecolors=color, linewidths=1.0,
            label=family,
        )
        for row in selected:
            axis.annotate(
                f'e{row["epoch"]}', (row["B_T20"], row["W_T20"]),
                fontsize=8, xytext=(4, 3), textcoords="offset points",
            )
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("B = trace(between-class covariance)")
    axis.set_ylabel("W = trace(within-class covariance)")
    axis.set_title("Absolute soft-label signal plane")
    axis.legend()
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap="viridis"), ax=axis
    )
    colorbar.set_label("DD utility: Soft − Hard Top-1")
    figure.tight_layout()
    figure.savefig(output / "softlabel_B_W_plane.png", dpi=200)
    plt.close(figure)

    result = {
        "definition": {
            "R": "trace(within-class covariance) / trace(between-class covariance)",
            "DD_utility": "strictly paired T20 soft-label Top1 minus hard-label Top1, equal average of Real and C1-synthetic sources",
            "A": "full ImageNette test coarse accuracy of the Teacher checkpoint",
            "primary_model": "DD_utility ~ intercept + A + log(R_T20)",
            "caution": "R is affine-scale invariant; softmax temperature invariance is approximate, not exact",
        },
        "teacher_seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "aggregate_18_points": aggregate_rows,
        "seed_level_36_points": seed_rows,
        "utility_summaries": utility_summaries,
        "regression": {
            "primary": primary,
            "family_adjusted": family_adjusted,
            "decomposed_B_W": decomposed,
            "decomposed_B_W_family_adjusted": decomposed_family_adjusted,
            "by_teacher_seed": seed_fits,
            "correlation_A_logR": correlation_A_logR,
        },
    }
    (output / "softlabel_value_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
