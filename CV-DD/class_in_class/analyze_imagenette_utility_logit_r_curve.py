import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import f as f_distribution
from scipy.stats import t as student_t

from audit_imagenette_best_teacher_channels import variance_summary
from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


SEEDS = (43, 44)
RSEEDS = (41, 42)
SSEEDS = (42, 43)
EPOCHS = (4, 8, 16, 32, 64, 100, 150, 200, 250, 300)
TEMPERATURE = 20.0


def load_best(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation result: {path}")
    return float(payload["best_top1"])


def equivalent_logits(probabilities):
    logits = TEMPERATURE * probabilities.double().clamp_min(1e-30).log()
    return logits - logits.mean(1, keepdim=True)


def fit_model(rows, predictors, name):
    y = np.asarray([row["dd_utility"] for row in rows], dtype=float)
    x = np.column_stack([
        np.ones(len(rows)),
        *[np.asarray([row[predictor] for row in rows]) for predictor in predictors],
    ])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    rank = int(np.linalg.matrix_rank(x))
    df = len(rows) - rank
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    sigma2 = sse / df
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    standard_error = np.sqrt(np.diag(covariance))
    statistic = beta / standard_error
    p_values = 2 * student_t.sf(np.abs(statistic), df)
    k = rank
    aic = len(rows) * math.log(max(sse / len(rows), 1e-30)) + 2 * k
    bic = len(rows) * math.log(max(sse / len(rows), 1e-30)) + k * math.log(len(rows))
    names = ["intercept", *predictors]
    result = {
        "name": name,
        "n": len(rows),
        "rank": rank,
        "df_residual": df,
        "sse": sse,
        "rmse": math.sqrt(sse / len(rows)),
        "residual_sd": math.sqrt(sigma2),
        "r_squared": 1 - sse / sst,
        "adjusted_r_squared": 1 - (1 - (1 - sse / sst)) * (len(rows) - 1) / df,
        "aic": aic,
        "bic": bic,
        "coefficients": {
            parameter: {
                "estimate": float(beta[index]),
                "standard_error": float(standard_error[index]),
                "t": float(statistic[index]),
                "two_sided_p": float(p_values[index]),
            }
            for index, parameter in enumerate(names)
        },
        "fitted": fitted.tolist(),
        "residuals": residual.tolist(),
    }
    if "R" in predictors and "R2" in predictors:
        linear = result["coefficients"]["R"]["estimate"]
        quadratic = result["coefficients"]["R2"]["estimate"]
        result["quadratic_vertex_R"] = -linear / (2 * quadratic)
        result["quadratic_is_concave"] = quadratic < 0
    return result


def nested_f_test(reduced, full):
    numerator_df = reduced["df_residual"] - full["df_residual"]
    denominator_df = full["df_residual"]
    if numerator_df <= 0:
        raise ValueError("models are not nested in the expected direction")
    numerator = (reduced["sse"] - full["sse"]) / numerator_df
    denominator = full["sse"] / denominator_df
    statistic = numerator / denominator
    return {
        "F": statistic,
        "df_numerator": numerator_df,
        "df_denominator": denominator_df,
        "p": float(f_distribution.sf(statistic, numerator_df, denominator_df)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-collected", required=True)
    parser.add_argument("--c100-collected", required=True)
    parser.add_argument("--downstream-root", required=True)
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    collected = {
        "C1": torch.load(args.c1_collected, map_location="cpu", weights_only=False),
        "C100": torch.load(args.c100_collected, map_location="cpu", weights_only=False),
    }
    reference = collected["C1"]
    for family, payload in collected.items():
        if payload["epochs"] != list(EPOCHS) or payload["images"] != 3925:
            raise ValueError(f"invalid collection: {family}")
        if payload["sample_paths"] != reference["sample_paths"]:
            raise ValueError("sample order mismatch")
        if not torch.equal(payload["targets"], reference["targets"]):
            raise ValueError("target mismatch")
    targets = reference["targets"].long()
    downstream = Path(args.downstream_root)
    factorial = Path(args.factorial_root)
    random_root = Path(args.random_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def soft_path(seed, c, epoch, source, recovery, student):
        return (
            downstream / f"tseed{seed}" / "per_class"
            / f"{source}__c{c}_e{epoch:03d}_e{epoch - 1:03d}_ref_rseed{recovery}_sseed{student}.json"
        )

    def hard_path(seed, source, recovery, student):
        if source == "real":
            return (
                factorial / f"tseed{seed}" / "per_class"
                / f"real__hard_rseed{recovery}_sseed{student}.json"
            )
        return (
            random_root / f"tseed{seed}" / "hard_per_class"
            / f"c1_rseed{recovery}_sseed{student}.json"
        )

    rows = []
    utility_details = {}
    for family, c in (("C1", 1), ("C100", 100)):
        payload = collected[family]
        for epoch in EPOCHS:
            source_values = {"real": {}, "c1": {}}
            averaged = {}
            for seed in SEEDS:
                for recovery in RSEEDS:
                    for student in SSEEDS:
                        key = (seed, recovery, student)
                        for source in ("real", "c1"):
                            soft = load_best(
                                soft_path(seed, c, epoch, source, recovery, student)
                            )
                            hard = load_best(
                                hard_path(seed, source, recovery, student)
                            )
                            source_values[source][key] = soft - hard
                        averaged[key] = 0.5 * (
                            source_values["real"][key] + source_values["c1"][key]
                        )
            utility_summary = {
                "real": three_level_summary(source_values["real"], SEEDS, RSEEDS, SSEEDS),
                "c1_synthetic": three_level_summary(source_values["c1"], SEEDS, RSEEDS, SSEEDS),
                "equal_source_average": three_level_summary(averaged, SEEDS, RSEEDS, SSEEDS),
            }
            utility_details[f"{family}_e{epoch:03d}"] = utility_summary
            seed_geometry = []
            for seed in SEEDS:
                q = payload["probabilities"][str(seed)][str(epoch)].double()
                seed_geometry.append(variance_summary(equivalent_logits(q), targets))
            mean_w = float(np.mean([item["within_trace"] for item in seed_geometry]))
            mean_b = float(np.mean([item["between_trace"] for item in seed_geometry]))
            r = mean_w / max(mean_b, 1e-30)
            rows.append({
                "family": family,
                "family_indicator": 0 if family == "C1" else 1,
                "epoch": epoch,
                "dd_utility": utility_summary["equal_source_average"]["grand_mean"],
                "utility_cell_sd": utility_summary["equal_source_average"][
                    "sample_sd_across_cells_descriptive"
                ],
                "R": r,
                "R2": r * r,
                "family_R": (0 if family == "C1" else 1) * r,
                "family_R2": (0 if family == "C1" else 1) * r * r,
                "W": mean_w,
                "B": mean_b,
                "mean_seed_R": float(np.mean([
                    item["R_within_over_between"] for item in seed_geometry
                ])),
            })

    models = {
        "pooled_linear": fit_model(rows, ["R"], "pooled_linear"),
        "pooled_quadratic": fit_model(rows, ["R", "R2"], "pooled_quadratic"),
        "family_intercept_linear": fit_model(
            rows, ["R", "family_indicator"], "family_intercept_linear"
        ),
        "family_interaction_linear": fit_model(
            rows, ["R", "family_indicator", "family_R"],
            "family_interaction_linear",
        ),
        "family_intercept_quadratic": fit_model(
            rows, ["R", "R2", "family_indicator"],
            "family_intercept_quadratic",
        ),
        "family_interaction_quadratic": fit_model(
            rows,
            ["R", "R2", "family_indicator", "family_R", "family_R2"],
            "family_interaction_quadratic",
        ),
        "C1_linear": fit_model(
            [row for row in rows if row["family"] == "C1"], ["R"], "C1_linear"
        ),
        "C100_linear": fit_model(
            [row for row in rows if row["family"] == "C100"], ["R"], "C100_linear"
        ),
    }
    tests = {
        "quadratic_vs_pooled_linear": nested_f_test(
            models["pooled_linear"], models["pooled_quadratic"]
        ),
        "family_intercept_added_to_quadratic": nested_f_test(
            models["pooled_quadratic"], models["family_intercept_quadratic"]
        ),
        "separate_family_quadratics_vs_pooled_quadratic": nested_f_test(
            models["pooled_quadratic"], models["family_interaction_quadratic"]
        ),
    }
    quadratic = models["pooled_quadratic"]
    vertex = quadratic["quadratic_vertex_R"]
    preregistration = {
        "single_peak_concave": quadratic["quadratic_is_concave"],
        "vertex_R": vertex,
        "vertex_in_preregistered_interval_0p63_0p70": 0.63 <= vertex <= 0.70,
        "pooled_quadratic_R2_exceeds_pooled_linear": (
            quadratic["r_squared"] > models["pooled_linear"]["r_squared"]
        ),
        "pooled_quadratic_adjusted_R2_exceeds_all_linear_models": (
            quadratic["adjusted_r_squared"]
            > max(
                models[name]["adjusted_r_squared"]
                for name in (
                    "pooled_linear", "family_intercept_linear",
                    "family_interaction_linear",
                )
            )
        ),
        "one_common_curve_supported_by_family_intercept_test_p_gt_0p05": (
            tests["family_intercept_added_to_quadratic"]["p"] > 0.05
        ),
        "one_common_curve_supported_by_separate_quadratics_test_p_gt_0p05": (
            tests["separate_family_quadratics_vs_pooled_quadratic"]["p"] > 0.05
        ),
    }

    figure, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    styles = {"C1": ("#2878B5", "o"), "C100": ("#C82423", "s")}
    for family in ("C1", "C100"):
        selected = [row for row in rows if row["family"] == family]
        color, marker = styles[family]
        axis.scatter(
            [row["R"] for row in selected],
            [row["dd_utility"] for row in selected],
            color=color, marker=marker, s=65, label=family, zorder=3,
        )
        for row in selected:
            axis.annotate(
                f'e{row["epoch"]}', (row["R"], row["dd_utility"]),
                fontsize=8, xytext=(4, 3), textcoords="offset points",
            )
    r_grid = np.linspace(min(row["R"] for row in rows), max(row["R"] for row in rows), 300)
    beta = quadratic["coefficients"]
    y_grid = (
        beta["intercept"]["estimate"]
        + beta["R"]["estimate"] * r_grid
        + beta["R2"]["estimate"] * r_grid ** 2
    )
    axis.plot(r_grid, y_grid, color="black", linewidth=2, label="pooled quadratic")
    axis.axvspan(0.63, 0.70, color="gray", alpha=0.10, label="preregistered vertex")
    axis.axvline(vertex, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Logit-space R = within trace / between trace")
    axis.set_ylabel("DD utility: T20 Soft − Hard Top-1")
    axis.set_title("ImageNette DD utility vs logit-space within/between ratio")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output / "dd_utility_vs_logit_R.png", dpi=220)
    figure.savefig(output / "dd_utility_vs_logit_R.pdf")
    plt.close(figure)

    with (output / "utility_logit_R_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "audit_schema_version": 1,
        "definition": {
            "R": "T20 equivalent-logit within trace / between trace",
            "utility": "strictly paired T20 Soft-Hard Top1, equal Real/C1-synthetic average",
            "extreme_arms_included": ["C1 e4/e8", "C100 e4/e8/e16/e32"],
        },
        "preregistered_predictions": {
            "shape": "single peak",
            "families": "C1/C100 lie on one common curve",
            "vertex_R_interval": [0.63, 0.70],
            "model": "pooled quadratic outperforms linear alternatives",
        },
        "points": rows,
        "utility_details": utility_details,
        "models": models,
        "nested_model_tests": tests,
        "preregistration_results": preregistration,
    }
    (output / "utility_logit_R_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "preregistration": preregistration,
        "quadratic": quadratic,
        "tests": tests,
    }, indent=2))


if __name__ == "__main__":
    main()
