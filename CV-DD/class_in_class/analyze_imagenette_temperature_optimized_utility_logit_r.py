import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

from analyze_imagenette_utility_logit_r_curve import (
    EPOCHS,
    RSEEDS,
    SEEDS,
    SSEEDS,
    equivalent_logits,
    fit_model,
    load_best,
    nested_f_test,
)
from audit_imagenette_best_teacher_channels import variance_summary
from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


FIXED_MODES = ("t8", "ref", "t46", "t100", "t200")


def residualize(values, covariates):
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    design = np.column_stack([np.ones(len(values)), covariates])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def partial_correlation(left, right, covariates):
    left_residual = residualize(left, covariates)
    right_residual = residualize(right, covariates)
    correlation = float(np.corrcoef(left_residual, right_residual)[0, 1])
    covariate_count = np.asarray(covariates).reshape(len(left), -1).shape[1]
    df = len(left) - covariate_count - 2
    statistic = correlation * math.sqrt(
        df / max(1 - correlation * correlation, 1e-30)
    )
    return {
        "partial_r": correlation,
        "t": statistic,
        "df": df,
        "two_sided_p": float(2 * student_t.sf(abs(statistic), df)),
    }


def rank(values):
    values = np.asarray(values)
    order = np.argsort(values)
    output = np.empty(len(values), dtype=float)
    output[order] = np.arange(len(values), dtype=float)
    return output


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

    def soft_path(seed, c, epoch, source, mode, recovery, student):
        return (
            downstream / f"tseed{seed}" / "per_class"
            / f"{source}__c{c}_e{epoch:03d}_e{epoch - 1:03d}_{mode}_rseed{recovery}_sseed{student}.json"
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
    all_mode_summaries = {}
    for family, c in (("C1", 1), ("C100", 100)):
        payload = collected[family]
        for epoch in EPOCHS:
            modes = list(FIXED_MODES)
            if epoch != 4:
                modes.append("pred")
            mode_summaries = {}
            for mode in modes:
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
                                source_values["real"][key]
                                + source_values["c1"][key]
                            )
                mode_summaries[mode] = {
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
            best_mode = max(
                modes,
                key=lambda mode: mode_summaries[mode]["equal_source_average"][
                    "grand_mean"
                ],
            )
            all_mode_summaries[f"{family}_e{epoch:03d}"] = {
                "selected_mode": best_mode,
                "modes": mode_summaries,
            }
            seed_geometry = []
            for seed in SEEDS:
                q = payload["probabilities"][str(seed)][str(epoch)].double()
                seed_geometry.append(variance_summary(equivalent_logits(q), targets))
            mean_w = float(np.mean([item["within_trace"] for item in seed_geometry]))
            mean_b = float(np.mean([item["between_trace"] for item in seed_geometry]))
            r_value = mean_w / max(mean_b, 1e-30)
            train_accuracy = float(np.mean([
                payload["trajectory_metrics"][str(seed)][str(epoch)][
                    "train_native_accuracy"
                ] for seed in SEEDS
            ]))
            coarse_accuracy = float(np.mean([
                payload["trajectory_metrics"][str(seed)][str(epoch)][
                    "val_coarse_accuracy"
                ] for seed in SEEDS
            ]))
            selected = mode_summaries[best_mode]["equal_source_average"]
            t20 = mode_summaries["ref"]["equal_source_average"]
            rows.append({
                "family": family,
                "family_indicator": 0 if family == "C1" else 1,
                "epoch": epoch,
                "log_epoch": math.log(epoch),
                "train_native_accuracy": train_accuracy,
                "coarse_accuracy": coarse_accuracy,
                "R": r_value,
                "best_mode": best_mode,
                "optimized_utility": selected["grand_mean"],
                "optimized_utility_cell_sd": selected[
                    "sample_sd_across_cells_descriptive"
                ],
                "T20_utility": t20["grand_mean"],
                "temperature_optimization_gain": (
                    selected["grand_mean"] - t20["grand_mean"]
                ),
            })

    mean_r = float(np.mean([row["R"] for row in rows]))
    mean_a = float(np.mean([row["coarse_accuracy"] for row in rows]))
    for row in rows:
        row["R_centered"] = row["R"] - mean_r
        row["R_centered2"] = row["R_centered"] ** 2
        row["A_centered"] = row["coarse_accuracy"] - mean_a
        row["family_R_centered"] = row["family_indicator"] * row["R_centered"]
        row["family_R_centered2"] = row["family_indicator"] * row["R_centered2"]
        row["dd_utility"] = row["optimized_utility"]

    models = {
        "gate_only": fit_model(rows, ["A_centered"], "gate_only"),
        "H0_gate_plus_common_R_quadratic": fit_model(
            rows, ["A_centered", "R_centered", "R_centered2"],
            "H0_gate_plus_common_R_quadratic",
        ),
        "H1_add_family_and_R_interactions": fit_model(
            rows,
            [
                "A_centered", "R_centered", "R_centered2",
                "family_indicator", "family_R_centered",
                "family_R_centered2",
            ],
            "H1_add_family_and_R_interactions",
        ),
        "pooled_optimized_linear_R": fit_model(
            rows, ["R_centered"], "pooled_optimized_linear_R"
        ),
        "pooled_optimized_quadratic_R": fit_model(
            rows, ["R_centered", "R_centered2"],
            "pooled_optimized_quadratic_R",
        ),
    }
    tests = {
        "R_quadratic_added_beyond_gate": nested_f_test(
            models["gate_only"], models["H0_gate_plus_common_R_quadratic"]
        ),
        "family_terms_added_to_gate_plus_R_quadratic": nested_f_test(
            models["H0_gate_plus_common_R_quadratic"],
            models["H1_add_family_and_R_interactions"],
        ),
        "optimized_quadratic_vs_optimized_linear_R": nested_f_test(
            models["pooled_optimized_linear_R"],
            models["pooled_optimized_quadratic_R"],
        ),
    }

    partial = {}
    for family in ("C1", "C100"):
        selected = [row for row in rows if row["family"] == family]
        log_epoch = [row["log_epoch"] for row in selected]
        optimized = [row["optimized_utility"] for row in selected]
        t20 = [row["T20_utility"] for row in selected]
        r_values = [row["R"] for row in selected]
        train = [row["train_native_accuracy"] for row in selected]
        partial[family] = {
            "zero_order_optimized_utility_R": float(np.corrcoef(optimized, r_values)[0, 1]),
            "optimized_utility_R_given_log_epoch": partial_correlation(
                optimized, r_values, log_epoch
            ),
            "T20_utility_R_given_log_epoch": partial_correlation(
                t20, r_values, log_epoch
            ),
            "optimized_utility_train_accuracy_given_log_epoch": partial_correlation(
                optimized, train, log_epoch
            ),
            "spearman_style_optimized_utility_R_given_log_epoch": partial_correlation(
                rank(optimized), rank(r_values), rank(log_epoch)
            ),
        }

    h0 = models["H0_gate_plus_common_R_quadratic"]
    h1 = models["H1_add_family_and_R_interactions"]
    preregistration = {
        "family_drops_out_p_gt_0p05": (
            tests["family_terms_added_to_gate_plus_R_quadratic"]["p"] > 0.05
        ),
        "R_quadratic_adds_beyond_gate_p_lt_0p05": (
            tests["R_quadratic_added_beyond_gate"]["p"] < 0.05
        ),
        "H0_preferred_by_AIC": h0["aic"] < h1["aic"],
        "H0_preferred_by_BIC": h0["bic"] < h1["bic"],
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    styles = {"C1": ("#2878B5", "o"), "C100": ("#C82423", "s")}
    for family in ("C1", "C100"):
        selected = [row for row in rows if row["family"] == family]
        color, marker = styles[family]
        axes[0].scatter(
            [row["R"] for row in selected],
            [row["optimized_utility"] for row in selected],
            color=color, marker=marker, s=65, label=family,
        )
        for row in selected:
            axes[0].annotate(
                f'e{row["epoch"]}', (row["R"], row["optimized_utility"]),
                fontsize=8, xytext=(4, 3), textcoords="offset points",
            )
    axes[0].set_xlabel("Logit-space R")
    axes[0].set_ylabel("Temperature-optimized DD utility")
    axes[0].set_title("Optimized utility vs R")
    axes[0].grid(alpha=0.22); axes[0].legend(frameon=False)
    observed = np.asarray([row["optimized_utility"] for row in rows])
    axes[1].scatter(observed, h0["fitted"], label="H0", alpha=0.8)
    axes[1].scatter(observed, h1["fitted"], label="H1", alpha=0.8)
    bounds = [
        min(observed.min(), min(h0["fitted"]), min(h1["fitted"])),
        max(observed.max(), max(h0["fitted"]), max(h1["fitted"])),
    ]
    axes[1].plot(bounds, bounds, color="black", linestyle="--")
    axes[1].set_xlabel("Observed optimized utility")
    axes[1].set_ylabel("Fitted utility")
    axes[1].set_title("Gate+R surface with/without family terms")
    axes[1].grid(alpha=0.22); axes[1].legend(frameon=False)
    figure.savefig(output / "temperature_optimized_utility_logit_R.png", dpi=220)
    figure.savefig(output / "temperature_optimized_utility_logit_R.pdf")
    plt.close(figure)

    with (output / "temperature_optimized_utility_logit_R_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "audit_schema_version": 1,
        "definition": {
            "utility": (
                "per-checkpoint globally selected best mean utility among fixed "
                "T8/T20/T46/T100/T200 and available Tpred; selection is not per-cell"
            ),
            "R": "T20 equivalent-logit within/between trace ratio",
            "gate": "mean full-test coarse accuracy of the Teacher checkpoint",
        },
        "preregistered_test": {
            "H0": "utility ~ coarse_acc + R + R^2",
            "H1": "H0 + family + family:R + family:R^2",
        },
        "points": rows,
        "all_temperature_mode_summaries": all_mode_summaries,
        "models": models,
        "nested_tests": tests,
        "partial_correlations": partial,
        "preregistration_results": preregistration,
    }
    (output / "temperature_optimized_utility_logit_R_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "tests": tests,
        "partial_correlations": partial,
        "preregistration": preregistration,
    }, indent=2))


if __name__ == "__main__":
    main()
