import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_report(payload, relation):
    rows = [
        row for row in payload["logit_pair_reports"]
        if row["relation"] == relation
    ]
    if relation.startswith("within_"):
        if len(rows) != 1:
            raise ValueError(f"expected one {relation} row")
        return rows[0]
    return rows


def spectrum(report):
    return report["CCA"]["within_class_centered_labels"][
        "canonical_correlations"
    ]


def pearson(left, right):
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def rank(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def cca_ratio_rows(phase, payload):
    c1 = spectrum(find_report(payload, "within_c1"))
    c100 = spectrum(find_report(payload, "within_c100"))
    cross_reports = find_report(payload, "cross_family_same_seed")
    cross = [spectrum(row) for row in cross_reports]
    components = min(len(c1), len(c100), *(len(row) for row in cross))
    rows = []
    for index in range(components):
        ceiling = math.sqrt(c1[index] * c100[index])
        ratios = [row[index] / ceiling for row in cross]
        rows.append({
            "phase": phase,
            "component": index + 1,
            "c1_self_rho": c1[index],
            "c100_self_rho": c100[index],
            "geometric_self_ceiling": ceiling,
            "cross_seed43_rho": cross[0][index],
            "cross_seed44_rho": cross[1][index],
            "cross_mean_rho": statistics.mean(row[index] for row in cross),
            "ratio_seed43": ratios[0],
            "ratio_seed44": ratios[1],
            "ratio_mean": statistics.mean(ratios),
            "ratio_sample_sd": statistics.stdev(ratios),
        })
    ratios = [row["ratio_mean"] for row in rows]
    summary = {
        "phase": phase,
        "components": components,
        "strictly_monotonic_decreasing": all(
            ratios[index + 1] < ratios[index]
            for index in range(len(ratios) - 1)
        ),
        "decreasing_steps": sum(
            ratios[index + 1] < ratios[index]
            for index in range(len(ratios) - 1)
        ),
        "total_steps": len(ratios) - 1,
        "pearson_component_ratio": pearson(range(1, len(ratios) + 1), ratios),
        "spearman_component_ratio": pearson(
            range(1, len(ratios) + 1), rank(ratios)
        ),
        "first_ratio": ratios[0],
        "last_ratio": ratios[-1],
        "absolute_drop": ratios[0] - ratios[-1],
    }
    return rows, summary


def mean_model_value(payload, family, key):
    return statistics.mean(
        payload["logit_model_summaries"][f"{family}_s{seed}"][key]
        for seed in (43, 44)
    )


def equal_utility_summary(label, payload, utility_c1, utility_c100):
    corrected = payload["reliability_corrected_within_CKA"]
    logit = corrected["equivalent_logit_space"]
    probability = corrected["probability_space"]
    c1 = find_report(payload, "within_c1")
    c100 = find_report(payload, "within_c100")
    cross = find_report(payload, "cross_family_same_seed")
    c1_sum = c1["CCA"]["within_class_centered_labels"][
        "sum_squared_canonical_correlations"
    ]
    c100_sum = c100["CCA"]["within_class_centered_labels"][
        "sum_squared_canonical_correlations"
    ]
    cross_sum = statistics.mean(
        row["CCA"]["within_class_centered_labels"][
            "sum_squared_canonical_correlations"
        ] for row in cross
    )
    cca_ceiling = math.sqrt(c1_sum * c100_sum)
    cross_angle = statistics.mean(
        row["within_class_covariance_principal_angles"][
            "pooled_within_class_residuals"
        ]["top_eigenvector_angle_degrees"] for row in cross
    )
    return {
        "label": label,
        "selection_label": payload["protocol"].get("selection_label"),
        "C1": payload["protocol"]["C1"],
        "Random_C100": payload["protocol"]["Random_C100"],
        "reported_downstream_utility": {
            "C1": utility_c1,
            "Random_C100": utility_c100,
            "absolute_gap": abs(utility_c1 - utility_c100),
        },
        "probability_deattenuated_within_CKA": probability[
            "deattenuated_shared_fraction"
        ],
        "logit_cross_within_CKA": logit["cross_family_same_seed_CKA_mean"],
        "logit_reliability_ceiling": logit[
            "geometric_mean_reliability_ceiling"
        ],
        "logit_deattenuated_within_CKA": logit[
            "deattenuated_shared_fraction"
        ],
        "logit_C1_self_CCA_sum_rho2": c1_sum,
        "logit_C100_self_CCA_sum_rho2": c100_sum,
        "logit_cross_CCA_sum_rho2": cross_sum,
        "logit_CCA_ceiling": cca_ceiling,
        "logit_deattenuated_CCA_sum_rho2": cross_sum / cca_ceiling,
        "logit_cross_top1_direction_angle_degrees": cross_angle,
        "logit_C1_R": mean_model_value(payload, "c1", "R_within_over_between"),
        "logit_C100_R": mean_model_value(payload, "c100", "R_within_over_between"),
    }


def plot_ratios(rows, output_png, output_pdf):
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = {"e100": "#2878B5", "e300": "#C82423"}
    for phase in ("e100", "e300"):
        selected = [row for row in rows if row["phase"] == phase]
        x = [row["component"] for row in selected]
        y = [row["ratio_mean"] for row in selected]
        error = [row["ratio_sample_sd"] for row in selected]
        axis.errorbar(
            x, y, yerr=error, marker="o", linewidth=2, capsize=3,
            color=colors[phase], label=phase,
        )
    axis.axhline(1.0, color="0.45", linestyle="--", linewidth=1)
    axis.set_xticks(range(1, 10))
    axis.set_xlabel("Canonical component k")
    axis.set_ylabel(r"$\rho_k(cross) / \sqrt{\rho_k(C1\ self)\rho_k(C100\ self)}$")
    axis.set_title("Accuracy-corrected cross-family CCA spectrum")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output_png, dpi=220)
    figure.savefig(output_pdf)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-e100", required=True)
    parser.add_argument("--phase-e300", required=True)
    parser.add_argument("--equal-source-average", required=True)
    parser.add_argument("--equal-random-real", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase_payloads = {
        "e100": load(args.phase_e100),
        "e300": load(args.phase_e300),
    }
    ratio_rows = []
    monotonicity = {}
    for phase, payload in phase_payloads.items():
        rows, summary = cca_ratio_rows(phase, payload)
        ratio_rows.extend(rows)
        monotonicity[phase] = summary
    ratio_csv = output / "cca_component_self_normalized_ratio.csv"
    with ratio_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratio_rows[0]))
        writer.writeheader(); writer.writerows(ratio_rows)
    plot_ratios(
        ratio_rows,
        output / "cca_component_self_normalized_ratio.png",
        output / "cca_component_self_normalized_ratio.pdf",
    )
    equal_rows = [
        equal_utility_summary(
            "source_average_best",
            load(args.equal_source_average), 71.685, 72.143,
        ),
        equal_utility_summary(
            "random_real_best",
            load(args.equal_random_real), 72.838, 72.777,
        ),
    ]
    equal_csv = output / "equal_utility_logit_summary.csv"
    with equal_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "label", "c1_epoch", "c1_temperature", "c100_epoch",
            "c100_temperature", "utility_c1", "utility_c100", "utility_gap",
            "probability_shared_CKA", "logit_cross_CKA", "logit_CKA_ceiling",
            "logit_shared_CKA", "logit_shared_CCA", "logit_cross_angle",
            "logit_C1_R", "logit_C100_R",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in equal_rows:
            writer.writerow({
                "label": row["label"],
                "c1_epoch": row["C1"]["training_epoch"],
                "c1_temperature": row["C1"]["temperature"],
                "c100_epoch": row["Random_C100"]["training_epoch"],
                "c100_temperature": row["Random_C100"]["temperature"],
                "utility_c1": row["reported_downstream_utility"]["C1"],
                "utility_c100": row["reported_downstream_utility"]["Random_C100"],
                "utility_gap": row["reported_downstream_utility"]["absolute_gap"],
                "probability_shared_CKA": row["probability_deattenuated_within_CKA"],
                "logit_cross_CKA": row["logit_cross_within_CKA"],
                "logit_CKA_ceiling": row["logit_reliability_ceiling"],
                "logit_shared_CKA": row["logit_deattenuated_within_CKA"],
                "logit_shared_CCA": row["logit_deattenuated_CCA_sum_rho2"],
                "logit_cross_angle": row["logit_cross_top1_direction_angle_degrees"],
                "logit_C1_R": row["logit_C1_R"],
                "logit_C100_R": row["logit_C100_R"],
            })
    result = {
        "audit_schema_version": 1,
        "cca_ratio_definition": (
            "rho_k(cross) / sqrt(rho_k(C1-self) * rho_k(C100-self)); "
            "cross is averaged over same-seed pairs"
        ),
        "preregistered_prediction": "ratio strictly decreases with component k",
        "cca_ratio_monotonicity": monotonicity,
        "cca_ratio_rows": ratio_rows,
        "equal_utility_logit_comparisons": equal_rows,
    }
    (output / "cca_ratio_equal_utility_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output),
        "monotonicity": monotonicity,
        "equal_utility_pairs": len(equal_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
