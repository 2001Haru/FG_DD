import argparse
import json
import math
import statistics
from pathlib import Path


RSEEDS = (41, 42)
SSEEDS = (42, 43, 44)
FAMILY_CONFIG = {
    "c1_e300": {
        "alphas": (0.70, 0.85, 1.00, 1.075, 1.20, 1.50, 1.80),
        "existing_tag": "c1_e300_e299_ref",
        "historical_real_best": 72.838,
    },
    "c100_e100": {
        "alphas": (0.70, 0.85, 0.93, 1.00, 1.20, 1.50, 1.80),
        "existing_tag": "c100_e100_e099_ref",
        "historical_real_best": 72.777,
    },
}


def tag(alpha):
    return f"{alpha:.3f}".replace(".", "p")


def load_best(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid result: {path}")
    return float(payload["best_top1"])


def summary(values):
    all_values = list(values.values())
    recovery_means = [
        statistics.mean(value for (recovery, _), value in values.items() if recovery == r)
        for r in RSEEDS
    ]
    within_sd = []
    for recovery in RSEEDS:
        selected = [value for (r, _), value in values.items() if r == recovery]
        within_sd.append(statistics.stdev(selected))
    return {
        "cells": len(all_values),
        "grand_mean": statistics.mean(all_values),
        "sample_sd_across_cells": statistics.stdev(all_values),
        "pooled_within_recovery_student_sd": math.sqrt(
            statistics.mean(value * value for value in within_sd)
        ),
        "recovery_seed_sd_of_student_means": statistics.stdev(recovery_means),
        "by_recovery_seed": {
            str(recovery): {
                "mean": recovery_means[index],
                "student_sd": within_sd[index],
            }
            for index, recovery in enumerate(RSEEDS)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_root)
    existing = Path(args.existing_root) / "tseed43" / "per_class"
    arms = {}
    paired = {}
    transform_audits = {}
    replay = {}
    for family, config in FAMILY_CONFIG.items():
        family_values = {}
        for alpha in config["alphas"]:
            values = {}
            for recovery in RSEEDS:
                for student in SSEEDS:
                    path = (
                        experiment / "per_class"
                        / f"{family}_alpha{tag(alpha)}_rseed{recovery}_sseed{student}.json"
                    )
                    values[(recovery, student)] = load_best(path)
            family_values[alpha] = values
            arms[f"{family}_alpha{tag(alpha)}"] = summary(values)
        base = family_values[1.0]
        for alpha, values in family_values.items():
            deltas = {key: values[key] - base[key] for key in values}
            paired[f"{family}_alpha{tag(alpha)}_minus_alpha1"] = summary(deltas)
        best_alpha = max(
            config["alphas"],
            key=lambda alpha: arms[f"{family}_alpha{tag(alpha)}"]["grand_mean"],
        )
        paired[f"{family}_best_alpha"] = {
            "alpha": best_alpha,
            "top1": arms[f"{family}_alpha{tag(best_alpha)}"]["grand_mean"],
            "gain_over_alpha1": paired[
                f"{family}_alpha{tag(best_alpha)}_minus_alpha1"
            ]["grand_mean"],
            "remaining_gap_to_historical_real_best": (
                config["historical_real_best"]
                - arms[f"{family}_alpha{tag(best_alpha)}"]["grand_mean"]
            ),
        }
        replay_values = {}
        for recovery in RSEEDS:
            for student in (42, 43):
                transformed = family_values[1.0][(recovery, student)]
                original = load_best(
                    existing
                    / f"real__{config['existing_tag']}_rseed{recovery}_sseed{student}.json"
                )
                replay_values[(recovery, student)] = transformed - original
        replay[family] = {
            "overlapping_cells": len(replay_values),
            "mean_top1_difference": statistics.mean(replay_values.values()),
            "maximum_absolute_top1_difference": max(
                abs(value) for value in replay_values.values()
            ),
            "differences": {
                f"r{r}_s{s}": value for (r, s), value in replay_values.items()
            },
        }
        audits = []
        for recovery in RSEEDS:
            path = (
                experiment / "fkd" / f"{family}_rseed{recovery}"
                / "alpha_transform_summary.json"
            )
            audits.append(json.loads(path.read_text(encoding="utf-8")))
        transform_audits[family] = audits
    result = {
        "audit_schema_version": 1,
        "protocol": {
            "dataset": "ImageNette IPC10 random-real sources only",
            "teacher_seed": 43,
            "recovery_subset_seeds": list(RSEEDS),
            "student_seeds": list(SSEEDS),
            "student_temperature": 1.0,
            "source_label_temperature": 20.0,
            "sigma_policy": "global centered-logit sd / source temperature",
            "class_assignment": "fixed pre-transform parent-logit argmax",
        },
        "arms": arms,
        "paired_comparisons": paired,
        "alpha1_replay_vs_original_T20": replay,
        "transform_audits": transform_audits,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
