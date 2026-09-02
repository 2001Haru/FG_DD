import argparse
import json
import math
import statistics
from pathlib import Path


RSEEDS = (41, 42)
SSEEDS = (42, 43, 44)
FAMILIES = {
    "c1_e300": "c1_e300_e299_ref",
    "c100_e100": "c100_e100_e099_ref",
}
PROTOCOLS = {
    "constantS_T1": {
        "student_temperature": 1.0,
        "constant_total_trace": True,
        "alphas": (0.70, 0.85, 1.00, 1.20, 1.50, 1.80),
    },
    "rawS_T20": {
        "student_temperature": 20.0,
        "constant_total_trace": False,
        "alphas": (1.00, 1.20, 1.50, 1.80),
    },
}


def tag(alpha):
    return f"{alpha:.3f}".replace(".", "p")


def load_best(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation set: {path}")
    if payload.get("training_target") != "fkd_soft_label":
        raise ValueError(f"invalid training target: {path}")
    return float(payload["best_top1"])


def summarize(values):
    all_values = list(values.values())
    recovery_means = []
    within_sd = []
    for recovery in RSEEDS:
        selected = [value for (r, _), value in values.items() if r == recovery]
        recovery_means.append(statistics.mean(selected))
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


def paired(left, right):
    if set(left) != set(right):
        raise ValueError("paired cell keys do not match")
    return summarize({key: left[key] - right[key] for key in left})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--old-alpha-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiment = Path(args.experiment_root)
    old_alpha = Path(args.old_alpha_root)
    existing = Path(args.existing_root) / "tseed43" / "per_class"
    arms = {}
    comparisons = {}
    values_by_arm = {}
    audits = {}

    for protocol, protocol_config in PROTOCOLS.items():
        for family, existing_tag in FAMILIES.items():
            family_values = {}
            for alpha in protocol_config["alphas"]:
                values = {}
                for recovery in RSEEDS:
                    for student in SSEEDS:
                        path = (
                            experiment / "per_class" / protocol
                            / f"{family}_alpha{tag(alpha)}_rseed{recovery}_sseed{student}.json"
                        )
                        values[(recovery, student)] = load_best(path)
                key = f"{protocol}__{family}__alpha{tag(alpha)}"
                values_by_arm[key] = values
                arms[key] = summarize(values)
                family_values[alpha] = values

            base = family_values[1.0]
            for alpha, values in family_values.items():
                key = f"{protocol}__{family}__alpha{tag(alpha)}_minus_alpha1"
                comparisons[key] = paired(values, base)

            audit_rows = []
            for recovery in RSEEDS:
                path = (
                    experiment / "fkd" / protocol / f"{family}_rseed{recovery}"
                    / "alpha_transform_summary.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                audit_rows.append(
                    {
                        "recovery_seed": recovery,
                        "base_R": payload["base_normalized_decomposition"][
                            "R_within_over_between"
                        ],
                        "global_centered_logit_sd": payload[
                            "global_centered_logit_sd"
                        ],
                        "output_scale": payload["output_scale"],
                        "alpha1_softmax_replay": payload["alpha1_softmax_replay"],
                        "alpha_rows": payload["alpha_rows"],
                    }
                )
            audits[f"{protocol}__{family}"] = audit_rows

            # alpha=1 under restored T=20 should replay the original trajectory
            # pipeline. That earlier grid has only student seeds 42 and 43.
            if protocol == "rawS_T20":
                overlap = {}
                for recovery in RSEEDS:
                    for student in (42, 43):
                        old = load_best(
                            existing
                            / f"real__{existing_tag}_rseed{recovery}_sseed{student}.json"
                        )
                        overlap[(recovery, student)] = base[(recovery, student)] - old
                comparisons[f"{protocol}__{family}__alpha1_replay_minus_original"] = {
                    "overlapping_cells": len(overlap),
                    "mean": statistics.mean(overlap.values()),
                    "maximum_absolute_difference": max(abs(v) for v in overlap.values()),
                    "by_cell": {f"r{r}_s{s}": v for (r, s), v in overlap.items()},
                }

    # Gate 1: constant-S versus the already completed unnormalized T=1 arms.
    for family in FAMILIES:
        for alpha in PROTOCOLS["constantS_T1"]["alphas"]:
            old_values = {}
            for recovery in RSEEDS:
                for student in SSEEDS:
                    old_values[(recovery, student)] = load_best(
                        old_alpha / "per_class"
                        / f"{family}_alpha{tag(alpha)}_rseed{recovery}_sseed{student}.json"
                    )
            new_key = f"constantS_T1__{family}__alpha{tag(alpha)}"
            comparisons[f"{new_key}_minus_rawS_T1"] = paired(
                values_by_arm[new_key], old_values
            )

    # Gate 2: restored T=20 versus the earlier T=1 protocol at equal raw alpha.
    for family in FAMILIES:
        for alpha in PROTOCOLS["rawS_T20"]["alphas"]:
            old_values = {}
            for recovery in RSEEDS:
                for student in SSEEDS:
                    old_values[(recovery, student)] = load_best(
                        old_alpha / "per_class"
                        / f"{family}_alpha{tag(alpha)}_rseed{recovery}_sseed{student}.json"
                    )
            new_key = f"rawS_T20__{family}__alpha{tag(alpha)}"
            comparisons[f"{new_key}_minus_rawS_T1"] = paired(
                values_by_arm[new_key], old_values
            )

    result = {
        "audit_schema_version": 1,
        "protocol": {
            "dataset": "ImageNette IPC10 random-real sources only",
            "teacher_seed": 43,
            "recovery_subset_seeds": list(RSEEDS),
            "student_seeds": list(SSEEDS),
            "source_label_temperature": 20.0,
            "class_assignment": "fixed pre-transform parent-logit argmax",
            "controls": PROTOCOLS,
            "constant_S_definition": "S=trace(S_B)+trace(S_W)",
        },
        "arms": arms,
        "paired_comparisons": comparisons,
        "transform_audits": audits,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
