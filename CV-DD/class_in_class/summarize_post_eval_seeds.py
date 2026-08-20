import argparse
import csv
import json
import statistics
from pathlib import Path


ARMS = (
    "baseline",
    "fine_coarse_target",
    "random_coarse_target",
    "random_pseudo_target",
    "oracle_fine_target",
)

COMPARISONS = {
    "fine_coarse_minus_baseline": lambda x: x["fine_coarse_target"] - x["baseline"],
    "random_coarse_minus_baseline": lambda x: x["random_coarse_target"] - x["baseline"],
    "fine_minus_random_teacher_at_coarse_target": (
        lambda x: x["fine_coarse_target"] - x["random_coarse_target"]
    ),
    "fine_target_effect": lambda x: x["oracle_fine_target"] - x["fine_coarse_target"],
    "random_target_effect": lambda x: x["random_pseudo_target"] - x["random_coarse_target"],
    "difference_in_differences": lambda x: (
        x["oracle_fine_target"] - x["fine_coarse_target"]
        - x["random_pseudo_target"] + x["random_coarse_target"]
    ),
    "oracle_minus_baseline": lambda x: x["oracle_fine_target"] - x["baseline"],
    "random_minus_baseline": lambda x: x["random_pseudo_target"] - x["baseline"],
    "oracle_minus_random": lambda x: x["oracle_fine_target"] - x["random_pseudo_target"],
}


def legacy_filename(arm, recovery_seed, partition_seed):
    names = {
        "baseline": f"baseline_seed{recovery_seed}.json",
        "fine_coarse_target": f"coarse_target_seed{recovery_seed}.json",
        "random_coarse_target": (
            f"random_coarse_target_pseed{partition_seed}_seed{recovery_seed}.json"
        ),
        "random_pseudo_target": f"random_pseed{partition_seed}_seed{recovery_seed}.json",
        "oracle_fine_target": f"oracle_seed{recovery_seed}.json",
    }
    return names[arm]


def result_path(root, arm, recovery_seed, student_seed, partition_seed, legacy_seed):
    if student_seed == legacy_seed:
        return root / legacy_filename(arm, recovery_seed, partition_seed)
    return root / f"{arm}_rseed{recovery_seed}_sseed{student_seed}.json"


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def hierarchical_summary(values, recovery_seeds, student_seeds):
    flat = [values[(rseed, sseed)] for rseed in recovery_seeds for sseed in student_seeds]
    by_recovery = {
        str(rseed): {
            "mean_over_student_seeds": statistics.fmean(
                values[(rseed, sseed)] for sseed in student_seeds
            ),
            "student_seed_sample_sd": sample_sd(
                [values[(rseed, sseed)] for sseed in student_seeds]
            ),
        }
        for rseed in recovery_seeds
    }
    by_student = {
        str(sseed): {
            "mean_over_recovery_seeds": statistics.fmean(
                values[(rseed, sseed)] for rseed in recovery_seeds
            ),
            "recovery_seed_sample_sd": sample_sd(
                [values[(rseed, sseed)] for rseed in recovery_seeds]
            ),
        }
        for sseed in student_seeds
    }
    within_recovery_variances = [
        sample_sd([values[(rseed, sseed)] for sseed in student_seeds]) ** 2
        for rseed in recovery_seeds
    ]
    recovery_means = [by_recovery[str(rseed)]["mean_over_student_seeds"]
                      for rseed in recovery_seeds]
    return {
        "cells": len(flat),
        "grand_mean": statistics.fmean(flat),
        "sample_sd_across_cells_descriptive": sample_sd(flat),
        "pooled_within_recovery_student_seed_sd": statistics.fmean(
            within_recovery_variances
        ) ** 0.5,
        "recovery_seed_sd_of_student_seed_means": sample_sd(recovery_means),
        "by_recovery_seed": by_recovery,
        "by_student_seed": by_student,
    }


def main():
    parser = argparse.ArgumentParser("Summarize crossed recovery/post-evaluation seeds")
    parser.add_argument("--per-class-dir", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--legacy-student-seed", type=int, default=42)
    parser.add_argument("--random-partition-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.per_class_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    values = {arm: {} for arm in ARMS}
    rows = []
    for recovery_seed in args.recovery_seeds:
        for student_seed in args.student_seeds:
            for arm in ARMS:
                path = result_path(
                    root, arm, recovery_seed, student_seed,
                    args.random_partition_seed, args.legacy_student_seed,
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                top1 = float(payload["best_top1"])
                values[arm][(recovery_seed, student_seed)] = top1
                rows.append({
                    "arm": arm,
                    "recovery_seed": recovery_seed,
                    "student_seed": student_seed,
                    "best_top1": top1,
                    "source": str(path),
                })

    with (output / "post_eval_crossed_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    cell_vectors = {}
    comparison_rows = []
    for recovery_seed in args.recovery_seeds:
        for student_seed in args.student_seeds:
            vector = {
                arm: values[arm][(recovery_seed, student_seed)] for arm in ARMS
            }
            cell_vectors[(recovery_seed, student_seed)] = vector
            for comparison, function in COMPARISONS.items():
                comparison_rows.append({
                    "comparison": comparison,
                    "recovery_seed": recovery_seed,
                    "student_seed": student_seed,
                    "paired_gain": function(vector),
                })
    with (output / "post_eval_crossed_paired_gains.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
        writer.writeheader(); writer.writerows(comparison_rows)

    summary = {
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "legacy_student_seed": args.legacy_student_seed,
        "note": (
            "Cells sharing a recovery seed reuse the same synthetic/FKD dataset; all cells are "
            "reported, but they are not treated as mutually independent replicates."
        ),
        "arms": {
            arm: hierarchical_summary(values[arm], args.recovery_seeds, args.student_seeds)
            for arm in ARMS
        },
        "paired_comparisons": {},
    }
    for comparison, function in COMPARISONS.items():
        comparison_values = {
            key: function(vector) for key, vector in cell_vectors.items()
        }
        summary["paired_comparisons"][comparison] = hierarchical_summary(
            comparison_values, args.recovery_seeds, args.student_seeds
        )

    serialized = json.dumps(summary, indent=2)
    (output / "post_eval_crossed_summary.json").write_text(
        serialized + "\n", encoding="utf-8"
    )
    print(serialized)
    print(f"Crossed post-evaluation summary written to {output}")


if __name__ == "__main__":
    main()
