import argparse
import json
import statistics
from pathlib import Path


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def three_level_summary(values, teacher_seeds, recovery_seeds, student_seeds):
    flat = [
        values[(teacher, recovery, student)]
        for teacher in teacher_seeds
        for recovery in recovery_seeds
        for student in student_seeds
    ]
    by_teacher = {}
    student_variances = []
    recovery_variances = []
    teacher_means = []
    for teacher in teacher_seeds:
        teacher_values = [
            values[(teacher, recovery, student)]
            for recovery in recovery_seeds for student in student_seeds
        ]
        recovery_rows = {}
        recovery_means = []
        for recovery in recovery_seeds:
            current = [
                values[(teacher, recovery, student)] for student in student_seeds
            ]
            current_sd = sample_sd(current)
            current_mean = statistics.fmean(current)
            student_variances.append(current_sd ** 2)
            recovery_means.append(current_mean)
            recovery_rows[str(recovery)] = {
                "mean_over_student_seeds": current_mean,
                "student_seed_sample_sd": current_sd,
            }
        recovery_sd = sample_sd(recovery_means)
        recovery_variances.append(recovery_sd ** 2)
        teacher_mean = statistics.fmean(teacher_values)
        teacher_means.append(teacher_mean)
        by_teacher[str(teacher)] = {
            "mean_over_recovery_and_student_seeds": teacher_mean,
            "sample_sd_across_nine_cells": sample_sd(teacher_values),
            "recovery_seed_sd_of_student_seed_means": recovery_sd,
            "by_recovery_seed": recovery_rows,
        }
    return {
        "cells": len(flat),
        "grand_mean": statistics.fmean(flat),
        "sample_sd_across_cells_descriptive": sample_sd(flat),
        "pooled_within_teacher_recovery_student_seed_sd": (
            statistics.fmean(student_variances) ** 0.5
        ),
        "pooled_within_teacher_recovery_seed_sd_of_student_means": (
            statistics.fmean(recovery_variances) ** 0.5
        ),
        "teacher_seed_sd_of_recovery_student_means": sample_sd(teacher_means),
        "by_teacher_seed": by_teacher,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--c-values", nargs="+", type=int, default=(1, 2, 5, 10))
    parser.add_argument("--per-class-subdir", default="per_class")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.master_root)
    values = {c: {} for c in args.c_values}
    for teacher in args.teacher_seeds:
        per_class = root / f"tseed{teacher}" / args.per_class_subdir
        for c in args.c_values:
            for recovery in args.recovery_seeds:
                for student in args.student_seeds:
                    path = per_class / f"c{c}_rseed{recovery}_sseed{student}.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if int(payload.get("validation_images", -1)) != 3925:
                        raise ValueError(f"invalid validation set: {path}")
                    values[c][(teacher, recovery, student)] = float(
                        payload["best_top1"]
                    )

    summary = {
        "protocol": args.protocol or (
            "ImageNette IPC10 ResNet18 random CiC-T, official split, Teacher "
            "seeds43/44, recovery iter4000 LR0.1 r_bn0.01, marg10 T20"
        ),
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "c_values": args.c_values,
        "arms": {
            f"C{c}": three_level_summary(
                values[c], args.teacher_seeds, args.recovery_seeds,
                args.student_seeds,
            ) for c in args.c_values
        },
        "paired_vs_C1": {},
    }
    if 1 in values:
        for c in args.c_values:
            if c == 1:
                continue
            paired = {
                key: values[c][key] - values[1][key] for key in values[c]
            }
            summary["paired_vs_C1"][f"C{c}_minus_C1"] = three_level_summary(
                paired, args.teacher_seeds, args.recovery_seeds,
                args.student_seeds,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
