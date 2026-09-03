import argparse
import json
import math
import statistics
from pathlib import Path


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42, 43)
STUDENT_SEEDS = (42, 43, 44)
LABEL_ARMS = ("hard", "c1_soft")


def load_result(path, expected_target):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise RuntimeError(f"result does not use full ImageNette test split: {path}")
    if payload.get("training_target") != expected_target:
        raise RuntimeError(f"training target mismatch: {path}")
    return float(payload["best_top1"])


def summarize(values):
    all_values = list(values.values())
    within_cell_student_sds = []
    recovery_means_by_teacher = {}
    for teacher in TEACHER_SEEDS:
        recovery_means = []
        for recovery in RECOVERY_SEEDS:
            selected = [
                value for (t, r, _), value in values.items()
                if t == teacher and r == recovery
            ]
            within_cell_student_sds.append(statistics.stdev(selected))
            recovery_means.append(statistics.mean(selected))
        recovery_means_by_teacher[teacher] = recovery_means
    teacher_means = [
        statistics.mean(value for (t, _, _), value in values.items() if t == teacher)
        for teacher in TEACHER_SEEDS
    ]
    recovery_sds = [
        statistics.stdev(recovery_means_by_teacher[teacher])
        for teacher in TEACHER_SEEDS
    ]
    return {
        "cells": len(all_values),
        "grand_mean": statistics.mean(all_values),
        "sample_sd_across_cells_descriptive": statistics.stdev(all_values),
        "pooled_within_teacher_recovery_student_seed_sd": math.sqrt(
            statistics.mean(value * value for value in within_cell_student_sds)
        ),
        "pooled_within_teacher_recovery_seed_sd_of_student_means": math.sqrt(
            statistics.mean(value * value for value in recovery_sds)
        ),
        "teacher_seed_sd_of_recovery_student_means": statistics.stdev(teacher_means),
        "by_teacher_seed": {
            str(teacher): {
                "mean": teacher_means[index],
                "recovery_seed_means": {
                    str(recovery): recovery_means_by_teacher[teacher][recovery_index]
                    for recovery_index, recovery in enumerate(RECOVERY_SEEDS)
                },
            }
            for index, teacher in enumerate(TEACHER_SEEDS)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)

    values = {arm: {} for arm in LABEL_ARMS}
    for teacher in TEACHER_SEEDS:
        for recovery in RECOVERY_SEEDS:
            collapsed = (
                root / f"tseed{teacher}" / "coarse_sources"
                / f"cluster_c10_native100_rseed{recovery}"
            )
            manifest = json.loads(
                (collapsed / ".collapse_manifest.json").read_text(encoding="utf-8")
            )
            provenance = manifest["provenance"]
            if (
                int(provenance["native_classes"]) != 100
                or int(provenance["coarse_classes"]) != 10
                or int(provenance["total_images"]) != 100
            ):
                raise RuntimeError(f"invalid collapsed source manifest: {collapsed}")
            for student in STUDENT_SEEDS:
                key = (teacher, recovery, student)
                hard_path = (
                    root / f"tseed{teacher}" / "per_class"
                    / f"hard_rseed{recovery}_sseed{student}.json"
                )
                soft_path = (
                    root / f"tseed{teacher}" / "per_class"
                    / f"c1_soft_rseed{recovery}_sseed{student}.json"
                )
                values["hard"][key] = load_result(hard_path, "hard_coarse_label")
                values["c1_soft"][key] = load_result(soft_path, "fkd_soft_label")

    arms = {arm: summarize(arm_values) for arm, arm_values in values.items()}
    paired_values = {
        key: values["c1_soft"][key] - values["hard"][key]
        for key in values["hard"]
    }
    result = {
        "audit_schema_version": 1,
        "protocol": (
            "ImageNette IPC10 ResNet18; DINOv2 Cluster C10 native-100 recovery "
            "with one image per subclass (100 total), no recovery marginalization; "
            "collapsed coarse10 downstream source; Hard versus paired C1-Teacher "
            "CutMix FKD soft labels at T20; full 3925-image test split"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "arms": arms,
        "paired_c1_soft_minus_hard": summarize(paired_values),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
