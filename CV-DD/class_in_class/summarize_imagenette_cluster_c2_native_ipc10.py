import argparse
import json
import statistics
from pathlib import Path


STUDENT_SEEDS = (42, 43, 44)


def load(path, target):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise RuntimeError(f"invalid validation split: {path}")
    if payload.get("training_target") != target:
        raise RuntimeError(f"training target mismatch: {path}")
    return float(payload["best_top1"])


def summarize(values):
    ordered = [values[seed] for seed in STUDENT_SEEDS]
    return {
        "cells": len(ordered),
        "mean": statistics.mean(ordered),
        "student_seed_sample_sd": statistics.stdev(ordered),
        "by_student_seed": {str(seed): values[seed] for seed in STUDENT_SEEDS},
    }


def paired(left, right):
    return summarize({seed: left[seed] - right[seed] for seed in STUDENT_SEEDS})


def load_arm(root, hard_pattern, soft_pattern):
    hard = {
        seed: load(root / hard_pattern.format(seed=seed), "hard_coarse_label")
        for seed in STUDENT_SEEDS
    }
    soft = {
        seed: load(root / soft_pattern.format(seed=seed), "fkd_soft_label")
        for seed in STUDENT_SEEDS
    }
    return hard, soft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--c10-native-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)

    new_hard, new_soft = load_arm(
        root / "per_class", "hard_sseed{seed}.json", "c1_soft_sseed{seed}.json"
    )
    c1_hard, c1_soft = load_arm(
        Path(args.c1_root) / "tseed43",
        "hard_per_class/c1_rseed41_sseed{seed}.json",
        "per_class/c1_rseed41_sseed{seed}.json",
    )
    c10_hard, c10_soft = load_arm(
        Path(args.c10_native_root) / "tseed43" / "per_class",
        "hard_rseed41_sseed{seed}.json",
        "c1_soft_rseed41_sseed{seed}.json",
    )

    def arm(hard, soft):
        return {
            "hard": summarize(hard),
            "c1_soft": summarize(soft),
            "c1_soft_minus_hard": paired(soft, hard),
        }

    result = {
        "audit_schema_version": 1,
        "protocol": (
            "ImageNette IPC10 ResNet18; DINO Cluster C2 native-20 recovery, "
            "recovery BS20, five images per subclass and 100 total; Teacher seed43, "
            "recovery seed41, student seeds42/43/44; collapsed coarse10 Hard versus "
            "C1 CutMix FKD T20; AdamW LR5e-4 eta1; full 3925-image test"
        ),
        "teacher_seed": 43,
        "recovery_seed": 41,
        "student_seeds": list(STUDENT_SEEDS),
        "new_c2_native_recovery": arm(new_hard, new_soft),
        "strict_same_seed_references": {
            "c1_coarse_recovery": {
                **arm(c1_hard, c1_soft),
                "c2_native_minus_reference": {
                    "hard": paired(new_hard, c1_hard),
                    "c1_soft": paired(new_soft, c1_soft),
                },
            },
            "c10_native_recovery": {
                **arm(c10_hard, c10_soft),
                "c2_native_minus_reference": {
                    "hard": paired(new_hard, c10_hard),
                    "c1_soft": paired(new_soft, c10_soft),
                },
            },
        },
        "historical_18_cell_grand_means": {
            "random_c2_coarse_recovery": {"hard": 43.71, "matched_soft": 64.81},
            "cluster_c2_coarse_recovery": {"hard": 42.73, "matched_soft": 61.91},
            "c1_coarse_recovery": {"hard": 43.32, "c1_soft": 62.05},
            "cluster_c10_native_recovery": {"hard": 42.13, "c1_soft": 64.03},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
