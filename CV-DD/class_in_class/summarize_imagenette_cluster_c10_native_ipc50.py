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


def summary(values):
    ordered = [values[seed] for seed in STUDENT_SEEDS]
    return {
        "cells": len(ordered),
        "mean": statistics.mean(ordered),
        "student_seed_sample_sd": statistics.stdev(ordered),
        "by_student_seed": {str(seed): values[seed] for seed in STUDENT_SEEDS},
    }


def paired(left, right):
    return summary({seed: left[seed] - right[seed] for seed in STUDENT_SEEDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--old-ipc-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    old = Path(args.old_ipc_root) / "tseed43" / "per_class"

    new_hard = {
        seed: load(root / "per_class" / f"hard_sseed{seed}.json", "hard_coarse_label")
        for seed in STUDENT_SEEDS
    }
    new_soft = {
        seed: load(root / "per_class" / f"c1_soft_sseed{seed}.json", "fkd_soft_label")
        for seed in STUDENT_SEEDS
    }
    reference = {}
    for source in ("real", "c1"):
        hard = {
            seed: load(
                old / f"ipc50_{source}__hard_rseed41_sseed{seed}.json",
                "hard_coarse_label",
            )
            for seed in STUDENT_SEEDS
        }
        soft = {
            seed: load(
                old / f"ipc50_{source}__c1_rseed41_sseed{seed}.json",
                "fkd_soft_label",
            )
            for seed in STUDENT_SEEDS
        }
        reference[source] = {
            "hard": summary(hard),
            "c1_soft": summary(soft),
            "c1_soft_minus_hard": paired(soft, hard),
        }
        reference[source]["native_minus_reference"] = {
            "hard": paired(new_hard, hard),
            "c1_soft": paired(new_soft, soft),
        }

    result = {
        "audit_schema_version": 1,
        "protocol": (
            "ImageNette IPC50 ResNet18; DINO Cluster C10 native-100 recovery, "
            "recovery BS100, five images per subclass and 500 total; Teacher seed43, "
            "recovery seed41, student seeds42/43/44; collapsed coarse10 Hard versus "
            "C1 CutMix FKD T20; AdamW LR5e-4 eta2; full 3925-image test"
        ),
        "teacher_seed": 43,
        "recovery_seed": 41,
        "student_seeds": list(STUDENT_SEEDS),
        "new_native_recovery": {
            "hard": summary(new_hard),
            "c1_soft": summary(new_soft),
            "c1_soft_minus_hard": paired(new_soft, new_hard),
        },
        "strict_same_seed_ipc50_references": reference,
        "historical_18_cell_grand_means": {
            "random_real": {"hard": 69.96, "c1_soft": 85.06, "random_c100_soft": 77.69},
            "c1_recovery": {"hard": 55.10, "c1_soft": 82.00, "random_c100_soft": 75.81},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
