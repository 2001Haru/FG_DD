import argparse
import json
import math
import statistics
from pathlib import Path


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric(values):
    return {"mean": statistics.fmean(values), "teacher_seed_sample_sd": sample_sd(values)}


def main():
    parser = argparse.ArgumentParser("Summarize DINO-cluster Teacher audits across seeds")
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--cluster-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.master_root)
    rows = []

    for c in args.c_values:
        per_teacher = []
        for teacher_seed in args.teacher_seeds:
            path = (
                root / f"tseed{teacher_seed}" / "audits"
                / f"dinov2_cluster_c{c}_teacher_audit.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            if int(record["subclasses_per_coarse"]) != c:
                raise ValueError(f"C mismatch: {path}")
            train, test = record["train"], record["val"]
            if int(train["images"]) != 9469 or int(test["images"]) != 3925:
                raise ValueError(f"split mismatch: {path}")
            conditional = float(test["conditional_native_given_coarse_correct"])
            per_teacher.append({
                "teacher_seed": teacher_seed,
                "train_native_top1": float(train["native_subclass_top1"]),
                "train_coarse_top1": float(train["collapsed_coarse10_top1"]),
                "train_within_parent_entropy": float(train["within_parent_entropy"]),
                "train_normalized_within_parent_entropy": (
                    float(train["within_parent_entropy"]) / math.log(c)
                ),
                "test_native_top1": float(test["native_subclass_top1"]),
                "test_coarse_top1": float(test["collapsed_coarse10_top1"]),
                "test_native_to_coarse_hit_ratio": float(
                    test["native_to_collapsed_hit_ratio"]
                ),
                "test_conditional_native_given_coarse": conditional,
                "test_conditional_excess_over_chance_percentage_points": (
                    100.0 * (conditional - 1.0 / c)
                ),
                "test_within_parent_entropy": float(test["within_parent_entropy"]),
                "test_normalized_within_parent_entropy": (
                    float(test["within_parent_entropy"]) / math.log(c)
                ),
                "conditional_binomial_test": test["conditional_ratio_binomial_test"],
                "audit_path": str(path),
            })

        fields = [
            "train_native_top1", "train_coarse_top1",
            "train_within_parent_entropy", "train_normalized_within_parent_entropy",
            "test_native_top1", "test_coarse_top1",
            "test_native_to_coarse_hit_ratio",
            "test_conditional_native_given_coarse",
            "test_conditional_excess_over_chance_percentage_points",
            "test_within_parent_entropy", "test_normalized_within_parent_entropy",
        ]
        rows.append({
            "C": c,
            "heads": 10 * c,
            "expected_conditional_native_chance": 1.0 / c,
            "metrics": {
                field: metric([teacher[field] for teacher in per_teacher])
                for field in fields
            },
            "by_teacher_seed": per_teacher,
        })

    result = {
        "protocol": (
            "ImageNette DINOv2 balanced within-parent Cluster Teachers; "
            "clean deterministic train/test audit; test subclasses assigned to "
            "nearest train centroid"
        ),
        "teacher_seeds": args.teacher_seeds,
        "cluster_seed": args.cluster_seed,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
