import argparse
import csv
import json
from pathlib import Path

from summarize_post_eval_seeds import hierarchical_summary, result_path


NEW_ARMS = (
    "oracle_aligned", "baseline_mismatched", "random_aligned", "oracle_100dim",
    "baseline_random_marg20", "baseline_random_100dim",
)
REFERENCE_ARMS = {
    "baseline": "baseline",
    "oracle_existing": "oracle_fine_target",
    "random_existing": "random_pseudo_target",
}
COMPARISONS = {
    "oracle_aligned_minus_oracle_existing": ("oracle_aligned", "oracle_existing"),
    "baseline_mismatched_minus_baseline": ("baseline_mismatched", "baseline"),
    "random_aligned_minus_random_existing": ("random_aligned", "random_existing"),
    "oracle_100dim_minus_oracle_aligned": ("oracle_100dim", "oracle_aligned"),
    "oracle_aligned_minus_baseline_mismatched": (
        "oracle_aligned", "baseline_mismatched"
    ),
    "baseline_random_marg20_minus_baseline": (
        "baseline_random_marg20", "baseline"
    ),
    "baseline_random_100dim_minus_baseline_random_marg20": (
        "baseline_random_100dim", "baseline_random_marg20"
    ),
    "baseline_random_marg20_minus_baseline_fine_marg20": (
        "baseline_random_marg20", "baseline_mismatched"
    ),
}


def main():
    parser = argparse.ArgumentParser("Summarize relabel alignment matrix")
    parser.add_argument("--alignment-per-class", required=True)
    parser.add_argument("--reference-per-class", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--random-partition-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    alignment_root, reference_root = Path(args.alignment_per_class), Path(args.reference_per_class)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)

    values = {arm: {} for arm in (*NEW_ARMS, *REFERENCE_ARMS.keys())}
    rows = []
    for recovery_seed in args.recovery_seeds:
        for student_seed in args.student_seeds:
            for arm in NEW_ARMS:
                path = alignment_root / f"{arm}_rseed{recovery_seed}_sseed{student_seed}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                values[arm][(recovery_seed, student_seed)] = float(payload["best_top1"])
                rows.append({
                    "arm": arm, "recovery_seed": recovery_seed,
                    "student_seed": student_seed, "best_coarse20_top1": payload["best_top1"],
                    "native_top1_at_best_checkpoint": payload.get(
                        "native_top1_at_best_checkpoint"
                    ), "source": str(path),
                })
            for label, reference_arm in REFERENCE_ARMS.items():
                path = result_path(
                    reference_root, reference_arm, recovery_seed, student_seed,
                    args.random_partition_seed, 42,
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                values[label][(recovery_seed, student_seed)] = float(payload["best_top1"])

    with (output / "relabel_alignment_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "temperature": args.temperature,
        "temperature_note": "100-way temperature reused from 20-way protocol; not retuned",
        "arms": {
            arm: hierarchical_summary(values[arm], args.recovery_seeds, args.student_seeds)
            for arm in values
        },
        "paired_comparisons": {},
    }
    for comparison, (positive, negative) in COMPARISONS.items():
        differences = {
            key: values[positive][key] - values[negative][key]
            for key in values[positive]
        }
        summary["paired_comparisons"][comparison] = hierarchical_summary(
            differences, args.recovery_seeds, args.student_seeds
        )
    serialized = json.dumps(summary, indent=2)
    (output / "relabel_alignment_summary.json").write_text(
        serialized + "\n", encoding="utf-8"
    )
    print(serialized)


if __name__ == "__main__":
    main()
