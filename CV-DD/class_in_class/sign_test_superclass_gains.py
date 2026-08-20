import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


GAIN_FIELDS = {
    "oracle_minus_baseline": "paired_gain_mean",
    "random_minus_baseline": "random_paired_gain_mean",
    "oracle_minus_random": "oracle_vs_random_mean",
    "coarse_target_minus_baseline": "coarse_target_paired_gain_mean",
    "oracle_minus_coarse_target": "oracle_vs_coarse_target_mean",
    "random_minus_random_coarse_target": "random_vs_random_coarse_target_mean",
    "difference_in_differences": "difference_in_differences_mean",
}


def upper_tail(k, n):
    return sum(math.comb(n, value) for value in range(k, n + 1)) / (2 ** n)


def lower_tail(k, n):
    return sum(math.comb(n, value) for value in range(0, k + 1)) / (2 ** n)


def summarize(values, epsilon):
    positive = sorted(name for name, value in values if value > epsilon)
    negative = sorted(name for name, value in values if value < -epsilon)
    ties = sorted(name for name, value in values if abs(value) <= epsilon)
    k, n = len(positive), len(positive) + len(negative)
    if n == 0:
        one_sided = two_sided = 1.0
    else:
        one_sided = upper_tail(k, n)
        two_sided = min(1.0, 2.0 * min(upper_tail(k, n), lower_tail(k, n)))
    return {
        "total_superclasses": len(values),
        "effective_non_ties": n,
        "positive": k,
        "negative": len(negative),
        "ties_excluded": len(ties),
        "positive_fraction_non_ties": k / n if n else None,
        "one_sided_p_gain_gt_half": one_sided,
        "two_sided_p": two_sided,
        "positive_classes": positive,
        "negative_classes": negative,
        "tie_classes": ties,
    }


def load_accuracy_gains(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for comparison, field in GAIN_FIELDS.items():
        if field not in rows[0] or rows[0][field] == "":
            continue
        result[comparison] = [
            (row["coarse_name"], float(row[field])) for row in rows
        ]
    return result


def load_f1_gains(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["comparison"]].append((row["class_name"], float(row["f1_gain_mean"])))
    return dict(grouped)


def main():
    parser = argparse.ArgumentParser("Exact superclass gain sign tests")
    parser.add_argument("--superclass-results", required=True)
    parser.add_argument("--f1-differences")
    parser.add_argument("--epsilon", type=float, default=1e-12,
                        help="absolute gain at or below this value is treated as a tie")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = {
        "test": (
            "Exact binomial sign test across superclass-level mean paired gains. "
            "Ties are excluded. The one-sided alternative is positive probability > 0.5."
        ),
        "epsilon": args.epsilon,
        "accuracy_or_recall_gain": {
            comparison: summarize(values, args.epsilon)
            for comparison, values in load_accuracy_gains(
                Path(args.superclass_results)
            ).items()
        },
    }
    if args.f1_differences:
        payload["f1_gain"] = {
            comparison: summarize(values, args.epsilon)
            for comparison, values in load_f1_gains(Path(args.f1_differences)).items()
        }
    serialized = json.dumps(payload, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
