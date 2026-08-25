import argparse
import json
from pathlib import Path

from compare_imagenette_cluster_random import load_values
from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


def main():
    parser = argparse.ArgumentParser("Paired Cluster/Random x Soft/Hard interface decomposition")
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--student-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    common = (args.c_values, args.teacher_seeds, args.recovery_seeds, args.student_seeds)
    cluster_soft = load_values(Path(args.cluster_root), "per_class", *common)
    cluster_hard = load_values(Path(args.cluster_root), "hard_per_class", *common)
    random_soft = load_values(Path(args.random_root), "per_class", *common)
    random_hard = load_values(Path(args.random_root), "hard_per_class", *common)

    comparisons = {}
    for c in args.c_values:
        keys = cluster_soft[c]
        effects = {
            "cluster_minus_random_hard": {
                key: cluster_hard[c][key] - random_hard[c][key] for key in keys
            },
            "cluster_minus_random_soft": {
                key: cluster_soft[c][key] - random_soft[c][key] for key in keys
            },
            "cluster_soft_minus_hard": {
                key: cluster_soft[c][key] - cluster_hard[c][key] for key in keys
            },
            "random_soft_minus_hard": {
                key: random_soft[c][key] - random_hard[c][key] for key in keys
            },
        }
        effects["interface_difference_in_differences"] = {
            key: (
                effects["cluster_soft_minus_hard"][key]
                - effects["random_soft_minus_hard"][key]
            ) for key in keys
        }
        comparisons[f"C{c}"] = {
            name: three_level_summary(
                values, args.teacher_seeds, args.recovery_seeds, args.student_seeds
            ) for name, values in effects.items()
        }
    result = {
        "protocol": "Strictly paired Cluster/Random x Soft/Hard 2x2 decomposition",
        "definition": (
            "interface DiD = (Cluster Soft - Cluster Hard) - "
            "(Random Soft - Random Hard)"
        ),
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "c_values": args.c_values,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
