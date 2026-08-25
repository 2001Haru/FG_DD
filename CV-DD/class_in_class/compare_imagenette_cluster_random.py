import argparse
import json
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


def load_values(root, subdir, c_values, teacher_seeds, recovery_seeds, student_seeds):
    values = {c: {} for c in c_values}
    for teacher in teacher_seeds:
        per_class = root / f"tseed{teacher}" / subdir
        for c in c_values:
            for recovery in recovery_seeds:
                for student in student_seeds:
                    path = per_class / f"c{c}_rseed{recovery}_sseed{student}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if int(record.get("validation_images", -1)) != 3925:
                        raise ValueError(f"invalid validation metadata: {path}")
                    values[c][(teacher, recovery, student)] = float(record["best_top1"])
    return values


def main():
    parser = argparse.ArgumentParser("Paired DINO-cluster versus random CiC-T comparison")
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--cluster-subdir", default="per_class")
    parser.add_argument("--random-subdir", default="per_class")
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--student-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cluster = load_values(
        Path(args.cluster_root), args.cluster_subdir, args.c_values, args.teacher_seeds,
        args.recovery_seeds, args.student_seeds,
    )
    random = load_values(
        Path(args.random_root), args.random_subdir, args.c_values, args.teacher_seeds,
        args.recovery_seeds, args.student_seeds,
    )
    arms, paired = {}, {}
    for c in args.c_values:
        arms[f"C{c}"] = {
            "cluster": three_level_summary(
                cluster[c], args.teacher_seeds, args.recovery_seeds, args.student_seeds
            ),
            "random": three_level_summary(
                random[c], args.teacher_seeds, args.recovery_seeds, args.student_seeds
            ),
        }
        differences = {key: cluster[c][key] - random[c][key] for key in cluster[c]}
        paired[f"C{c}_cluster_minus_random"] = three_level_summary(
            differences, args.teacher_seeds, args.recovery_seeds, args.student_seeds
        )
    result = {
        "protocol": "Strictly paired DINO-cluster minus balanced-random CiC-T by C/Teacher/recovery/student seed",
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "c_values": args.c_values,
        "arms": arms,
        "paired_cluster_minus_random": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
