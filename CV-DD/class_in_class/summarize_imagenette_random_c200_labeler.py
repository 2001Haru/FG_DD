import argparse
import json
import statistics
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42, 43)
STUDENT_SEEDS = (42, 43, 44)
ROWS = ("real", "c1")


def load(path, expected_target=None, strict_target=True):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise RuntimeError(f"invalid validation split: {path}")
    if strict_target and expected_target is not None and payload.get("training_target") != expected_target:
        raise RuntimeError(f"training target mismatch: {path}")
    return float(payload["best_top1"])


def paired(left, right):
    if set(left) != set(right):
        raise RuntimeError("paired cell keys differ")
    return {key: left[key] - right[key] for key in left}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_root)
    random_root = Path(args.random_root)
    factorial = Path(args.factorial_root)

    values = {row: {} for row in ROWS}
    hard = {row: {} for row in ROWS}
    c1 = {row: {} for row in ROWS}
    c100 = {row: {} for row in ROWS}
    for row in ROWS:
        for teacher in TEACHER_SEEDS:
            for recovery in RECOVERY_SEEDS:
                for student in STUDENT_SEEDS:
                    key = (teacher, recovery, student)
                    values[row][key] = load(
                        experiment / f"tseed{teacher}" / "per_class"
                        / f"{row}__random200_rseed{recovery}_sseed{student}.json",
                        "fkd_soft_label",
                    )
                    if row == "real":
                        hard_path = (
                            factorial / f"tseed{teacher}" / "per_class"
                            / f"real__hard_rseed{recovery}_sseed{student}.json"
                        )
                        c1_path = (
                            factorial / f"tseed{teacher}" / "per_class"
                            / f"real__c1_rseed{recovery}_sseed{student}.json"
                        )
                        c100_path = (
                            factorial / f"tseed{teacher}" / "per_class"
                            / f"real__random100_rseed{recovery}_sseed{student}.json"
                        )
                    else:
                        hard_path = (
                            random_root / f"tseed{teacher}" / "hard_per_class"
                            / f"c1_rseed{recovery}_sseed{student}.json"
                        )
                        c1_path = (
                            random_root / f"tseed{teacher}" / "per_class"
                            / f"c1_rseed{recovery}_sseed{student}.json"
                        )
                        c100_path = (
                            factorial / f"tseed{teacher}" / "per_class"
                            / f"c1__random100_rseed{recovery}_sseed{student}.json"
                        )
                    # Historical files predate strict training_target metadata in
                    # some cells. Full-test metadata and path provenance are used.
                    hard[row][key] = load(hard_path, strict_target=False)
                    c1[row][key] = load(c1_path, strict_target=False)
                    c100[row][key] = load(c100_path, strict_target=False)

    def summarize(cells):
        return three_level_summary(
            cells, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )

    rows = {}
    for row in ROWS:
        rows[row] = {
            "hard": summarize(hard[row]),
            "c1_soft": summarize(c1[row]),
            "random_c100_soft": summarize(c100[row]),
            "random_c200_soft": summarize(values[row]),
            "random_c200_minus_hard": summarize(paired(values[row], hard[row])),
            "random_c200_minus_c1": summarize(paired(values[row], c1[row])),
            "random_c200_minus_random_c100": summarize(paired(values[row], c100[row])),
        }

    def average_rows(source):
        return {
            key: statistics.fmean(source[row][key] for row in ROWS)
            for key in source["real"]
        }

    column = average_rows(values)
    hard_column = average_rows(hard)
    c1_column = average_rows(c1)
    c100_column = average_rows(c100)
    result = {
        "audit_schema_version": 1,
        "protocol": (
            "ImageNette IPC10 ResNet18; existing Random Real and C1 synthetic "
            "sources relabeled by Random C200 (2000-head) Teachers, marginalized "
            "to coarse10 at T20; Teacher seeds43/44, recovery/source seeds41/42/43, "
            "student seeds42/43/44; full 3925-image test"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "rows": rows,
        "two_source_equal_weight_column": {
            "hard": summarize(hard_column),
            "c1_soft": summarize(c1_column),
            "random_c100_soft": summarize(c100_column),
            "random_c200_soft": summarize(column),
            "random_c200_minus_hard": summarize(paired(column, hard_column)),
            "random_c200_minus_c1": summarize(paired(column, c1_column)),
            "random_c200_minus_random_c100": summarize(paired(column, c100_column)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
