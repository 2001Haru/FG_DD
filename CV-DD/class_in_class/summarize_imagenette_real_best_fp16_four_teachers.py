import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t as student_t

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44, 45, 46)
RECOVERY_SEEDS = (41, 42)
STUDENT_SEEDS = (42, 43)


def load_best(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation set: {path}")
    return float(payload["best_top1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.sweep_root)
    values = {"c1_T800": {}, "random100_T200": {}}
    for teacher in TEACHER_SEEDS:
        per_class = root / f"tseed{teacher}" / "per_class"
        for recovery in RECOVERY_SEEDS:
            for student in STUDENT_SEEDS:
                key = (teacher, recovery, student)
                for arm in values:
                    path = per_class / f"real__{arm}_rseed{recovery}_sseed{student}.json"
                    values[arm][key] = load_best(path)

    delta = {
        key: values["random100_T200"][key] - values["c1_T800"][key]
        for key in values["c1_T800"]
    }
    teacher_delta_means = [
        statistics.fmean(
            delta[(teacher, recovery, student)]
            for recovery in RECOVERY_SEEDS for student in STUDENT_SEEDS
        )
        for teacher in TEACHER_SEEDS
    ]
    mean = statistics.fmean(teacher_delta_means)
    sd = statistics.stdev(teacher_delta_means)
    se = sd / math.sqrt(len(teacher_delta_means))
    statistic = mean / se
    df = len(teacher_delta_means) - 1
    two_sided = 2.0 * student_t.sf(abs(statistic), df)
    one_sided = student_t.sf(statistic, df)

    result = {
        "protocol": (
            "ImageNette IPC10 Real source, native FP16 FKD protocol; C1 T800 "
            "versus Random C100 T200; 4 Teacher x 2 source/recovery x 2 student seeds"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "arms": {
            arm: three_level_summary(
                arm_values, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
            )
            for arm, arm_values in values.items()
        },
        "paired_random_minus_c1": three_level_summary(
            delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        ),
        "teacher_seed_mean_paired_t_test": {
            "unit": "Teacher-seed mean after averaging recovery/source and student seeds",
            "teacher_seed_means": dict(zip(map(str, TEACHER_SEEDS), teacher_delta_means)),
            "n": len(teacher_delta_means),
            "mean": mean,
            "sample_sd": sd,
            "standard_error": se,
            "t": statistic,
            "df": df,
            "two_sided_p": two_sided,
            "one_sided_p_random_gt_c1": one_sided,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
