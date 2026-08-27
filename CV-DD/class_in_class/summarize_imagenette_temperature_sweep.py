import argparse
import json
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42)
STUDENT_SEEDS = (42, 43)
TEMPERATURES = (1, 2, 4, 8, 20, 46, 100, 200, 400, 800, 1600)
ROWS = ("real", "c1")
COLS = ("c1", "random100")


def tag(col, temperature):
    return f"{col}_T{str(temperature).replace('.', 'p')}"


def result_path(roots, teacher, recovery, student, row, col, temperature):
    random_root, factorial_root, match_root, sweep_root = roots
    if temperature == 20:
        if row == "real":
            return (factorial_root / f"tseed{teacher}" / "per_class" /
                    f"real__{col}_rseed{recovery}_sseed{student}.json")
        if col == "c1":
            return (random_root / f"tseed{teacher}" / "per_class" /
                    f"c1_rseed{recovery}_sseed{student}.json")
        return (factorial_root / f"tseed{teacher}" / "per_class" /
                f"c1__random100_rseed{recovery}_sseed{student}.json")
    if col == "c1" and temperature == 46:
        return (match_root / f"tseed{teacher}" / "per_class" /
                f"ipc10_{row}__c1_T46p0_rseed{recovery}_sseed{student}.json")
    return (sweep_root / f"tseed{teacher}" / "per_class" /
            f"{row}__{tag(col, temperature)}_rseed{recovery}_sseed{student}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--factorial-root", required=True)
    parser.add_argument("--match-root", required=True)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = tuple(map(Path, (
        args.random_root, args.factorial_root, args.match_root, args.sweep_root
    )))

    values = {}
    for row in ROWS:
        for col in COLS:
            for temperature in TEMPERATURES:
                current = {}
                for teacher in TEACHER_SEEDS:
                    for recovery in RECOVERY_SEEDS:
                        for student in STUDENT_SEEDS:
                            path = result_path(
                                roots, teacher, recovery, student, row, col,
                                temperature,
                            )
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            if int(payload.get("validation_images", -1)) != 3925:
                                raise ValueError(f"invalid validation split: {path}")
                            current[(teacher, recovery, student)] = float(
                                payload["best_top1"]
                            )
                values[(row, col, temperature)] = current

    arms = {
        f"{row}_{col}_T{temperature}": three_level_summary(
            values[(row, col, temperature)], TEACHER_SEEDS,
            RECOVERY_SEEDS, STUDENT_SEEDS,
        )
        for row in ROWS for col in COLS for temperature in TEMPERATURES
    }

    average_over_sources = {}
    for col in COLS:
        for temperature in TEMPERATURES:
            paired = {
                key: (
                    values[("real", col, temperature)][key]
                    + values[("c1", col, temperature)][key]
                ) / 2.0
                for key in values[("real", col, temperature)]
            }
            average_over_sources[(col, temperature)] = paired

    best_by_source = {}
    for row in ROWS:
        for col in COLS:
            best_temperature = max(
                TEMPERATURES,
                key=lambda temp: arms[f"{row}_{col}_T{temp}"]["grand_mean"],
            )
            best_by_source[f"{row}_{col}"] = {
                "best_temperature": best_temperature,
                "summary": arms[f"{row}_{col}_T{best_temperature}"],
            }

    source_average_summaries = {
        f"{col}_T{temperature}": three_level_summary(
            average_over_sources[(col, temperature)], TEACHER_SEEDS,
            RECOVERY_SEEDS, STUDENT_SEEDS,
        )
        for col in COLS for temperature in TEMPERATURES
    }
    best_average_temperature = {
        col: max(
            TEMPERATURES,
            key=lambda temp: source_average_summaries[
                f"{col}_T{temp}"
            ]["grand_mean"],
        )
        for col in COLS
    }

    c1_best = best_average_temperature["c1"]
    random_best = best_average_temperature["random100"]
    optimized_comparisons = {}
    for row in ROWS:
        delta = {
            key: (
                values[(row, "random100", random_best)][key]
                - values[(row, "c1", c1_best)][key]
            )
            for key in values[(row, "c1", c1_best)]
        }
        optimized_comparisons[f"{row}_random_minus_c1"] = three_level_summary(
            delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )
    averaged_delta = {
        key: (
            average_over_sources[("random100", random_best)][key]
            - average_over_sources[("c1", c1_best)][key]
        )
        for key in average_over_sources[("c1", c1_best)]
    }
    optimized_comparisons["source_average_random_minus_c1"] = (
        three_level_summary(
            averaged_delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )
    )

    result = {
        "protocol": (
            "ImageNette IPC10 ResNet18 temperature shape sweep; real and C1 "
            "synthetic sources; C1 and RandomC100 labelers; 2 Teacher x 2 "
            "recovery x 2 student seeds; full 3925-image test"
        ),
        "temperatures": list(TEMPERATURES),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "kd_t_squared_compensation": False,
        "note": (
            "Temperature changes both target shape and KL gradient scale because "
            "the reference implementation does not multiply the KD loss by T^2."
        ),
        "arms": arms,
        "best_by_source": best_by_source,
        "source_average_summaries": source_average_summaries,
        "best_temperature_by_equal_source_average": best_average_temperature,
        "optimized_temperature_comparisons": optimized_comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
