import argparse
import json
from pathlib import Path

from summarize_imagenette_cic_t_teacher_seeds import three_level_summary


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42)
STUDENT_SEEDS = (42, 43)
TEMPERATURES = (200, 400, 800, 1600)
ROWS = ("real", "c1")
COLS = ("c1", "random100")


def load_best(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("validation_images", -1)) != 3925:
        raise ValueError(f"invalid validation split: {path}")
    return float(payload["best_top1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-root", required=True)
    parser.add_argument("--fp32-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fp16_root, fp32_root = Path(args.fp16_root), Path(args.fp32_root)

    fp16, fp32 = {}, {}
    for row in ROWS:
        for col in COLS:
            for temperature in TEMPERATURES:
                key = (row, col, temperature)
                fp16[key], fp32[key] = {}, {}
                for teacher in TEACHER_SEEDS:
                    for recovery in RECOVERY_SEEDS:
                        for student in STUDENT_SEEDS:
                            seed_key = (teacher, recovery, student)
                            old = (fp16_root / f"tseed{teacher}" / "per_class" /
                                   f"{row}__{col}_T{temperature}_rseed{recovery}_sseed{student}.json")
                            new = (fp32_root / f"tseed{teacher}" / "per_class" /
                                   f"{row}__{col}_T{temperature}_fp32_rseed{recovery}_sseed{student}.json")
                            fp16[key][seed_key] = load_best(old)
                            fp32[key][seed_key] = load_best(new)

    arms = {}
    paired = {}
    for key, values in fp32.items():
        row, col, temperature = key
        name = f"{row}_{col}_T{temperature}"
        arms[name] = {
            "fp32": three_level_summary(
                values, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
            ),
            "fp16": three_level_summary(
                fp16[key], TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
            ),
        }
        delta = {seed: values[seed] - fp16[key][seed] for seed in values}
        paired[f"{name}_fp32_minus_fp16"] = three_level_summary(
            delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )

    source_average = {}
    for col in COLS:
        for temperature in TEMPERATURES:
            source_average[(col, temperature)] = {
                seed: (
                    fp32[("real", col, temperature)][seed]
                    + fp32[("c1", col, temperature)][seed]
                ) / 2.0
                for seed in fp32[("real", col, temperature)]
            }
    source_average_summaries = {
        f"{col}_T{temperature}": three_level_summary(
            values, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )
        for (col, temperature), values in source_average.items()
    }
    best_temperature = {
        col: max(
            TEMPERATURES,
            key=lambda temperature: source_average_summaries[
                f"{col}_T{temperature}"
            ]["grand_mean"],
        )
        for col in COLS
    }

    c1_temp = best_temperature["c1"]
    random_temp = best_temperature["random100"]
    optimized = {}
    for row in ROWS:
        delta = {
            seed: (
                fp32[(row, "random100", random_temp)][seed]
                - fp32[(row, "c1", c1_temp)][seed]
            )
            for seed in fp32[(row, "c1", c1_temp)]
        }
        optimized[f"{row}_random_minus_c1"] = three_level_summary(
            delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
        )
    averaged_delta = {
        seed: (
            source_average[("random100", random_temp)][seed]
            - source_average[("c1", c1_temp)][seed]
        )
        for seed in source_average[("c1", c1_temp)]
    }
    optimized["source_average_random_minus_c1"] = three_level_summary(
        averaged_delta, TEACHER_SEEDS, RECOVERY_SEEDS, STUDENT_SEEDS
    )

    result = {
        "protocol": (
            "ImageNette IPC10 high-temperature FP32 FKD storage control; only "
            "saved marginalized logits dtype changes from FP16 to FP32"
        ),
        "temperatures": list(TEMPERATURES),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "student_seeds": list(STUDENT_SEEDS),
        "arms": arms,
        "paired_fp32_minus_fp16": paired,
        "fp32_source_average_summaries": source_average_summaries,
        "fp32_best_temperature_by_equal_source_average_within_high_temperature_grid": best_temperature,
        "fp32_optimized_comparisons_within_high_temperature_grid": optimized,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
