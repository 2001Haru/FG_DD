import argparse
import json
from pathlib import Path

from summarize_post_eval_seeds import hierarchical_summary


ARMS = (
    "official_teacher_c1_patches",
    "c1_teacher_official_patches",
)


def read_cells(root, pattern, recovery_seeds, student_seeds):
    values = {}
    for recovery_seed in recovery_seeds:
        for student_seed in student_seeds:
            path = root / pattern.format(r=recovery_seed, s=student_seed)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("validation_images", -1)) != 3925:
                raise ValueError(f"{path} was not evaluated on 3925 images")
            values[(recovery_seed, student_seed)] = float(payload["best_top1"])
    return values


def paired(left, right):
    return {key: left[key] - right[key] for key in left}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-per-class-dir", required=True)
    parser.add_argument("--ablation-per-class-dir", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    control = read_cells(
        Path(args.control_per_class_dir),
        "c1_rseed{r}_sseed{s}.json",
        args.recovery_seeds,
        args.student_seeds,
    )
    values = {"controlled_c1": control}
    for arm in ARMS:
        values[arm] = read_cells(
            Path(args.ablation_per_class_dir),
            f"{arm}_rseed{{r}}_sseed{{s}}.json",
            args.recovery_seeds,
            args.student_seeds,
        )

    summary = {
        "protocol": (
            "ImageNette IPC10 ResNet18 C1 Teacher/Patch crossed ablation, "
            "FKD batch10, T20, full 3925-image test"
        ),
        "recovery_seeds": args.recovery_seeds,
        "student_seeds": args.student_seeds,
        "arms": {
            name: hierarchical_summary(cells, args.recovery_seeds, args.student_seeds)
            for name, cells in values.items()
        },
        "paired_vs_controlled_c1": {},
    }
    for arm in ARMS:
        summary["paired_vs_controlled_c1"][f"{arm}_minus_controlled_c1"] = (
            hierarchical_summary(
                paired(values[arm], control),
                args.recovery_seeds,
                args.student_seeds,
            )
        )
    summary["paired_teacher_vs_patch_swap"] = hierarchical_summary(
        paired(
            values["official_teacher_c1_patches"],
            values["c1_teacher_official_patches"],
        ),
        args.recovery_seeds,
        args.student_seeds,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
