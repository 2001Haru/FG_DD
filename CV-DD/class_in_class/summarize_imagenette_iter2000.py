import argparse
import json
from pathlib import Path

from summarize_post_eval_seeds import hierarchical_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class-dir", required=True)
    parser.add_argument("--recovery-seed", type=int, required=True)
    parser.add_argument("--recovery-iterations", type=int, required=True)
    parser.add_argument("--recovery-lr", type=float, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.per_class_dir)
    values = {arm: {} for arm in ("official", "controlled_seed42")}
    for arm in values:
        for student_seed in args.student_seeds:
            path = root / f"{arm}_rseed{args.recovery_seed}_sseed{student_seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("validation_images", -1)) != 3925:
                raise ValueError(f"invalid validation set: {path}")
            values[arm][(args.recovery_seed, student_seed)] = float(payload["best_top1"])
    paired = {
        key: values["official"][key] - values["controlled_seed42"][key]
        for key in values["official"]
    }
    summary = {
        "protocol": (
            f"ImageNette IPC10 ResNet18 Recovery iteration{args.recovery_iterations} "
            f"LR{args.recovery_lr:g}; official vs official-split controlled "
            "seed42; BS10 T20 epochs300"
        ),
        "recovery_seeds": [args.recovery_seed],
        "student_seeds": args.student_seeds,
        "arms": {
            arm: hierarchical_summary(
                cells, [args.recovery_seed], args.student_seeds
            ) for arm, cells in values.items()
        },
        "paired_official_minus_controlled": hierarchical_summary(
            paired, [args.recovery_seed], args.student_seeds
        ),
        "recovery_4000_lr0p25_references": {
            "official_mean": 61.33,
            "controlled_seed42_mean": 58.96,
            "note": (
                "Descriptive references for recovery iteration4000 LR0.25. The "
                "old official relabel did not fix the same global runtime seed."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
