import argparse
import json
from pathlib import Path

from summarize_post_eval_seeds import hierarchical_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class-dir", required=True)
    parser.add_argument("--recovery-seed", type=int, required=True)
    parser.add_argument("--student-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.per_class_dir)
    cells = {}
    for student_seed in args.student_seeds:
        path = root / f"rseed{args.recovery_seed}_sseed{student_seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("validation_images", -1)) != 3925:
            raise ValueError(f"{path} was not evaluated on 3925 images")
        cells[(args.recovery_seed, student_seed)] = float(payload["best_top1"])

    summary = {
        "protocol": (
            "ImageNette IPC10 ResNet18 official Teacher+patches; recovery iter4000 "
            f"seed{args.recovery_seed}; relabel/post BS16; AdamW LR0.001; "
            "eta2; T20; 300 epochs"
        ),
        "recovery_seeds": [args.recovery_seed],
        "student_seeds": args.student_seeds,
        "arm": hierarchical_summary(
            cells, [args.recovery_seed], args.student_seeds
        ),
        "repository_result_png_reference": 62.4,
    }
    summary["mean_minus_repository_reference"] = (
        summary["arm"]["grand_mean"] - 62.4
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
