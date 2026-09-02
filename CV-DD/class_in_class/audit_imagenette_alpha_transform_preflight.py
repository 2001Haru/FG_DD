import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fkd-root", required=True)
    parser.add_argument("--max-softmax-mae", type=float, default=2e-4)
    parser.add_argument("--max-softmax-error", type=float, default=2e-3)
    args = parser.parse_args()
    root = Path(args.fkd_root)
    rows = []
    for family in ("c1_e300", "c100_e100"):
        for recovery in (41, 42):
            path = root / f"{family}_rseed{recovery}" / "alpha_transform_summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            replay = payload["alpha1_softmax_replay"]
            if replay["values"] <= 0:
                raise RuntimeError(f"alpha=1 was not generated: {path}")
            if replay["mae"] > args.max_softmax_mae:
                raise RuntimeError(f"alpha=1 softmax MAE too high: {path}: {replay}")
            if replay["maximum_absolute_error"] > args.max_softmax_error:
                raise RuntimeError(f"alpha=1 softmax max error too high: {path}: {replay}")
            for alpha_row in payload["alpha_rows"]:
                if alpha_row["batch_files"] != 3000:
                    raise RuntimeError(f"incomplete alpha output: {alpha_row}")
                if abs(
                    alpha_row["R_ratio_observed_by_algebra"]
                    - alpha_row["R_ratio_expected"]
                ) > 1e-12:
                    raise RuntimeError(f"R scaling identity failed: {alpha_row}")
            rows.append({
                "family": family,
                "recovery": recovery,
                "global_centered_logit_sd": payload["global_centered_logit_sd"],
                "sigma": payload["sigma"],
                "base_R": payload["base_normalized_decomposition"][
                    "R_within_over_between"
                ],
                "alpha1_softmax_mae": replay["mae"],
                "alpha1_softmax_max_error": replay["maximum_absolute_error"],
            })
    print(json.dumps({"passed": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
