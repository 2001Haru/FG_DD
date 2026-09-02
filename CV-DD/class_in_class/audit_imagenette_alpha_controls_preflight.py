import argparse
import json
import math
from pathlib import Path


PROTOCOLS = {
    "constantS_T1": {
        "student_temperature": 1.0,
        "constant_total_trace": True,
        "alphas": (0.70, 0.85, 1.00, 1.20, 1.50, 1.80),
    },
    "rawS_T20": {
        "student_temperature": 20.0,
        "constant_total_trace": False,
        "alphas": (1.00, 1.20, 1.50, 1.80),
    },
}
FAMILIES = ("c1_e300", "c100_e100")
RSEEDS = (41, 42)


def alpha_key(value):
    return round(float(value), 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fkd-root", required=True)
    parser.add_argument("--max-softmax-mae", type=float, default=2e-4)
    parser.add_argument("--max-softmax-error", type=float, default=2e-3)
    parser.add_argument("--trace-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    root = Path(args.fkd_root)
    rows = []
    for protocol, expected in PROTOCOLS.items():
        for family in FAMILIES:
            for recovery in RSEEDS:
                path = (
                    root / protocol / f"{family}_rseed{recovery}"
                    / "alpha_transform_summary.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not math.isclose(
                    float(payload["student_temperature"]),
                    expected["student_temperature"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(f"student temperature mismatch: {path}")
                if bool(payload["constant_total_trace"]) != expected["constant_total_trace"]:
                    raise RuntimeError(f"total-trace policy mismatch: {path}")
                expected_output_scale = (
                    float(payload["global_centered_logit_sd"])
                    * expected["student_temperature"]
                    / float(payload["source_temperature"])
                )
                if not math.isclose(
                    float(payload["output_scale"]),
                    expected_output_scale,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(f"stored-logit scale mismatch: {path}")

                replay = payload["alpha1_softmax_replay"]
                if replay["values"] <= 0:
                    raise RuntimeError(f"alpha=1 was not generated: {path}")
                if replay["mae"] > args.max_softmax_mae:
                    raise RuntimeError(f"alpha=1 softmax MAE too high: {path}: {replay}")
                if replay["maximum_absolute_error"] > args.max_softmax_error:
                    raise RuntimeError(f"alpha=1 softmax max error too high: {path}: {replay}")

                alpha_rows = {
                    alpha_key(row["alpha"]): row for row in payload["alpha_rows"]
                }
                if set(alpha_rows) != {alpha_key(value) for value in expected["alphas"]}:
                    raise RuntimeError(f"alpha grid mismatch: {path}")
                decomposition = payload["base_normalized_decomposition"]
                between = float(decomposition["between_trace"])
                within = float(decomposition["within_trace"])
                for alpha, row in alpha_rows.items():
                    if row["batch_files"] != 3000:
                        raise RuntimeError(f"incomplete transformed FKD: {row}")
                    if not math.isclose(
                        row["R_ratio_observed_by_algebra"],
                        alpha * alpha,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(f"R scaling identity failed: {row}")
                    if expected["constant_total_trace"] and not math.isclose(
                        row["total_trace_ratio_to_alpha1"],
                        1.0,
                        rel_tol=0.0,
                        abs_tol=args.trace_tolerance,
                    ):
                        raise RuntimeError(f"constant-S identity failed: {row}")
                    expected_factor = 1.0
                    if expected["constant_total_trace"]:
                        expected_factor = math.sqrt(
                            (between + within) / (between + alpha * alpha * within)
                        )
                    if not math.isclose(
                        row["total_trace_renormalization_factor"],
                        expected_factor,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(f"total-trace factor mismatch: {row}")

                rows.append(
                    {
                        "protocol": protocol,
                        "family": family,
                        "recovery": recovery,
                        "student_temperature": payload["student_temperature"],
                        "constant_total_trace": payload["constant_total_trace"],
                        "base_R": payload["base_normalized_decomposition"][
                            "R_within_over_between"
                        ],
                        "renormalization_factors": {
                            f"{alpha:g}": row["total_trace_renormalization_factor"]
                            for alpha, row in alpha_rows.items()
                        },
                        "alpha1_softmax_mae": replay["mae"],
                        "alpha1_softmax_max_error": replay["maximum_absolute_error"],
                    }
                )

    print(json.dumps({"passed": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
