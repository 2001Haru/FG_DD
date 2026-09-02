import argparse
import csv
import json
from pathlib import Path


def load(path, expected_epoch):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    for family in ("C1", "Random_C100"):
        spec = protocol[family]
        if int(spec["training_epoch"]) != expected_epoch:
            raise ValueError(f"{family} epoch mismatch in {path}")
        if float(spec["temperature"]) != 20.0:
            raise ValueError(f"{family} temperature must be T20 in {path}")
    return payload


def row_for_pair(phase, report, space):
    angles = report["within_class_covariance_principal_angles"][
        "pooled_within_class_residuals"
    ]
    return {
        "phase": phase,
        "space": space,
        "left": report["left"],
        "right": report["right"],
        "relation": report["relation"],
        "global_CKA": report["linear_CKA"]["globally_centered_labels"],
        "within_CKA": report["linear_CKA"]["within_class_centered_labels"],
        "between_CKA": report["linear_CKA"]["between_class_prototypes"],
        "within_CCA_sum_rho2": report["CCA"]["within_class_centered_labels"][
            "sum_squared_canonical_correlations"
        ],
        "within_CCA_dims_ge_0p9": report["CCA"]["within_class_centered_labels"][
            "shared_dimensions_rho_ge_0p9"
        ],
        "within_top1_direction_angle_degrees": angles[
            "top_eigenvector_angle_degrees"
        ],
        "top1_agreement": report.get("agreement", {}).get("top1_agreement_fraction"),
        "top5_overlap": report.get("agreement", {}).get("mean_top5_overlap_count"),
        "centered_interclass_similarity_pearson": report[
            "interclass_similarity"
        ]["globally_centered_prototype_cosine"]["comparison"][
            "upper_triangle_pearson"
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e100", required=True)
    parser.add_argument("--e300", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    payloads = {
        "e100": load(args.e100, 100),
        "e300": load(args.e300, 300),
    }
    rows = [
        row_for_pair(phase, report, "probability")
        for phase, payload in payloads.items()
        for report in payload["pair_reports"]
    ]
    rows.extend(
        row_for_pair(phase, report, "equivalent_logit")
        for phase, payload in payloads.items()
        for report in payload["logit_pair_reports"]
    )
    result = {
        "audit_schema_version": 1,
        "question": (
            "Cross-family soft-label channel comparison at LR-phase-matched "
            "e100/e100 and e300/e300, with both families evaluated at T20; "
            "equivalent-logit geometry is primary and probability geometry is audit."
        ),
        "phases": {
            phase: {
                "protocol": payload["protocol"],
                "model_summaries": payload["model_summaries"],
                "logit_model_summaries": payload["logit_model_summaries"],
                "paired_cross_family_aggregate": payload[
                    "paired_cross_family_aggregate"
                ],
                "logit_paired_cross_family_aggregate": payload[
                    "logit_paired_cross_family_aggregate"
                ],
                "reliability_corrected_within_CKA": payload[
                    "reliability_corrected_within_CKA"
                ],
            }
            for phase, payload in payloads.items()
        },
        "pair_rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
