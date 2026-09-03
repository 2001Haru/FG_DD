import argparse
import json
import math
import statistics
from pathlib import Path

from audit_imagenette_consumed_fkd_labels import summarize_roots


TEACHER_SEEDS = (43, 44)
RECOVERY_SEEDS = (41, 42, 43)
IMAGE_FIELDS = (
    "cluster_teacher_native100_top1",
    "cluster_teacher_collapsed_coarse10_top1",
    "c1_teacher_coarse10_top1",
    "cluster_c1_coarse_prediction_agreement",
    "both_coarse_correct_fraction",
    "cluster_correct_c1_wrong_fraction",
    "c1_correct_cluster_wrong_fraction",
    "mean_cluster_native_target_probability",
    "mean_cluster_native_target_nll",
    "mean_cluster_coarse_target_probability",
    "mean_c1_target_probability_T1",
    "mean_c1_target_probability_T20",
    "mean_cluster_native_entropy_T1",
    "mean_cluster_coarse_entropy_T1",
    "mean_c1_entropy_T1",
    "mean_c1_entropy_T20",
)


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_field(rows, field):
    values = [float(row[field]) for row in rows]
    teacher_means = []
    recovery_variances = []
    for teacher in TEACHER_SEEDS:
        current = [
            float(row[field]) for row in rows if row["teacher_seed"] == teacher
        ]
        if len(current) != len(RECOVERY_SEEDS):
            raise RuntimeError(f"incomplete roots for teacher={teacher}, field={field}")
        teacher_means.append(statistics.fmean(current))
        recovery_variances.append(sample_sd(current) ** 2)
    return {
        "mean_across_six_teacher_recovery_roots": statistics.fmean(values),
        "sample_sd_across_six_roots": sample_sd(values),
        "pooled_recovery_seed_sd_within_teacher": math.sqrt(
            statistics.fmean(recovery_variances)
        ),
        "teacher_seed_sd_of_recovery_means": sample_sd(teacher_means),
        "by_teacher_seed_mean": {
            str(teacher): teacher_means[index]
            for index, teacher in enumerate(TEACHER_SEEDS)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    payloads = [
        json.loads((input_dir / f"tseed{teacher}.json").read_text(encoding="utf-8"))
        for teacher in TEACHER_SEEDS
    ]
    image_rows = [row for payload in payloads for row in payload["image_metrics"]]
    fkd_rows = [row for payload in payloads for row in payload["consumed_fkd_metrics"]]
    image_rows.sort(key=lambda row: (row["teacher_seed"], row["recovery_seed"]))
    fkd_rows.sort(key=lambda row: (row["teacher_seed"], row["recovery_seed"]))
    expected_keys = {
        (teacher, recovery)
        for teacher in TEACHER_SEEDS for recovery in RECOVERY_SEEDS
    }
    if {(row["teacher_seed"], row["recovery_seed"]) for row in image_rows} != expected_keys:
        raise RuntimeError("image-audit roots are incomplete")
    if {(row["teacher_seed"], row["recovery_seed"]) for row in fkd_rows} != expected_keys:
        raise RuntimeError("FKD-audit roots are incomplete")

    per_parent = []
    for parent in range(10):
        cluster = [row["per_parent"][parent]["cluster_coarse_top1"] for row in image_rows]
        c1 = [row["per_parent"][parent]["c1_coarse_top1"] for row in image_rows]
        per_parent.append({
            "parent": parent,
            "cluster_coarse_top1_mean": statistics.fmean(cluster),
            "cluster_coarse_top1_sd_across_roots": sample_sd(cluster),
            "c1_coarse_top1_mean": statistics.fmean(c1),
            "c1_coarse_top1_sd_across_roots": sample_sd(c1),
        })

    result = {
        "audit_schema_version": 1,
        "protocol": (
            "Existing ImageNette DINO Cluster C10 native-100 recovery outputs; "
            "exact-pixel Cluster/C1 Teacher audit plus reconstructed student-consumed "
            "C1 CutMix FKD-label audit"
        ),
        "teacher_seeds": list(TEACHER_SEEDS),
        "recovery_seeds": list(RECOVERY_SEEDS),
        "image_metric_summary": {
            field: summarize_field(image_rows, field) for field in IMAGE_FIELDS
        },
        "per_parent_summary": per_parent,
        "consumed_fkd_summary": summarize_roots(
            fkd_rows, TEACHER_SEEDS, RECOVERY_SEEDS
        ),
        "image_metrics_by_root": image_rows,
        "consumed_fkd_metrics_by_root": fkd_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
