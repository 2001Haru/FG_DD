import argparse
import csv
import json
import statistics
from pathlib import Path


EPOCHS = (8, 16, 32, 64, 100, 150, 200, 250, 300)
SEEDS = (43, 44)


def mean_sd(values):
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "by_teacher_seed": dict(zip(map(str, SEEDS), values)),
    }


def nested(payload, path):
    for key in path.split("."):
        payload = payload[key]
    return float(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-root", required=True)
    parser.add_argument("--downstream-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.cam_root)
    downstream = json.loads(Path(args.downstream_summary).read_text(encoding="utf-8"))
    data = {}
    for seed in SEEDS:
        for c in (1, 100):
            payload = json.loads(
                (root / f"tseed{seed}" / f"c{c}" / "summary.json").read_text(encoding="utf-8")
            )
            for row in payload["rows"]:
                data[(seed, c, int(row["training_epoch"]))] = row

    metric_paths = (
        "parent_cam.normalized_spatial_entropy_all",
        "parent_cam.normalized_spatial_entropy_coarse_correct",
        "parent_cam.within_parent_centroid_variance_all",
        "parent_cam.within_parent_centroid_variance_coarse_correct",
        "parent_cam.zero_cam_fraction",
        "penultimate_features.centered_covariance_participation_rank",
        "penultimate_features.centered_covariance_entropy_effective_rank",
        "marginal_labels_T20.mean_entropy",
        "marginal_labels_T20.centered_equivalent_logit_sd",
    )
    rows, summaries, paired = [], {}, {}
    subhead_entropy_artifact_test = {}
    for epoch in EPOCHS:
        for c in (1, 100):
            key = f"c{c}_epoch{epoch:03d}"
            summaries[key] = {
                "coarse_accuracy": mean_sd([
                    float(data[(seed, c, epoch)]["coarse_accuracy"]) for seed in SEEDS
                ])
            }
            for path in metric_paths:
                summaries[key][path] = mean_sd([
                    nested(data[(seed, c, epoch)], path) for seed in SEEDS
                ])
            if c == 100:
                for path in (
                    "within_parent_top_subhead_cam.mean_pairwise_js_all",
                    "within_parent_top_subhead_cam.mean_pairwise_js_coarse_correct",
                    "within_parent_top_subhead_cam.mean_valid_pair_fraction",
                ):
                    summaries[key][path] = mean_sd([
                        nested(data[(seed, c, epoch)], path) for seed in SEEDS
                    ])
        pair_key = f"epoch{epoch:03d}_c100_minus_c1"
        paired[pair_key] = {}
        for path in metric_paths:
            paired[pair_key][path] = mean_sd([
                nested(data[(seed, 100, epoch)], path)
                - nested(data[(seed, 1, epoch)], path)
                for seed in SEEDS
            ])
        label = f"e{epoch:03d}"
        dd_real = downstream["comparisons"][
            f"same_label_{label}_real_ref_c100_minus_c1"
        ]["grand_mean"]
        dd_syn = downstream["comparisons"][
            f"same_label_{label}_c1_ref_c100_minus_c1"
        ]["grand_mean"]
        paired[pair_key]["downstream_T20"] = {
            "real_delta": dd_real,
            "c1_synthetic_delta": dd_syn,
            "equal_source_average_delta": (dd_real + dd_syn) / 2.0,
        }
        c1_parent_entropy = [
            nested(
                data[(seed, 1, epoch)],
                "parent_cam.normalized_spatial_entropy_all",
            ) for seed in SEEDS
        ]
        c100_parent_entropy = [
            nested(
                data[(seed, 100, epoch)],
                "parent_cam.normalized_spatial_entropy_all",
            ) for seed in SEEDS
        ]
        c100_top1_entropy = [
            nested(
                data[(seed, 100, epoch)],
                "within_parent_top_subhead_cam.top1_subhead_normalized_spatial_entropy_all",
            ) for seed in SEEDS
        ]
        c100_topk_entropy = [
            nested(
                data[(seed, 100, epoch)],
                "within_parent_top_subhead_cam.mean_topk_individual_normalized_spatial_entropy_all",
            ) for seed in SEEDS
        ]
        c1_parent_entropy_correct = [
            nested(
                data[(seed, 1, epoch)],
                "parent_cam.normalized_spatial_entropy_coarse_correct",
            ) for seed in SEEDS
        ]
        c100_parent_entropy_correct = [
            nested(
                data[(seed, 100, epoch)],
                "parent_cam.normalized_spatial_entropy_coarse_correct",
            ) for seed in SEEDS
        ]
        c100_top1_entropy_correct = [
            nested(
                data[(seed, 100, epoch)],
                "within_parent_top_subhead_cam.top1_subhead_normalized_spatial_entropy_coarse_correct",
            ) for seed in SEEDS
        ]
        c100_topk_entropy_correct = [
            nested(
                data[(seed, 100, epoch)],
                "within_parent_top_subhead_cam.mean_topk_individual_normalized_spatial_entropy_coarse_correct",
            ) for seed in SEEDS
        ]
        artifact_key = f"epoch{epoch:03d}"
        subhead_entropy_artifact_test[artifact_key] = {
            "c1_parent_entropy": mean_sd(c1_parent_entropy),
            "c100_parent_entropy": mean_sd(c100_parent_entropy),
            "c100_top1_subhead_entropy": mean_sd(c100_top1_entropy),
            "c100_mean_top5_individual_entropy": mean_sd(c100_topk_entropy),
            "aggregation_uplift_parent_minus_top1": mean_sd([
                parent - child
                for parent, child in zip(c100_parent_entropy, c100_top1_entropy)
            ]),
            "intrinsic_top1_subhead_minus_c1_parent": mean_sd([
                child - baseline
                for child, baseline in zip(c100_top1_entropy, c1_parent_entropy)
            ]),
            "intrinsic_mean_top5_minus_c1_parent": mean_sd([
                child - baseline
                for child, baseline in zip(c100_topk_entropy, c1_parent_entropy)
            ]),
            "coarse_correct": {
                "c1_parent_entropy": mean_sd(c1_parent_entropy_correct),
                "c100_parent_entropy": mean_sd(c100_parent_entropy_correct),
                "c100_top1_subhead_entropy": mean_sd(c100_top1_entropy_correct),
                "c100_mean_top5_individual_entropy": mean_sd(c100_topk_entropy_correct),
                "aggregation_uplift_parent_minus_top1": mean_sd([
                    parent - child for parent, child in zip(
                        c100_parent_entropy_correct, c100_top1_entropy_correct
                    )
                ]),
                "intrinsic_top1_subhead_minus_c1_parent": mean_sd([
                    child - baseline for child, baseline in zip(
                        c100_top1_entropy_correct, c1_parent_entropy_correct
                    )
                ]),
                "intrinsic_mean_top5_minus_c1_parent": mean_sd([
                    child - baseline for child, baseline in zip(
                        c100_topk_entropy_correct, c1_parent_entropy_correct
                    )
                ]),
            },
        }
        c100 = data[(43, 100, epoch)]
        row = {
            "epoch": epoch,
            "dd_equal_source_average_delta": (dd_real + dd_syn) / 2.0,
            "c100_subhead_js": statistics.fmean([
                nested(data[(seed, 100, epoch)], "within_parent_top_subhead_cam.mean_pairwise_js_all")
                for seed in SEEDS
            ]),
            "c100_top1_subhead_entropy": statistics.fmean(c100_top1_entropy),
            "c100_mean_top5_individual_entropy": statistics.fmean(c100_topk_entropy),
            "aggregation_entropy_uplift": subhead_entropy_artifact_test[
                artifact_key
            ]["aggregation_uplift_parent_minus_top1"]["mean"],
            "intrinsic_subhead_entropy_delta": subhead_entropy_artifact_test[
                artifact_key
            ]["intrinsic_top1_subhead_minus_c1_parent"]["mean"],
        }
        for path in metric_paths:
            short = path.replace(".", "_")
            row[f"c1_{short}"] = summaries[f"c1_epoch{epoch:03d}"][path]["mean"]
            row[f"c100_{short}"] = summaries[f"c100_epoch{epoch:03d}"][path]["mean"]
            row[f"delta_{short}"] = paired[pair_key][path]["mean"]
        rows.append(row)

    preregistered = {}
    for epoch in (16, 64, 100):
        key = f"epoch{epoch:03d}_c100_minus_c1"
        preregistered[str(epoch)] = {
            "downstream_delta": paired[key]["downstream_T20"]["equal_source_average_delta"],
            "parent_cam_entropy_delta": paired[key][
                "parent_cam.normalized_spatial_entropy_all"
            ]["mean"],
            "parent_cam_centroid_variance_delta": paired[key][
                "parent_cam.within_parent_centroid_variance_all"
            ]["mean"],
            "penultimate_participation_rank_delta": paired[key][
                "penultimate_features.centered_covariance_participation_rank"
            ]["mean"],
            "c100_subhead_js": summaries[f"c100_epoch{epoch:03d}"][
                "within_parent_top_subhead_cam.mean_pairwise_js_all"
            ]["mean"],
        }

    result = {
        "protocol": (
            "Paired C1/C100 same-epoch Grad-CAM trajectory; parent CAM uses "
            "marginalized parent logit; C100 subhead JS uses top5 heads within true parent"
        ),
        "teacher_seeds": list(SEEDS),
        "training_epochs": list(EPOCHS),
        "summaries": summaries,
        "paired_c100_minus_c1": paired,
        "subhead_entropy_artifact_test": subhead_entropy_artifact_test,
        "preregistered_epochs": preregistered,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
