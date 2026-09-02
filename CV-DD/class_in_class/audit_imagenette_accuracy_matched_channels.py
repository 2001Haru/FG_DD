import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from audit_imagenette_best_teacher_channels import (
    MEAN,
    STD,
    cca_spectrum,
    class_means_and_within,
    linear_cka,
    load_model,
    principal_angle_report,
)


EPOCHS = (4, 8, 16, 32, 64, 100, 150, 200, 250, 300)
SEEDS = (43, 44)
TEMPERATURE = 20.0
MATCHED_PAIRS = (
    ("c1e16_c100e32", 16, 32, "nearest_accuracy"),
    ("c1e32_c100e64", 32, 64, "nearest_accuracy"),
    ("c1e32_c100e100", 32, 100, "c1e32_one_to_many"),
    ("c1e32_c100e150", 32, 150, "c1e32_one_to_many"),
    ("c1e32_c100e200", 32, 200, "c1e32_one_to_many"),
    ("c1e32_c100e250", 32, 250, "c1e32_one_to_many"),
    ("c1e32_c100e300", 32, 300, "c1e32_one_to_many"),
)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def marginal_probability(logits, subclasses):
    logits = logits.double().view(logits.shape[0], 10, subclasses)
    parent_logits = TEMPERATURE * torch.logsumexp(logits / TEMPERATURE, dim=2)
    return torch.softmax(parent_logits / TEMPERATURE, dim=1).float()


def equivalent_logits(probabilities):
    logits = TEMPERATURE * probabilities.double().clamp_min(1e-30).log()
    return logits - logits.mean(1, keepdim=True)


@torch.inference_mode()
def collect(args):
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    if len(dataset) != 3925 or len(dataset.classes) != 10:
        raise ValueError("expected complete ImageNette test split")
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        options["prefetch_factor"] = 4
    loader = DataLoader(dataset, **options)
    trajectory = Path(args.trajectory_root)
    metrics = {}
    for seed in SEEDS:
        root = trajectory / f"tseed{seed}" / "models" / f"c{args.C}_tseed{seed}"
        metrics[seed] = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    matrices = {str(seed): {} for seed in SEEDS}
    metric_rows = {str(seed): {} for seed in SEEDS}
    targets_reference = None
    for epoch in EPOCHS:
        index = epoch - 1
        models_by_seed = {}
        for seed in SEEDS:
            checkpoint = (
                trajectory / f"tseed{seed}" / "models" / f"c{args.C}_tseed{seed}"
                / "checkpoints" / f"epoch_{index:03d}.pth"
            )
            models_by_seed[seed] = load_model(checkpoint, 10 * args.C, device)
            record = metrics[seed][index]
            metric_rows[str(seed)][str(epoch)] = {
                "train_native_accuracy": float(record["train_acc"]),
                "val_coarse_accuracy": float(record["val_acc"]),
                "checkpoint": str(checkpoint),
            }
        chunks = {seed: [] for seed in SEEDS}
        targets_all = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            for seed in SEEDS:
                chunks[seed].append(
                    marginal_probability(models_by_seed[seed](images), args.C).cpu()
                )
            targets_all.append(targets.cpu())
        targets = torch.cat(targets_all).long()
        if targets_reference is None:
            targets_reference = targets
        elif not torch.equal(targets_reference, targets):
            raise ValueError("test target order changed")
        for seed in SEEDS:
            matrices[str(seed)][str(epoch)] = torch.cat(chunks[seed]).float()
        print(json.dumps({"C": args.C, "epoch": epoch}), flush=True)
        del models_by_seed, chunks
        torch.cuda.empty_cache()
    payload = {
        "audit_schema_version": 1,
        "C": args.C,
        "family": f"C{args.C}",
        "temperature": TEMPERATURE,
        "epochs": list(EPOCHS),
        "seeds": list(SEEDS),
        "images": len(dataset),
        "class_names": list(dataset.classes),
        "sample_paths": [str(Path(path).relative_to(args.test_root)) for path, _ in dataset.samples],
        "targets": targets_reference,
        "probabilities": matrices,
        "trajectory_metrics": metric_rows,
    }
    atomic_torch_save(payload, args.output)


def light_geometry(left, right, targets):
    left_means, left_within = class_means_and_within(left, targets)
    right_means, right_within = class_means_and_within(right, targets)
    angles = principal_angle_report(left_within, right_within)
    cca = cca_spectrum(left_within, right_within)
    return {
        "global_CKA": linear_cka(left, right),
        "within_CKA": linear_cka(left_within, right_within),
        "between_CKA": linear_cka(left_means, right_means),
        "within_CCA_sum_rho2": cca["sum_squared_canonical_correlations"],
        "within_CCA_spectrum": cca["canonical_correlations"],
        "top1_direction_angle_degrees": angles["top_eigenvector_angle_degrees"],
        "top3_mean_angle_degrees": angles["subspaces"]["top_3"]["mean_angle_degrees"],
        "top5_mean_angle_degrees": angles["subspaces"]["top_5"]["mean_angle_degrees"],
    }


def mean(values):
    return sum(values) / len(values)


def analyze_cross_pair(name, c1_epoch, c100_epoch, group, data, space):
    targets = data["targets"]
    representations = data[space]
    self_c1 = light_geometry(
        representations["C1"][43][c1_epoch],
        representations["C1"][44][c1_epoch], targets,
    )
    self_c100 = light_geometry(
        representations["C100"][43][c100_epoch],
        representations["C100"][44][c100_epoch], targets,
    )
    cross = [
        light_geometry(
            representations["C1"][seed][c1_epoch],
            representations["C100"][seed][c100_epoch], targets,
        )
        for seed in SEEDS
    ]
    cross_mean = {
        key: mean([row[key] for row in cross])
        for key in (
            "global_CKA", "within_CKA", "between_CKA",
            "within_CCA_sum_rho2", "top1_direction_angle_degrees",
            "top3_mean_angle_degrees", "top5_mean_angle_degrees",
        )
    }
    cka_ceiling = math.sqrt(self_c1["within_CKA"] * self_c100["within_CKA"])
    cca_ceiling = math.sqrt(
        self_c1["within_CCA_sum_rho2"] * self_c100["within_CCA_sum_rho2"]
    )
    c1_accuracy = data["mean_val_accuracy"]["C1"][c1_epoch]
    c100_accuracy = data["mean_val_accuracy"]["C100"][c100_epoch]
    return {
        "name": name,
        "group": group,
        "space": space,
        "c1_epoch": c1_epoch,
        "c100_epoch": c100_epoch,
        "c1_val_accuracy": c1_accuracy,
        "c100_val_accuracy": c100_accuracy,
        "absolute_accuracy_gap": abs(c1_accuracy - c100_accuracy),
        "self_c1": self_c1,
        "self_c100": self_c100,
        "cross_seed43": cross[0],
        "cross_seed44": cross[1],
        "cross_mean": cross_mean,
        "deattenuated_within_CKA": cross_mean["within_CKA"] / max(cka_ceiling, 1e-30),
        "deattenuated_CCA_sum_rho2": cross_mean["within_CCA_sum_rho2"] / max(cca_ceiling, 1e-30),
        "cross_over_c1_angle_baseline": cross_mean["top1_direction_angle_degrees"] / max(self_c1["top1_direction_angle_degrees"], 1e-30),
        "cross_over_c100_angle_baseline": cross_mean["top1_direction_angle_degrees"] / max(self_c100["top1_direction_angle_degrees"], 1e-30),
    }


def within_family_calibration(data):
    targets = data["targets"]
    rows = []
    for family in ("C1", "C100"):
        for seed in SEEDS:
            for left_index, left_epoch in enumerate(EPOCHS):
                for right_epoch in EPOCHS[left_index + 1:]:
                    geometry = light_geometry(
                        data["logit"][family][seed][left_epoch],
                        data["logit"][family][seed][right_epoch], targets,
                    )
                    left_accuracy = data["val_accuracy"][family][seed][left_epoch]
                    right_accuracy = data["val_accuracy"][family][seed][right_epoch]
                    rows.append({
                        "family": family,
                        "seed": seed,
                        "left_epoch": left_epoch,
                        "right_epoch": right_epoch,
                        "epoch_gap": right_epoch - left_epoch,
                        "left_val_accuracy": left_accuracy,
                        "right_val_accuracy": right_accuracy,
                        "absolute_accuracy_gap": abs(left_accuracy - right_accuracy),
                        **geometry,
                    })
    return rows


def percentile(values, observation):
    if not values:
        return None
    return sum(value <= observation for value in values) / len(values)


def calibration_for_cross(cross_row, calibration_rows):
    angle = cross_row["cross_mean"]["top1_direction_angle_degrees"]
    result = {}
    for family in ("C1", "C100"):
        rows = [row for row in calibration_rows if row["family"] == family]
        nearest = sorted(
            rows, key=lambda row: abs(row["top1_direction_angle_degrees"] - angle)
        )[:5]
        reaching = [row for row in rows if row["top1_direction_angle_degrees"] >= angle]
        result[family] = {
            "cross_angle_degrees": angle,
            "median_accuracy_gap_of_five_nearest_angles": statistics.median(
                row["absolute_accuracy_gap"] for row in nearest
            ),
            "five_nearest_angle_pairs": nearest,
            "minimum_accuracy_gap_with_angle_at_least_cross": (
                min(row["absolute_accuracy_gap"] for row in reaching)
                if reaching else None
            ),
            "angle_percentile_among_all_within_family_pairs": percentile(
                [row["top1_direction_angle_degrees"] for row in rows], angle
            ),
            "angle_percentile_among_pairs_with_accuracy_gap_le_1": percentile(
                [row["top1_direction_angle_degrees"] for row in rows
                 if row["absolute_accuracy_gap"] <= 1.0], angle
            ),
            "angle_percentile_among_pairs_with_accuracy_gap_le_2": percentile(
                [row["top1_direction_angle_degrees"] for row in rows
                 if row["absolute_accuracy_gap"] <= 2.0], angle
            ),
        }
    return result


def calibration_bins(rows):
    bins = ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, float("inf")))
    output = []
    for family in ("C1", "C100"):
        family_rows = [row for row in rows if row["family"] == family]
        for lower, upper in bins:
            selected = [
                row for row in family_rows
                if row["absolute_accuracy_gap"] > lower
                and row["absolute_accuracy_gap"] <= upper
            ]
            if not selected:
                continue
            output.append({
                "family": family,
                "accuracy_gap_lower_exclusive": lower,
                "accuracy_gap_upper_inclusive": upper,
                "pairs": len(selected),
                "mean_angle_degrees": mean(
                    [row["top1_direction_angle_degrees"] for row in selected]
                ),
                "median_angle_degrees": statistics.median(
                    row["top1_direction_angle_degrees"] for row in selected
                ),
                "mean_within_CKA": mean([row["within_CKA"] for row in selected]),
                "mean_CCA_sum_rho2": mean(
                    [row["within_CCA_sum_rho2"] for row in selected]
                ),
            })
    return output


def analyze(args):
    payloads = {
        "C1": torch.load(args.c1, map_location="cpu", weights_only=False),
        "C100": torch.load(args.c100, map_location="cpu", weights_only=False),
    }
    reference = payloads["C1"]
    for family, payload in payloads.items():
        if payload["images"] != 3925 or payload["epochs"] != list(EPOCHS):
            raise ValueError(f"invalid collection: {family}")
        if payload["sample_paths"] != reference["sample_paths"]:
            raise ValueError("sample order mismatch")
        if not torch.equal(payload["targets"], reference["targets"]):
            raise ValueError("target mismatch")
    data = {
        "targets": reference["targets"].long(),
        "probability": {"C1": {}, "C100": {}},
        "logit": {"C1": {}, "C100": {}},
        "val_accuracy": {"C1": {}, "C100": {}},
        "mean_val_accuracy": {"C1": {}, "C100": {}},
    }
    for family, payload in payloads.items():
        for seed in SEEDS:
            data["probability"][family][seed] = {}
            data["logit"][family][seed] = {}
            data["val_accuracy"][family][seed] = {}
            for epoch in EPOCHS:
                q = payload["probabilities"][str(seed)][str(epoch)].double()
                data["probability"][family][seed][epoch] = q
                data["logit"][family][seed][epoch] = equivalent_logits(q)
                data["val_accuracy"][family][seed][epoch] = float(
                    payload["trajectory_metrics"][str(seed)][str(epoch)][
                        "val_coarse_accuracy"
                    ]
                )
        for epoch in EPOCHS:
            data["mean_val_accuracy"][family][epoch] = mean(
                [data["val_accuracy"][family][seed][epoch] for seed in SEEDS]
            )

    matched = []
    for name, c1_epoch, c100_epoch, group in MATCHED_PAIRS:
        probability = analyze_cross_pair(
            name, c1_epoch, c100_epoch, group, data, "probability"
        )
        logit = analyze_cross_pair(
            name, c1_epoch, c100_epoch, group, data, "logit"
        )
        matched.append({
            "name": name,
            "group": group,
            "c1_epoch": c1_epoch,
            "c100_epoch": c100_epoch,
            "c1_val_accuracy": logit["c1_val_accuracy"],
            "c100_val_accuracy": logit["c100_val_accuracy"],
            "absolute_accuracy_gap": logit["absolute_accuracy_gap"],
            "probability_space": probability,
            "logit_space": logit,
        })
    calibration_rows = within_family_calibration(data)
    for row in matched:
        row["logit_angle_calibration"] = calibration_for_cross(
            row["logit_space"], calibration_rows
        )
    binned = calibration_bins(calibration_rows)
    result = {
        "audit_schema_version": 1,
        "question": (
            "How large is the cross-family channel rotation at accuracy-matched "
            "checkpoints, calibrated against within-family epoch-pair rotations?"
        ),
        "protocol": {
            "dataset": "complete 3925-image ImageNette test split",
            "temperature": TEMPERATURE,
            "teacher_seeds": list(SEEDS),
            "epochs": list(EPOCHS),
            "primary_space": "equivalent centered logits",
            "matched_pairs": [list(row) for row in MATCHED_PAIRS],
        },
        "mean_val_accuracy": data["mean_val_accuracy"],
        "matched_cross_family_pairs": matched,
        "within_family_calibration_rows": calibration_rows,
        "within_family_calibration_bins": binned,
    }
    atomic_json_dump(result, args.output_json)
    for path in (
        args.output_pairs_csv, args.output_calibration_csv, args.output_bins_csv
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_pairs_csv).open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "name", "group", "c1_epoch", "c100_epoch", "c1_val_accuracy",
            "c100_val_accuracy", "absolute_accuracy_gap", "space",
            "cross_within_CKA", "deattenuated_within_CKA",
            "cross_CCA_sum_rho2", "deattenuated_CCA_sum_rho2",
            "cross_top1_angle_degrees", "cross_over_c1_angle_baseline",
            "cross_over_c100_angle_baseline", "between_CKA",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in matched:
            for space_key, label in (
                ("probability_space", "probability"), ("logit_space", "logit")
            ):
                value = row[space_key]
                writer.writerow({
                    "name": row["name"], "group": row["group"],
                    "c1_epoch": row["c1_epoch"], "c100_epoch": row["c100_epoch"],
                    "c1_val_accuracy": row["c1_val_accuracy"],
                    "c100_val_accuracy": row["c100_val_accuracy"],
                    "absolute_accuracy_gap": row["absolute_accuracy_gap"],
                    "space": label,
                    "cross_within_CKA": value["cross_mean"]["within_CKA"],
                    "deattenuated_within_CKA": value["deattenuated_within_CKA"],
                    "cross_CCA_sum_rho2": value["cross_mean"]["within_CCA_sum_rho2"],
                    "deattenuated_CCA_sum_rho2": value["deattenuated_CCA_sum_rho2"],
                    "cross_top1_angle_degrees": value["cross_mean"]["top1_direction_angle_degrees"],
                    "cross_over_c1_angle_baseline": value["cross_over_c1_angle_baseline"],
                    "cross_over_c100_angle_baseline": value["cross_over_c100_angle_baseline"],
                    "between_CKA": value["cross_mean"]["between_CKA"],
                })
    with Path(args.output_calibration_csv).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(calibration_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(calibration_rows)
    with Path(args.output_bins_csv).open("w", newline="", encoding="utf-8") as handle:
        fields = list(binned[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(binned)
    print(json.dumps({
        "output_json": args.output_json,
        "matched_pairs": len(matched),
        "within_family_calibration_pairs": len(calibration_rows),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--trajectory-root", required=True)
    collect_parser.add_argument("--test-root", required=True)
    collect_parser.add_argument("--C", type=int, choices=(1, 100), required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--batch-size", type=int, default=256)
    collect_parser.add_argument("--workers", type=int, default=8)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--c1", required=True)
    analyze_parser.add_argument("--c100", required=True)
    analyze_parser.add_argument("--output-json", required=True)
    analyze_parser.add_argument("--output-pairs-csv", required=True)
    analyze_parser.add_argument("--output-calibration-csv", required=True)
    analyze_parser.add_argument("--output-bins-csv", required=True)
    args = parser.parse_args()
    if args.command == "collect": collect(args)
    else: analyze(args)


if __name__ == "__main__":
    main()
