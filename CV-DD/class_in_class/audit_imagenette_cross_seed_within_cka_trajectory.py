import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from audit_imagenette_best_teacher_channels import (
    MEAN,
    STD,
    class_means_and_within,
    linear_cka,
    load_model,
)


EPOCHS = (4, 8, 16, 32, 64, 100, 150, 200, 250, 300)
TEACHER_SEEDS = (43, 44)


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def marginal_outputs(logits, subclasses, temperature):
    logits = logits.double().view(logits.shape[0], 10, subclasses)
    parent_logits = temperature * torch.logsumexp(logits / temperature, dim=2)
    probabilities = torch.softmax(parent_logits / temperature, dim=1)
    centered_logits = parent_logits - parent_logits.mean(1, keepdim=True)
    return probabilities.float(), centered_logits.float()


def cka_decomposition(left, right, targets):
    left_means, left_within = class_means_and_within(left, targets)
    right_means, right_within = class_means_and_within(right, targets)
    per_class = []
    for class_id in range(10):
        mask = targets.eq(class_id)
        per_class.append({
            "class_id": class_id,
            "images": int(mask.sum()),
            "within_class_CKA": linear_cka(left_within[mask], right_within[mask]),
        })
    return {
        "global_centered_CKA": linear_cka(left, right),
        "within_class_centered_CKA": linear_cka(left_within, right_within),
        "between_class_prototype_CKA": linear_cka(left_means, right_means),
        "mean_per_class_within_CKA": sum(
            row["within_class_CKA"] for row in per_class
        ) / len(per_class),
        "minimum_per_class_within_CKA": min(
            row["within_class_CKA"] for row in per_class
        ),
        "maximum_per_class_within_CKA": max(
            row["within_class_CKA"] for row in per_class
        ),
        "per_class": per_class,
    }


@torch.inference_mode()
def collect(args):
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    if len(dataset) != 3925 or len(dataset.classes) != 10:
        raise ValueError("expected complete 3925-image/10-class ImageNette test split")
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_options)
    trajectory_root = Path(args.trajectory_root)
    metrics = {}
    for seed in TEACHER_SEEDS:
        model_root = (
            trajectory_root / f"tseed{seed}" / "models" / f"c{args.C}_tseed{seed}"
        )
        metrics[seed] = json.loads(
            (model_root / "metrics.json").read_text(encoding="utf-8")
        )
        if len(metrics[seed]) != 300:
            raise ValueError(f"expected 300 trajectory records: {model_root}")

    rows = []
    output = Path(args.output)
    for training_epoch in EPOCHS:
        index = training_epoch - 1
        models_by_seed = {}
        checkpoints = {}
        for seed in TEACHER_SEEDS:
            checkpoint = (
                trajectory_root / f"tseed{seed}" / "models"
                / f"c{args.C}_tseed{seed}" / "checkpoints"
                / f"epoch_{index:03d}.pth"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            models_by_seed[seed] = load_model(checkpoint, 10 * args.C, device)
            checkpoints[str(seed)] = str(checkpoint)

        probabilities = {seed: [] for seed in TEACHER_SEEDS}
        equivalent_logits = {seed: [] for seed in TEACHER_SEEDS}
        targets_all = []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            for seed in TEACHER_SEEDS:
                logits = models_by_seed[seed](images)
                q, z = marginal_outputs(logits, args.C, args.temperature)
                probabilities[seed].append(q.cpu())
                equivalent_logits[seed].append(z.cpu())
            targets_all.append(targets.cpu())
        targets = torch.cat(targets_all).long()
        q43, q44 = (torch.cat(probabilities[seed]) for seed in TEACHER_SEEDS)
        z43, z44 = (torch.cat(equivalent_logits[seed]) for seed in TEACHER_SEEDS)
        probability_cka = cka_decomposition(q43, q44, targets)
        logit_cka = cka_decomposition(z43, z44, targets)
        row = {
            "C": args.C,
            "family": f"C{args.C}",
            "training_epoch": training_epoch,
            "checkpoint_epoch_index": index,
            "temperature": args.temperature,
            "images": len(dataset),
            "checkpoints": checkpoints,
            "seed43_train_native_accuracy": metrics[43][index]["train_acc"],
            "seed44_train_native_accuracy": metrics[44][index]["train_acc"],
            "mean_train_native_accuracy": 0.5 * (
                metrics[43][index]["train_acc"] + metrics[44][index]["train_acc"]
            ),
            "seed43_val_coarse_accuracy": metrics[43][index]["val_acc"],
            "seed44_val_coarse_accuracy": metrics[44][index]["val_acc"],
            "mean_val_coarse_accuracy": 0.5 * (
                metrics[43][index]["val_acc"] + metrics[44][index]["val_acc"]
            ),
            "probability_labels": probability_cka,
            "centered_equivalent_logits": logit_cka,
        }
        rows.append(row)
        payload = {
            "audit_schema_version": 1,
            "definition": (
                "Cross-Teacher-seed linear CKA on complete ImageNette test images. "
                "Primary metric removes each coarse class mean before CKA."
            ),
            "C": args.C,
            "family": f"C{args.C}",
            "teacher_seeds": list(TEACHER_SEEDS),
            "temperature": args.temperature,
            "training_epochs": list(EPOCHS),
            "class_names": list(dataset.classes),
            "rows": rows,
        }
        atomic_json_dump(payload, output)
        print(json.dumps({
            "C": args.C,
            "epoch": training_epoch,
            "train_accuracy": row["mean_train_native_accuracy"],
            "val_accuracy": row["mean_val_coarse_accuracy"],
            "within_probability_CKA": probability_cka["within_class_centered_CKA"],
            "within_logit_CKA": logit_cka["within_class_centered_CKA"],
        }), flush=True)
        del models_by_seed, probabilities, equivalent_logits, q43, q44, z43, z44
        torch.cuda.empty_cache()


def merge(args):
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in (args.c1, args.c100)
    ]
    by_family = {payload["family"]: payload for payload in payloads}
    if set(by_family) != {"C1", "C100"}:
        raise ValueError("merge requires C1 and C100 payloads")
    for payload in payloads:
        if payload["training_epochs"] != list(EPOCHS):
            raise ValueError(f"incomplete epochs: {payload['family']}")
        if len(payload["rows"]) != len(EPOCHS):
            raise ValueError(f"incomplete rows: {payload['family']}")
        if float(payload["temperature"]) != float(payloads[0]["temperature"]):
            raise ValueError("C1/C100 temperature mismatch")
    result = {
        "audit_schema_version": 1,
        "question": "How does cross-seed within-class CKA evolve with Teacher epoch?",
        "protocol": {
            "dataset": "complete 3925-image ImageNette test split",
            "temperature": payloads[0]["temperature"],
            "teacher_seeds": list(TEACHER_SEEDS),
            "training_epochs": list(EPOCHS),
            "primary_metric": "within-class-centered probability-label linear CKA",
            "audit_metric": "within-class-centered equivalent-logit linear CKA",
        },
        "families": by_family,
    }
    output_json = Path(args.output_json)
    atomic_json_dump(result, output_json)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "family", "C", "training_epoch", "temperature",
            "mean_train_native_accuracy", "mean_val_coarse_accuracy",
            "probability_global_CKA", "probability_within_CKA",
            "probability_between_CKA", "mean_per_class_probability_within_CKA",
            "minimum_per_class_probability_within_CKA",
            "maximum_per_class_probability_within_CKA",
            "logit_global_CKA", "logit_within_CKA", "logit_between_CKA",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in ("C1", "C100"):
            for row in by_family[family]["rows"]:
                probability = row["probability_labels"]
                logit = row["centered_equivalent_logits"]
                writer.writerow({
                    "family": family,
                    "C": row["C"],
                    "training_epoch": row["training_epoch"],
                    "temperature": row["temperature"],
                    "mean_train_native_accuracy": row["mean_train_native_accuracy"],
                    "mean_val_coarse_accuracy": row["mean_val_coarse_accuracy"],
                    "probability_global_CKA": probability["global_centered_CKA"],
                    "probability_within_CKA": probability["within_class_centered_CKA"],
                    "probability_between_CKA": probability["between_class_prototype_CKA"],
                    "mean_per_class_probability_within_CKA": probability["mean_per_class_within_CKA"],
                    "minimum_per_class_probability_within_CKA": probability["minimum_per_class_within_CKA"],
                    "maximum_per_class_probability_within_CKA": probability["maximum_per_class_within_CKA"],
                    "logit_global_CKA": logit["global_centered_CKA"],
                    "logit_within_CKA": logit["within_class_centered_CKA"],
                    "logit_between_CKA": logit["between_class_prototype_CKA"],
                })
    print(json.dumps({
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "families": list(by_family),
        "epochs": list(EPOCHS),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--trajectory-root", required=True)
    collect_parser.add_argument("--test-root", required=True)
    collect_parser.add_argument("--C", type=int, choices=(1, 100), required=True)
    collect_parser.add_argument("--temperature", type=float, default=20.0)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--batch-size", type=int, default=256)
    collect_parser.add_argument("--workers", type=int, default=8)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--c1", required=True)
    merge_parser.add_argument("--c100", required=True)
    merge_parser.add_argument("--output-json", required=True)
    merge_parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
