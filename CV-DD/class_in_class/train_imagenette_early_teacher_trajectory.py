import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from imagenette_subclass_dataset import EncodedSubclassFolder


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(history, path):
    fields = [
        "epoch", "lr", "lr_after_scheduler_step", "train_augmented_accuracy",
        "train_acc", "val_acc", "train_native_accuracy", "train_coarse_accuracy",
        "val_native_accuracy", "val_coarse_accuracy", "sd_z",
        "marg_label_entropy_T20", "participation_rank",
        "train_sd_z", "val_sd_z", "train_marg_entropy_T20",
        "val_marg_entropy_T20", "train_participation_rank",
        "val_participation_rank", "train_seconds", "evaluation_seconds",
    ]
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow({field: record.get(field) for field in fields})
    os.replace(temporary, path)


def make_loaders(root, batch_size, workers, seed, expected_classes):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    clean_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_augmented = datasets.ImageFolder(str(root / "train"), train_transform)
    train_clean = datasets.ImageFolder(str(root / "train"), clean_transform)
    val = EncodedSubclassFolder(
        root / "val", num_classes=expected_classes, transform=clean_transform
    )
    if len(train_augmented.classes) != expected_classes:
        raise RuntimeError(
            f"ImageFolder classes={len(train_augmented.classes)}, expected "
            f"{expected_classes}: {root}"
        )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_augmented, batch_size=batch_size, shuffle=True,
        generator=generator, **common,
    )
    train_clean_loader = DataLoader(
        train_clean, batch_size=256, shuffle=False, **common
    )
    val_loader = DataLoader(val, batch_size=256, shuffle=False, **common)
    return train_loader, train_clean_loader, val_loader


def covariance_participation_rank(probabilities):
    centered = probabilities.double() - probabilities.double().mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square()
    return float(
        eigenvalues.sum().square()
        / eigenvalues.square().sum().clamp_min(1e-15)
    )


@torch.inference_mode()
def evaluate_geometry(model, loader, fine_to_coarse, temperature):
    model.eval()
    native_correct = coarse_correct = total = 0
    probabilities = []
    effective_centered_logits = []
    entropy_sum = max_probability_sum = 0.0
    mapping = fine_to_coarse.cuda()
    coarse_classes = int(mapping.max().item()) + 1
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        logits = model(images)
        native_correct += logits.argmax(1).eq(targets).sum().item()
        native_probabilities = torch.softmax(logits / temperature, dim=1)
        coarse_probabilities = torch.zeros(
            logits.shape[0], coarse_classes, dtype=native_probabilities.dtype,
            device=logits.device,
        )
        coarse_probabilities.scatter_add_(
            1, mapping.unsqueeze(0).expand(logits.shape[0], -1),
            native_probabilities,
        )
        coarse_targets = mapping[targets]
        coarse_correct += coarse_probabilities.argmax(1).eq(coarse_targets).sum().item()
        entropy = -(
            coarse_probabilities
            * coarse_probabilities.clamp_min(1e-30).log()
        ).sum(1)
        equivalent_logits = temperature * coarse_probabilities.clamp_min(1e-30).log()
        equivalent_logits = equivalent_logits - equivalent_logits.mean(1, keepdim=True)
        probabilities.append(coarse_probabilities.cpu())
        effective_centered_logits.append(equivalent_logits.cpu())
        entropy_sum += entropy.sum().item()
        max_probability_sum += coarse_probabilities.max(1).values.sum().item()
        total += images.shape[0]
    q = torch.cat(probabilities, 0)
    centered_logits = torch.cat(effective_centered_logits, 0)
    return {
        "images": total,
        "native_accuracy": 100.0 * native_correct / total,
        "collapsed_coarse_accuracy": 100.0 * coarse_correct / total,
        "centered_marginal_equivalent_logit_sd": float(
            centered_logits.double().std(unbiased=False)
        ),
        "marginal_label_entropy_T20": entropy_sum / total,
        "marginal_effective_class_count": math.exp(entropy_sum / total),
        "marginal_mean_max_probability": max_probability_sum / total,
        "centered_covariance_participation_rank": (
            covariance_participation_rank(q)
        ),
    }


def main():
    parser = argparse.ArgumentParser("Train per-epoch ImageNette Teacher trajectory")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=20.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    if (output / ".training_complete.json").is_file():
        marker = json.loads((output / ".training_complete.json").read_text())
        if int(marker.get("epochs", -1)) == args.epochs:
            print(f"Complete trajectory already exists: {output}", flush=True)
            return
    existing = list(checkpoints.glob("epoch_*.pth"))
    if existing:
        raise RuntimeError(
            f"partial trajectory exists ({len(existing)} checkpoints): {output}; "
            "archive it before a clean deterministic restart"
        )

    hierarchy = json.loads(Path(args.hierarchy).read_text(encoding="utf-8"))
    mapping_dict = hierarchy["fine_to_coarse"]
    fine_to_coarse = torch.tensor(
        [int(mapping_dict[str(index)]) for index in range(args.classes)],
        dtype=torch.long,
    )
    if fine_to_coarse.numel() != args.classes or int(fine_to_coarse.max()) != 9:
        raise ValueError("hierarchy must map all native heads to ten ImageNette parents")

    train_loader, train_clean_loader, val_loader = make_loaders(
        Path(args.data_dir), args.batch_size, args.workers, args.seed, args.classes
    )
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, args.classes)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    history = []
    definitions = {
        "train_acc": "clean train native K*C accuracy (%)",
        "val_acc": "validation/test accuracy after marginalizing K*C probabilities to K=10 (%)",
        "sd_z": (
            "population SD over clean-train, per-sample centered 10-way equivalent "
            "marginal logits z=T*log(q_parent), T=20"
        ),
        "marg_label_entropy_T20": "mean entropy of clean-train marginalized q_parent at T=20",
        "participation_rank": (
            "covariance participation rank of centered clean-train marginalized "
            "probability matrix at T=20"
        ),
        "lr": "optimizer learning rate used during this epoch",
    }
    metadata = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "hierarchy": str(Path(args.hierarchy).resolve()),
        "classes": args.classes,
        "coarse_classes": 10,
        "seed": args.seed,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD(lr=0.01,momentum=0.9,weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR(T_max=100)",
        "architecture": "torchvision.models.resnet18(weights=None)",
        "definitions": definitions,
    }
    atomic_json_dump(metadata, output / "metadata.json")

    for epoch in range(args.epochs):
        lr_used = optimizer.param_groups[0]["lr"]
        model.train()
        correct = total = 0
        loss_sum = 0.0
        train_started = time.time()
        for images, targets in train_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            correct += logits.argmax(1).eq(targets).sum().item()
            total += images.shape[0]
        train_seconds = time.time() - train_started

        evaluation_started = time.time()
        train_metrics = evaluate_geometry(
            model, train_clean_loader, fine_to_coarse, args.temperature
        )
        val_metrics = evaluate_geometry(
            model, val_loader, fine_to_coarse, args.temperature
        )
        evaluation_seconds = time.time() - evaluation_started

        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        checkpoint = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        checkpoint_path = checkpoints / f"epoch_{epoch:03d}.pth"
        atomic_torch_save(checkpoint, checkpoint_path)

        record = {
            "epoch": epoch,
            "checkpoint": checkpoint_path.name,
            "lr": lr_used,
            "lr_after_scheduler_step": lr_after,
            "train_augmented_accuracy": 100.0 * correct / total,
            "train_augmented_loss": loss_sum / total,
            "train_acc": train_metrics["native_accuracy"],
            "val_acc": val_metrics["collapsed_coarse_accuracy"],
            "sd_z": train_metrics["centered_marginal_equivalent_logit_sd"],
            "marg_label_entropy_T20": train_metrics["marginal_label_entropy_T20"],
            "participation_rank": train_metrics[
                "centered_covariance_participation_rank"
            ],
            "train_native_accuracy": train_metrics["native_accuracy"],
            "train_coarse_accuracy": train_metrics["collapsed_coarse_accuracy"],
            "val_native_accuracy": val_metrics["native_accuracy"],
            "val_coarse_accuracy": val_metrics["collapsed_coarse_accuracy"],
            "train_sd_z": train_metrics["centered_marginal_equivalent_logit_sd"],
            "val_sd_z": val_metrics["centered_marginal_equivalent_logit_sd"],
            "train_marg_entropy_T20": train_metrics["marginal_label_entropy_T20"],
            "val_marg_entropy_T20": val_metrics["marginal_label_entropy_T20"],
            "train_participation_rank": train_metrics[
                "centered_covariance_participation_rank"
            ],
            "val_participation_rank": val_metrics[
                "centered_covariance_participation_rank"
            ],
            "train": train_metrics,
            "val": val_metrics,
            "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
        history.append(record)
        atomic_json_dump(history, output / "metrics.json")
        write_csv(history, output / "metrics.csv")
        print(json.dumps(record), flush=True)

    final_checkpoint = output / "ResNet18.pth"
    atomic_torch_save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        final_checkpoint,
    )
    atomic_json_dump(
        {
            "epochs": args.epochs,
            "classes": args.classes,
            "seed": args.seed,
            "checkpoints": args.epochs,
            "final_checkpoint": final_checkpoint.name,
            "metrics": "metrics.json",
        },
        output / ".training_complete.json",
    )
    print(f"Complete trajectory: {output}", flush=True)


if __name__ == "__main__":
    main()
