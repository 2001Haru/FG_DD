import argparse
import hashlib
import json
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


def make_loaders(root, batch_size, workers, seed, expected_classes, skip_validation):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.ImageFolder(str(root / "train"), train_transform)
    val = None if skip_validation else EncodedSubclassFolder(
        root / "val", num_classes=expected_classes, transform=val_transform
    )
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    if workers > 0:
        options["prefetch_factor"] = 4
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=generator, **options),
        (None if val is None else
         DataLoader(val, batch_size=256, shuffle=False, **options)),
        len(train.classes),
    )


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); correct = total = 0; total_loss = 0.0
    for images, targets in loader:
        images = images.cuda(non_blocking=True); targets = targets.cuda(non_blocking=True)
        output = model(images)
        total_loss += criterion(output, targets).item() * images.shape[0]
        correct += output.argmax(1).eq(targets).sum().item(); total += images.shape[0]
    return correct / total, total_loss / total


def main():
    parser = argparse.ArgumentParser("Train ImageNette random-subclass ResNet18 Teacher")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="train only; used when C exceeds images in a validation parent class",
    )
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data_dir) / "hierarchy.json"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train_loader, val_loader, classes = make_loaders(
        Path(args.data_dir), args.batch_size, args.workers, args.seed, args.classes,
        args.skip_validation,
    )
    if classes != args.classes:
        raise RuntimeError(
            f"ImageFolder classes={classes}, expected {args.classes}; "
            f"data_dir={args.data_dir}; class_names={train_loader.dataset.classes}"
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
    for epoch in range(args.epochs):
        model.train(); correct = total = 0; loss_sum = 0.0; started = time.time()
        for images, targets in train_loader:
            images = images.cuda(non_blocking=True); targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output_logits = model(images); loss = criterion(output_logits, targets)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            correct += output_logits.argmax(1).eq(targets).sum().item()
            total += images.shape[0]
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_augmented_accuracy": correct / total,
            "train_augmented_loss": loss_sum / total,
            "seconds": time.time() - started,
        }
        if (not args.skip_validation) and (epoch % 10 == 0 or epoch == args.epochs - 1):
            val_acc, val_loss = evaluate(model, val_loader, criterion)
            record.update({"val_native_accuracy": val_acc, "val_native_loss": val_loss})
            torch.save(model.state_dict(), output / "ResNet18.pth")
        history.append(record)
        print(json.dumps(record), flush=True)

    torch.save(model.state_dict(), output / "ResNet18.pth")
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output / ".training_complete.json").write_text(
        json.dumps({
            "epochs": args.epochs,
            "classes": args.classes,
            "seed": args.seed,
            "checkpoint": "ResNet18.pth",
            "data_manifest_sha256": manifest_sha256,
            "validation_enabled": not args.skip_validation,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Teacher saved: {output / 'ResNet18.pth'}")


if __name__ == "__main__":
    main()
