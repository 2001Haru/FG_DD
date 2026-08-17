import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from utils_squeeze import load_dataset


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, targets).item() * images.shape[0]
            correct += logits.argmax(1).eq(targets).sum().item()
            total += images.shape[0]
    return loss_sum / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser("Plain CUB teacher for the restored SRe2L++ baseline")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.dataset_name = "CUB_imsize224"
    args.mean_norm = [0.4857, 0.4994, 0.4326]
    args.std_norm = [0.2260, 0.2215, 0.2595]
    args.ncls = 200
    args.input_size = 224
    args.use_multi_gpu = False
    args.world_size = 1
    args.base_seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_dataset(0, args)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.ncls)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.9 ** (epoch / 2.0)
    )

    best_acc = -1.0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            correct += logits.argmax(1).eq(targets).sum().item()
            total += images.shape[0]

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.8f} "
            f"train_loss={loss_sum / total:.6f} train_acc={100.0 * correct / total:.2f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.2f}",
            flush=True,
        )
        if val_acc > best_acc:
            torch.save(model.state_dict(), args.output)
            best_acc = val_acc
        scheduler.step()

    print(f"plain ResNet18 teacher finished: best Top1={best_acc:.2f}; {args.output}")


if __name__ == "__main__":
    main()
