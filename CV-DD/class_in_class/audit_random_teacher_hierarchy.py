import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import ResNet18  # noqa: E402


def evaluate(model, data_root, split, workers):
    dataset = datasets.ImageFolder(
        str(Path(data_root) / split),
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]),
    )
    loader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    pseudo_correct = coarse_correct = total = 0
    within_parent_entropy_sum = 0.0
    with torch.no_grad():
        for images, pseudo_targets in loader:
            images = images.cuda(non_blocking=True)
            pseudo_targets = pseudo_targets.cuda(non_blocking=True)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            coarse_probabilities = probabilities.reshape(-1, 20, 5).sum(dim=2)
            coarse_targets = pseudo_targets.div(5, rounding_mode="floor")
            pseudo_correct += logits.argmax(dim=1).eq(pseudo_targets).sum().item()
            coarse_correct += coarse_probabilities.argmax(dim=1).eq(coarse_targets).sum().item()

            parent_groups = probabilities.reshape(-1, 20, 5)[
                torch.arange(images.shape[0], device=images.device), coarse_targets
            ]
            parent_groups = parent_groups / parent_groups.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(parent_groups * parent_groups.clamp_min(1e-12).log()).sum(dim=1)
            within_parent_entropy_sum += entropy.sum().item()
            total += images.shape[0]
    return {
        "images": total,
        "pseudo100_top1": 100.0 * pseudo_correct / total,
        "collapsed_coarse20_top1": 100.0 * coarse_correct / total,
        "pseudo_top1_to_collapsed_coarse_top1_ratio": pseudo_correct / max(coarse_correct, 1),
        "within_parent_group_entropy": within_parent_entropy_sum / total,
        "max_random_group_entropy": float(torch.tensor(5.0).log()),
    }


def main():
    parser = argparse.ArgumentParser("Audit what a memorizing random100 Teacher generalizes")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    model = ResNet18(100)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    result = {
        "interpretation": (
            "Pseudo IDs are random only within each parent. Collapsed coarse accuracy measures "
            "generalized semantic information; pseudo100 accuracy measures random-group prediction."
        ),
        "train": evaluate(model, args.data_dir, "train", args.workers),
        "test": evaluate(model, args.data_dir, "test", args.workers),
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
