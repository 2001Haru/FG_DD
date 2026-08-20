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


def evaluate(model, data_root, split, workers, fine_to_coarse, coarse_to_fine):
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
    native_correct = coarse_correct = total = 0
    within_parent_entropy_sum = 0.0
    with torch.no_grad():
        for images, native_targets in loader:
            images = images.cuda(non_blocking=True)
            native_targets = native_targets.cuda(non_blocking=True)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            coarse_probabilities = torch.zeros(
                images.shape[0], 20, device=images.device, dtype=probabilities.dtype
            )
            coarse_probabilities.scatter_add_(
                1, fine_to_coarse.unsqueeze(0).expand(images.shape[0], -1), probabilities
            )
            coarse_targets = fine_to_coarse[native_targets]
            native_correct += logits.argmax(dim=1).eq(native_targets).sum().item()
            coarse_correct += coarse_probabilities.argmax(dim=1).eq(coarse_targets).sum().item()

            parent_groups = probabilities.gather(1, coarse_to_fine[coarse_targets])
            parent_groups = parent_groups / parent_groups.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(parent_groups * parent_groups.clamp_min(1e-12).log()).sum(dim=1)
            within_parent_entropy_sum += entropy.sum().item()
            total += images.shape[0]
    return {
        "images": total,
        "native100_top1": 100.0 * native_correct / total,
        "collapsed_coarse20_top1": 100.0 * coarse_correct / total,
        "native_top1_to_collapsed_coarse_top1_ratio": native_correct / max(coarse_correct, 1),
        "within_parent_group_entropy": within_parent_entropy_sum / total,
        "max_random_group_entropy": float(torch.tensor(5.0).log()),
    }


def main():
    parser = argparse.ArgumentParser("Audit native100 and hierarchy-collapsed coarse20 Teacher accuracy")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    model = ResNet18(100)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    hierarchy = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    fine_to_coarse = torch.tensor(
        [int(hierarchy["fine_to_coarse"][str(index)]) for index in range(100)],
        dtype=torch.long, device="cuda",
    )
    coarse_to_fine = torch.tensor(
        [hierarchy["coarse_to_fine"][str(index)] for index in range(20)],
        dtype=torch.long, device="cuda",
    )
    result = {
        "interpretation": (
            "native100_top1 uses the dataset's 100-way labels; collapsed_coarse20_top1 sums "
            "probabilities according to the supplied hierarchy before taking argmax."
        ),
        "mapping": str(Path(args.mapping).resolve()),
        "train": evaluate(model, args.data_dir, "train", args.workers,
                          fine_to_coarse, coarse_to_fine),
        "test": evaluate(model, args.data_dir, "test", args.workers,
                         fine_to_coarse, coarse_to_fine),
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
