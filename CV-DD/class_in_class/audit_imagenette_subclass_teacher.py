import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def evaluate(model, root, split, mapping, groups, workers):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(str(root / split), transform)
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    native_correct = coarse_correct = total = 0; entropy_sum = 0.0
    mapping = mapping.cuda(); groups = groups.cuda()
    for images, native_targets in loader:
        images = images.cuda(non_blocking=True); native_targets = native_targets.cuda(non_blocking=True)
        probabilities = model(images).softmax(1)
        coarse_probabilities = torch.zeros(
            images.shape[0], groups.shape[0], device=images.device
        )
        coarse_probabilities.scatter_add_(
            1, mapping.unsqueeze(0).expand(images.shape[0], -1), probabilities
        )
        coarse_targets = mapping[native_targets]
        native_correct += probabilities.argmax(1).eq(native_targets).sum().item()
        coarse_correct += coarse_probabilities.argmax(1).eq(coarse_targets).sum().item()
        within = probabilities.gather(1, groups[coarse_targets])
        within = within / within.sum(1, keepdim=True).clamp_min(1e-12)
        entropy_sum += (-(within * within.clamp_min(1e-12).log()).sum(1)).sum().item()
        total += images.shape[0]
    return {
        "images": total,
        "native_subclass_top1": 100.0 * native_correct / total,
        "collapsed_coarse10_top1": 100.0 * coarse_correct / total,
        "native_to_collapsed_hit_ratio": native_correct / max(coarse_correct, 1),
        "within_parent_entropy": entropy_sum / total,
    }


def main():
    parser = argparse.ArgumentParser("Audit ImageNette random-subclass Teacher")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    hierarchy = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    classes = int(hierarchy["num_pseudo_classes"])
    subclasses = int(hierarchy["subclasses_per_coarse"])
    mapping = torch.tensor([
        int(hierarchy["fine_to_coarse"][str(index)]) for index in range(classes)
    ], dtype=torch.long)
    groups = torch.tensor([
        hierarchy["coarse_to_fine"][str(index)] for index in range(10)
    ], dtype=torch.long)
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    result = {
        "subclasses_per_coarse": subclasses,
        "num_pseudo_classes": classes,
        "expected_test_native_to_coarse_ratio": 1.0 / subclasses,
        "max_uniform_within_parent_entropy": float(torch.tensor(float(subclasses)).log()),
        "train": evaluate(model, Path(args.data_dir), "train", mapping, groups, args.workers),
        "val": evaluate(model, Path(args.data_dir), "val", mapping, groups, args.workers),
    }
    serialized = json.dumps(result, indent=2)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8"); print(serialized)


if __name__ == "__main__":
    main()
