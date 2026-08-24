import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from imagenette_subclass_dataset import EncodedSubclassFolder


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def evaluate(model, root, split, mapping, groups, workers):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if split == "val":
        dataset = EncodedSubclassFolder(
            root / split, num_classes=mapping.numel(), transform=transform
        )
    else:
        dataset = datasets.ImageFolder(str(root / split), transform)
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    native_correct = coarse_correct = joint_correct = total = 0; entropy_sum = 0.0
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
        native_matches = probabilities.argmax(1).eq(native_targets)
        coarse_matches = coarse_probabilities.argmax(1).eq(coarse_targets)
        native_correct += native_matches.sum().item()
        coarse_correct += coarse_matches.sum().item()
        joint_correct += (native_matches & coarse_matches).sum().item()
        within = probabilities.gather(1, groups[coarse_targets])
        within = within / within.sum(1, keepdim=True).clamp_min(1e-12)
        entropy_sum += (-(within * within.clamp_min(1e-12).log()).sum(1)).sum().item()
        total += images.shape[0]
    return {
        "images": total,
        "native_correct": native_correct,
        "collapsed_coarse_correct": coarse_correct,
        "native_and_coarse_correct": joint_correct,
        "native_subclass_top1": 100.0 * native_correct / total,
        "collapsed_coarse10_top1": 100.0 * coarse_correct / total,
        "native_to_collapsed_hit_ratio": native_correct / max(coarse_correct, 1),
        "conditional_native_given_coarse_correct": joint_correct / max(coarse_correct, 1),
        "within_parent_entropy": entropy_sum / total,
    }


def main():
    parser = argparse.ArgumentParser("Audit ImageNette random-subclass Teacher")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--train-only", action="store_true",
        help="audit the clean training split only; do not construct or evaluate validation data",
    )
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
    def binomial_test(k, n, probability):
        if probability >= 1.0:
            return {"z": 0.0 if k == n else float("-inf"),
                    "one_sided_p_in_observed_direction": 1.0 if k == n else 0.0,
                    "two_sided_p": 1.0 if k == n else 0.0}
        observed = k / n
        z = (observed - probability) / math.sqrt(probability * (1 - probability) / n)
        log_masses = [
            math.lgamma(n + 1) - math.lgamma(index + 1) - math.lgamma(n - index + 1)
            + index * math.log(probability) + (n - index) * math.log1p(-probability)
            for index in range(n + 1)
        ]
        observed_log_mass = log_masses[k]
        two_sided = sum(
            math.exp(value) for value in log_masses if value <= observed_log_mass + 1e-12
        )
        if observed < probability:
            one_sided = sum(math.exp(log_masses[index]) for index in range(k + 1))
            direction = "less"
        else:
            one_sided = sum(math.exp(log_masses[index]) for index in range(k, n + 1))
            direction = "greater"
        return {"z": z, "direction": direction,
                "one_sided_p_in_observed_direction": min(one_sided, 1.0),
                "two_sided_p": min(two_sided, 1.0)}

    train = evaluate(model, Path(args.data_dir), "train", mapping, groups, args.workers)
    result = {
        "audit_schema_version": 2,
        "audit_scope": "train_only" if args.train_only else "train_and_validation",
        "subclasses_per_coarse": subclasses,
        "num_pseudo_classes": classes,
        "max_uniform_within_parent_entropy": float(torch.tensor(float(subclasses)).log()),
        "train": train,
    }
    if not args.train_only:
        val = evaluate(model, Path(args.data_dir), "val", mapping, groups, args.workers)
        val["conditional_ratio_binomial_test"] = binomial_test(
            val["native_and_coarse_correct"], val["collapsed_coarse_correct"],
            1.0 / subclasses,
        )
        result["expected_test_native_to_coarse_ratio"] = 1.0 / subclasses
        result["val"] = val
    serialized = json.dumps(result, indent=2)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8"); print(serialized)


if __name__ == "__main__":
    main()
