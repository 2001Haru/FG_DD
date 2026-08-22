import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "ResNet18" in state:
        state = state["ResNet18"]
    model.load_state_dict(state, strict=True)
    return model.cuda().eval(), state


def calibration_error(confidence, correct, bins=15):
    edges = torch.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            value += selected.float().mean().item() * abs(
                correct[selected].float().mean().item()
                - confidence[selected].mean().item()
            )
    return value


@torch.no_grad()
def evaluate_pair(models_by_name, root, split, workers):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(str(Path(root) / split), transform)
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    storage = {
        name: {"loss": 0.0, "correct1": 0, "correct5": 0,
               "confidence": [], "correct": [], "entropy": [], "margin": [],
               "class_correct": torch.zeros(10, dtype=torch.long),
               "class_total": torch.zeros(10, dtype=torch.long),
               "confusion": torch.zeros(10, 10, dtype=torch.long)}
        for name in models_by_name
    }
    agreements = 0
    probability_l1 = 0.0
    symmetric_kl = 0.0
    total = 0
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets_gpu = targets.cuda(non_blocking=True)
        probabilities = {}
        predictions = {}
        for name, model in models_by_name.items():
            logits = model(images)
            probs = logits.softmax(1)
            prediction = probs.argmax(1)
            probabilities[name] = probs
            predictions[name] = prediction
            current = storage[name]
            current["loss"] += F.cross_entropy(
                logits, targets_gpu, reduction="sum"
            ).item()
            current["correct1"] += prediction.eq(targets_gpu).sum().item()
            current["correct5"] += logits.topk(5, 1).indices.eq(
                targets_gpu.unsqueeze(1)
            ).any(1).sum().item()
            confidence, _ = probs.max(1)
            sorted_probs = probs.topk(2, 1).values
            current["confidence"].append(confidence.cpu())
            current["correct"].append(prediction.eq(targets_gpu).cpu())
            current["entropy"].append(
                (-(probs * probs.clamp_min(1e-12).log()).sum(1)).cpu()
            )
            current["margin"].append((sorted_probs[:, 0] - sorted_probs[:, 1]).cpu())
            prediction_cpu = prediction.cpu()
            current["class_total"].scatter_add_(
                0, targets, torch.ones_like(targets, dtype=torch.long)
            )
            current["class_correct"].scatter_add_(
                0, targets, prediction_cpu.eq(targets).long()
            )
            flat = targets * 10 + prediction_cpu
            current["confusion"] += torch.bincount(flat, minlength=100).reshape(10, 10)

        names = list(models_by_name)
        left, right = probabilities[names[0]], probabilities[names[1]]
        agreements += predictions[names[0]].eq(predictions[names[1]]).sum().item()
        probability_l1 += (left - right).abs().sum(1).sum().item()
        symmetric_kl += 0.5 * (
            (left * (left.clamp_min(1e-12).log() - right.clamp_min(1e-12).log())).sum(1)
            + (right * (right.clamp_min(1e-12).log() - left.clamp_min(1e-12).log())).sum(1)
        ).sum().item()
        total += targets.shape[0]

    metrics = {}
    for name, current in storage.items():
        confidence = torch.cat(current["confidence"])
        correct = torch.cat(current["correct"])
        entropy = torch.cat(current["entropy"])
        margin = torch.cat(current["margin"])
        metrics[name] = {
            "images": total,
            "top1": 100.0 * current["correct1"] / total,
            "top5": 100.0 * current["correct5"] / total,
            "cross_entropy": current["loss"] / total,
            "mean_max_probability": confidence.mean().item(),
            "mean_entropy": entropy.mean().item(),
            "mean_top1_top2_margin": margin.mean().item(),
            "ece_15_bins": calibration_error(confidence, correct),
            "per_class_top1": [
                100.0 * correct_count.item() / max(total_count.item(), 1)
                for correct_count, total_count in zip(
                    current["class_correct"], current["class_total"]
                )
            ],
            "confusion_counts": current["confusion"].tolist(),
        }
    return dataset.classes, metrics, {
        "prediction_agreement": agreements / total,
        "mean_probability_l1": probability_l1 / total,
        "mean_symmetric_kl": symmetric_kl / total,
    }


def state_comparison(left, right):
    keys = sorted(set(left) & set(right))
    numerator = denominator_left = denominator_right = dot = 0.0
    for key in keys:
        if not torch.is_floating_point(left[key]):
            continue
        a = left[key].float().reshape(-1)
        b = right[key].float().reshape(-1)
        numerator += (a - b).square().sum().item()
        denominator_left += a.square().sum().item()
        denominator_right += b.square().sum().item()
        dot += (a * b).sum().item()
    return {
        "global_relative_l2_official_denominator": math.sqrt(
            numerator / max(denominator_left, 1e-30)
        ),
        "global_cosine": dot / math.sqrt(
            max(denominator_left * denominator_right, 1e-30)
        ),
    }


def main():
    parser = argparse.ArgumentParser("Compare official and controlled C1 Teachers")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--controlled-checkpoint", required=True)
    parser.add_argument("--controlled-hierarchy", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hierarchy = json.loads(Path(args.controlled_hierarchy).read_text(encoding="utf-8"))
    expected_classes = hierarchy["coarse_names"]
    official, official_state = load_teacher(args.official_checkpoint)
    controlled, controlled_state = load_teacher(args.controlled_checkpoint)
    models_by_name = {"official": official, "controlled_c1": controlled}
    result = {
        "checkpoints": {
            "official": {
                "path": str(Path(args.official_checkpoint).resolve()),
                "sha256": checkpoint_sha256(args.official_checkpoint),
            },
            "controlled_c1": {
                "path": str(Path(args.controlled_checkpoint).resolve()),
                "sha256": checkpoint_sha256(args.controlled_checkpoint),
            },
        },
        "state_dict_comparison": state_comparison(official_state, controlled_state),
        "splits": {},
    }
    for split in ("train", "test"):
        print(f"Evaluating both Teachers on official {split}", flush=True)
        classes, metrics, comparison = evaluate_pair(
            models_by_name, args.official_root, split, args.workers
        )
        result["splits"][split] = {
            "class_order": classes,
            "controlled_hierarchy_class_order": expected_classes,
            "class_names_exact_match": classes == expected_classes,
            "teachers": metrics,
            "pair": comparison,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
