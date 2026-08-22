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
def evaluate_pair(models_by_name, root, split, workers, temperature):
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
        name: {"loss": 0.0, "temperature_loss": 0.0,
               "correct1": 0, "correct5": 0,
               "confidence": [], "correct": [], "entropy": [], "margin": [],
               "logits_l2": [], "logits_std": [],
               "temperature_confidence": [], "temperature_entropy": [],
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
            temperature_probs = (logits / temperature).softmax(1)
            prediction = probs.argmax(1)
            probabilities[name] = probs
            predictions[name] = prediction
            current = storage[name]
            current["loss"] += F.cross_entropy(
                logits, targets_gpu, reduction="sum"
            ).item()
            current["temperature_loss"] += F.cross_entropy(
                logits / temperature, targets_gpu, reduction="sum"
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
            current["logits_l2"].append(logits.norm(dim=1).cpu())
            current["logits_std"].append(logits.std(dim=1, unbiased=False).cpu())
            current["temperature_confidence"].append(
                temperature_probs.max(1).values.cpu()
            )
            current["temperature_entropy"].append(
                (-(temperature_probs * temperature_probs.clamp_min(1e-12).log())
                 .sum(1)).cpu()
            )
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
        logits_l2 = torch.cat(current["logits_l2"])
        logits_std = torch.cat(current["logits_std"])
        temperature_confidence = torch.cat(current["temperature_confidence"])
        temperature_entropy = torch.cat(current["temperature_entropy"])
        metrics[name] = {
            "images": total,
            "top1": 100.0 * current["correct1"] / total,
            "top5": 100.0 * current["correct5"] / total,
            "cross_entropy": current["loss"] / total,
            "negative_log_likelihood": current["loss"] / total,
            "mean_max_probability": confidence.mean().item(),
            "mean_entropy": entropy.mean().item(),
            "mean_top1_top2_margin": margin.mean().item(),
            "mean_logits_l2_norm": logits_l2.mean().item(),
            "std_logits_l2_norm": logits_l2.std(unbiased=False).item(),
            "mean_within_sample_logits_std": logits_std.mean().item(),
            f"temperature_{temperature:g}": {
                "negative_log_likelihood": current["temperature_loss"] / total,
                "mean_max_probability": temperature_confidence.mean().item(),
                "mean_entropy": temperature_entropy.mean().item(),
                "mean_entropy_over_log_num_classes": (
                    temperature_entropy.mean().item() / math.log(10)
                ),
            },
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
        "state_dict_keys_exact_match": list(left.keys()) == list(right.keys()),
        "state_dict_shapes_exact_match": (
            {key: tuple(value.shape) for key, value in left.items()}
            == {key: tuple(value.shape) for key, value in right.items()}
        ),
        "global_relative_l2_official_denominator": math.sqrt(
            numerator / max(denominator_left, 1e-30)
        ),
        "global_cosine": dot / math.sqrt(
            max(denominator_left * denominator_right, 1e-30)
        ),
    }


def architecture_signature(model):
    modules = [
        {"name": name, "type": f"{module.__class__.__module__}.{module.__class__.__name__}"}
        for name, module in model.named_modules()
    ]
    parameters = {
        name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    }
    canonical = json.dumps(
        {"modules": modules, "parameters": parameters}, sort_keys=True
    ).encode("utf-8")
    return {
        "root_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "module_count": len(modules),
        "signature_sha256": hashlib.sha256(canonical).hexdigest(),
        "modules": modules,
        "parameter_shapes": parameters,
    }


def relative_l2(left, right):
    left = left.float()
    right = right.float()
    return (left - right).norm().item() / max(left.norm().item(), 1e-30)


def batchnorm_state_comparison(official_state, controlled_state):
    prefixes = sorted(
        key[:-len(".running_mean")]
        for key in official_state
        if key.endswith(".running_mean")
    )
    layers = []
    for prefix in prefixes:
        entry = {"layer": prefix}
        for suffix in ("running_mean", "running_var", "weight", "bias"):
            key = f"{prefix}.{suffix}"
            entry[f"{suffix}_relative_l2_official_denominator"] = relative_l2(
                official_state[key], controlled_state[key]
            )
        tracked_key = f"{prefix}.num_batches_tracked"
        entry["official_num_batches_tracked"] = int(official_state[tracked_key])
        entry["controlled_num_batches_tracked"] = int(controlled_state[tracked_key])
        layers.append(entry)
    return {
        "layers": layers,
        "mean_running_mean_relative_l2": sum(
            layer["running_mean_relative_l2_official_denominator"] for layer in layers
        ) / len(layers),
        "mean_running_var_relative_l2": sum(
            layer["running_var_relative_l2_official_denominator"] for layer in layers
        ) / len(layers),
        "mean_affine_weight_relative_l2": sum(
            layer["weight_relative_l2_official_denominator"] for layer in layers
        ) / len(layers),
        "mean_affine_bias_relative_l2": sum(
            layer["bias_relative_l2_official_denominator"] for layer in layers
        ) / len(layers),
    }


def main():
    parser = argparse.ArgumentParser("Compare official and controlled C1 Teachers")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--controlled-checkpoint", required=True)
    parser.add_argument("--controlled-hierarchy", required=True)
    parser.add_argument("--official-label", default="official")
    parser.add_argument("--controlled-label", default="controlled_c1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hierarchy = json.loads(Path(args.controlled_hierarchy).read_text(encoding="utf-8"))
    expected_classes = hierarchy["coarse_names"]
    official, official_state = load_teacher(args.official_checkpoint)
    controlled, controlled_state = load_teacher(args.controlled_checkpoint)
    if args.official_label == args.controlled_label:
        raise ValueError("Teacher labels must be distinct")
    models_by_name = {
        args.official_label: official,
        args.controlled_label: controlled,
    }
    official_architecture = architecture_signature(official)
    controlled_architecture = architecture_signature(controlled)
    result = {
        "checkpoints": {
            args.official_label: {
                "path": str(Path(args.official_checkpoint).resolve()),
                "sha256": checkpoint_sha256(args.official_checkpoint),
            },
            args.controlled_label: {
                "path": str(Path(args.controlled_checkpoint).resolve()),
                "sha256": checkpoint_sha256(args.controlled_checkpoint),
            },
        },
        "architecture": {
            args.official_label: official_architecture,
            args.controlled_label: controlled_architecture,
            "exact_signature_match": (
                official_architecture["signature_sha256"]
                == controlled_architecture["signature_sha256"]
            ),
            "both_strictly_loaded_into_torchvision_resnet18": True,
        },
        "state_dict_comparison": state_comparison(official_state, controlled_state),
        "batchnorm_state_comparison": batchnorm_state_comparison(
            official_state, controlled_state
        ),
        "splits": {},
    }
    for split in ("train", "test"):
        print(f"Evaluating both Teachers on official {split}", flush=True)
        classes, metrics, comparison = evaluate_pair(
            models_by_name, args.official_root, split, args.workers,
            args.temperature,
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
