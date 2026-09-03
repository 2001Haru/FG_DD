import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from audit_imagenette_consumed_fkd_labels import analyze_root, summarize_roots


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)


def load_resnet18(checkpoint, classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def audit_one_source(cluster_model, c1_model, source, mapping, workers, device):
    transform = transforms.Compose([
        # Recovery writes native 224x224 images. Audit the exact optimized pixels;
        # no random or validation-time crop is introduced here.
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(str(source), transform=transform)
    expected_classes = [f"new{index:03d}" for index in range(100)]
    if dataset.classes != expected_classes:
        raise RuntimeError(f"native recovery class directories are invalid: {source}")
    if len(dataset) != 100:
        raise RuntimeError(f"expected 100 native recovery images: {source}")
    loader = DataLoader(
        dataset, batch_size=100, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    mapping = mapping.to(device)
    counts = torch.zeros(10, dtype=torch.long)
    cluster_coarse_correct_by_parent = torch.zeros(10, dtype=torch.long)
    c1_correct_by_parent = torch.zeros(10, dtype=torch.long)
    cluster_confusion = torch.zeros(10, 10, dtype=torch.long)
    c1_confusion = torch.zeros(10, 10, dtype=torch.long)
    totals = {
        "images": 0,
        "cluster_native_correct": 0,
        "cluster_coarse_correct": 0,
        "c1_coarse_correct": 0,
        "cluster_c1_coarse_agree": 0,
        "both_coarse_correct": 0,
        "cluster_correct_c1_wrong": 0,
        "c1_correct_cluster_wrong": 0,
        "cluster_native_target_probability": 0.0,
        "cluster_native_target_nll": 0.0,
        "cluster_coarse_target_probability": 0.0,
        "c1_target_probability_T1": 0.0,
        "c1_target_probability_T20": 0.0,
        "cluster_native_entropy_T1": 0.0,
        "cluster_coarse_entropy_T1": 0.0,
        "c1_entropy_T1": 0.0,
        "c1_entropy_T20": 0.0,
    }
    for images, native_targets in loader:
        images = images.to(device, non_blocking=True)
        native_targets = native_targets.to(device, non_blocking=True)
        parent_targets = mapping[native_targets]

        cluster_logits = cluster_model(images)
        cluster_native_probabilities = torch.softmax(cluster_logits, dim=1)
        cluster_coarse_probabilities = torch.zeros(
            images.shape[0], 10, dtype=cluster_native_probabilities.dtype,
            device=device,
        )
        cluster_coarse_probabilities.scatter_add_(
            1, mapping.unsqueeze(0).expand(images.shape[0], -1),
            cluster_native_probabilities,
        )
        c1_logits = c1_model(images)
        c1_probabilities_T1 = torch.softmax(c1_logits, dim=1)
        c1_probabilities_T20 = torch.softmax(c1_logits / 20.0, dim=1)

        native_predictions = cluster_native_probabilities.argmax(1)
        cluster_coarse_predictions = cluster_coarse_probabilities.argmax(1)
        c1_predictions = c1_probabilities_T1.argmax(1)
        cluster_matches = cluster_coarse_predictions.eq(parent_targets)
        c1_matches = c1_predictions.eq(parent_targets)
        size = images.shape[0]
        totals["images"] += size
        totals["cluster_native_correct"] += native_predictions.eq(native_targets).sum().item()
        totals["cluster_coarse_correct"] += cluster_matches.sum().item()
        totals["c1_coarse_correct"] += c1_matches.sum().item()
        totals["cluster_c1_coarse_agree"] += cluster_coarse_predictions.eq(c1_predictions).sum().item()
        totals["both_coarse_correct"] += (cluster_matches & c1_matches).sum().item()
        totals["cluster_correct_c1_wrong"] += (cluster_matches & ~c1_matches).sum().item()
        totals["c1_correct_cluster_wrong"] += (c1_matches & ~cluster_matches).sum().item()
        totals["cluster_native_target_probability"] += cluster_native_probabilities.gather(
            1, native_targets[:, None]
        ).sum().item()
        totals["cluster_native_target_nll"] += -cluster_native_probabilities.gather(
            1, native_targets[:, None]
        ).clamp_min(1e-12).log().sum().item()
        totals["cluster_coarse_target_probability"] += cluster_coarse_probabilities.gather(
            1, parent_targets[:, None]
        ).sum().item()
        totals["c1_target_probability_T1"] += c1_probabilities_T1.gather(
            1, parent_targets[:, None]
        ).sum().item()
        totals["c1_target_probability_T20"] += c1_probabilities_T20.gather(
            1, parent_targets[:, None]
        ).sum().item()
        totals["cluster_native_entropy_T1"] += entropy(cluster_native_probabilities).sum().item()
        totals["cluster_coarse_entropy_T1"] += entropy(cluster_coarse_probabilities).sum().item()
        totals["c1_entropy_T1"] += entropy(c1_probabilities_T1).sum().item()
        totals["c1_entropy_T20"] += entropy(c1_probabilities_T20).sum().item()

        cpu_targets = parent_targets.cpu()
        cpu_cluster = cluster_coarse_predictions.cpu()
        cpu_c1 = c1_predictions.cpu()
        counts.scatter_add_(0, cpu_targets, torch.ones_like(cpu_targets))
        cluster_coarse_correct_by_parent.scatter_add_(
            0, cpu_targets, cluster_matches.cpu().long()
        )
        c1_correct_by_parent.scatter_add_(0, cpu_targets, c1_matches.cpu().long())
        for target, prediction in zip(cpu_targets.tolist(), cpu_cluster.tolist()):
            cluster_confusion[target, prediction] += 1
        for target, prediction in zip(cpu_targets.tolist(), cpu_c1.tolist()):
            c1_confusion[target, prediction] += 1

    n = totals["images"]
    result = {
        "images": n,
        "audit_view": "exact saved 224x224 pixels; ToTensor+ImageNet normalization only",
        "cluster_teacher_native100_top1": 100.0 * totals["cluster_native_correct"] / n,
        "cluster_teacher_collapsed_coarse10_top1": 100.0 * totals["cluster_coarse_correct"] / n,
        "c1_teacher_coarse10_top1": 100.0 * totals["c1_coarse_correct"] / n,
        "cluster_c1_coarse_prediction_agreement": totals["cluster_c1_coarse_agree"] / n,
        "both_coarse_correct_fraction": totals["both_coarse_correct"] / n,
        "cluster_correct_c1_wrong_fraction": totals["cluster_correct_c1_wrong"] / n,
        "c1_correct_cluster_wrong_fraction": totals["c1_correct_cluster_wrong"] / n,
    }
    for field in (
        "cluster_native_target_probability", "cluster_coarse_target_probability",
        "cluster_native_target_nll",
        "c1_target_probability_T1", "c1_target_probability_T20",
        "cluster_native_entropy_T1", "cluster_coarse_entropy_T1",
        "c1_entropy_T1", "c1_entropy_T20",
    ):
        result[f"mean_{field}"] = totals[field] / n
    result["per_parent"] = [
        {
            "parent": parent,
            "images": int(counts[parent]),
            "cluster_coarse_top1": 100.0 * int(cluster_coarse_correct_by_parent[parent])
            / max(int(counts[parent]), 1),
            "c1_coarse_top1": 100.0 * int(c1_correct_by_parent[parent])
            / max(int(counts[parent]), 1),
        }
        for parent in range(10)
    ]
    result["cluster_coarse_confusion_target_rows"] = cluster_confusion.tolist()
    result["c1_coarse_confusion_target_rows"] = c1_confusion.tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-seed", type=int, required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--cluster-checkpoint", required=True)
    parser.add_argument("--c1-checkpoint", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fkd-epoch-stride", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hierarchy = json.loads(Path(args.hierarchy).read_text(encoding="utf-8"))
    mapping = torch.tensor([
        int(hierarchy["fine_to_coarse"][str(index)]) for index in range(100)
    ], dtype=torch.long)
    if mapping.tolist() != [index // 10 for index in range(100)]:
        raise RuntimeError("expected parent-major C10 hierarchy")
    device = torch.device("cuda")
    cluster_model = load_resnet18(args.cluster_checkpoint, 100, device)
    c1_model = load_resnet18(args.c1_checkpoint, 10, device)
    root = Path(args.experiment_root) / f"tseed{args.teacher_seed}"

    image_rows = []
    fkd_rows = []
    for recovery in args.recovery_seeds:
        raw_source = (
            root / "native_synthetic" / f"cluster_c10_native100_rseed{recovery}"
        )
        image_metrics = audit_one_source(
            cluster_model, c1_model, raw_source, mapping, args.workers, device
        )
        image_metrics.update({
            "teacher_seed": args.teacher_seed,
            "recovery_seed": recovery,
            "synthetic_root": str(raw_source),
        })
        image_rows.append(image_metrics)

        collapsed = (
            root / "coarse_sources" / f"cluster_c10_native100_rseed{recovery}"
        )
        fkd = root / "fkd" / f"c1_soft_rseed{recovery}_bs10_ipc10"
        fkd_rows.append(analyze_root((
            "c1_soft", 1, args.teacher_seed, recovery,
            str(collapsed), str(fkd), 300, args.fkd_epoch_stride, 20.0, 42,
        )))

    result = {
        "audit_schema_version": 1,
        "definition": {
            "image_audit": "exact saved native-recovery pixels, no crop or flip",
            "cluster_coarse": "sum native100 T=1 softmax probabilities by DINO C10 parent",
            "c1_T1": "C1 hard semantic prediction diagnostic",
            "c1_T20": "distribution shape used by the downstream FKD protocol",
            "fkd_audit": (
                "student-consumed C1 CutMix labels sampled every configured epoch stride; "
                "sampler order and realized CutMix constituents reconstructed exactly"
            ),
        },
        "teacher_seed": args.teacher_seed,
        "recovery_seeds": args.recovery_seeds,
        "image_metrics": image_rows,
        "consumed_fkd_metrics": fkd_rows,
        "consumed_fkd_summary": summarize_roots(
            fkd_rows, [args.teacher_seed], args.recovery_seeds
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
