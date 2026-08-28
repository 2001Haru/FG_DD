import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRAINING_EPOCHS = (8, 16, 32, 64, 100, 150, 200, 250, 300)


def forward_resnet18(model, images):
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    activations = model.layer4(x)
    features = torch.flatten(model.avgpool(activations), 1)
    logits = model.fc(features)
    return logits, activations, features


def grad_cam(score, activations, retain_graph):
    gradients = torch.autograd.grad(
        score.sum(), activations, retain_graph=retain_graph,
        create_graph=False, allow_unused=False,
    )[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    return torch.relu((weights * activations).sum(1))


def cam_distribution(cam):
    flat = cam.double().flatten(1)
    mass = flat.sum(1)
    valid = mass > 1e-15
    probabilities = torch.zeros_like(flat)
    probabilities[valid] = flat[valid] / mass[valid, None]
    return probabilities, valid


def cam_statistics(cam):
    probabilities, valid = cam_distribution(cam)
    height, width = cam.shape[-2:]
    entropy = torch.full(
        (cam.shape[0],), float("nan"), dtype=torch.float64,
        device=cam.device,
    )
    entropy[valid] = -(
        probabilities[valid]
        * probabilities[valid].clamp_min(1e-30).log()
    ).sum(1) / math.log(height * width)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, dtype=torch.float64, device=cam.device),
        torch.linspace(0, 1, width, dtype=torch.float64, device=cam.device),
        indexing="ij",
    )
    centroid_x = torch.full_like(entropy, float("nan"))
    centroid_y = torch.full_like(entropy, float("nan"))
    centroid_x[valid] = (probabilities[valid] * xx.flatten()).sum(1)
    centroid_y[valid] = (probabilities[valid] * yy.flatten()).sum(1)
    return probabilities, valid, entropy, centroid_x, centroid_y


def pairwise_js(distributions, valid_masks):
    # distributions: list of Bx(HW), one list item per selected subhead rank.
    values = []
    pair_validity = []
    for left in range(len(distributions)):
        for right in range(left + 1, len(distributions)):
            p, q = distributions[left], distributions[right]
            valid = valid_masks[left] & valid_masks[right]
            midpoint = 0.5 * (p + q)
            kl_p = (
                p.clamp_min(1e-30)
                * (p.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())
            ).sum(1)
            kl_q = (
                q.clamp_min(1e-30)
                * (q.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())
            ).sum(1)
            js = 0.5 * (kl_p + kl_q) / math.log(2.0)
            js[~valid] = float("nan")
            values.append(js)
            pair_validity.append(valid)
    stacked = torch.stack(values, 1)
    valid_pairs = torch.stack(pair_validity, 1)
    count = valid_pairs.sum(1)
    total = torch.nan_to_num(stacked, nan=0.0).sum(1)
    mean = torch.full(
        (stacked.shape[0],), float("nan"), dtype=torch.float64,
        device=stacked.device,
    )
    nonzero = count > 0
    mean[nonzero] = total[nonzero] / count[nonzero]
    return mean, count.double() / stacked.shape[1]


def effective_ranks(features):
    centered = features.double() - features.double().mean(0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    normalized = eigenvalues / total
    participation = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    entropy_rank = torch.exp(-(
        normalized * normalized.clamp_min(1e-30).log()
    ).sum())
    return float(participation), float(entropy_rank)


def nanmean(values):
    values = values.double()
    valid = torch.isfinite(values)
    return float(values[valid].mean()) if valid.any() else float("nan")


def macro_centroid_variance(x, y, targets, selected):
    values = []
    for parent in range(10):
        mask = selected & targets.eq(parent) & torch.isfinite(x) & torch.isfinite(y)
        if mask.sum() < 2:
            continue
        values.append(
            x[mask].var(unbiased=False) + y[mask].var(unbiased=False)
        )
    return float(torch.stack(values).mean()) if values else float("nan")


def load_model(checkpoint, heads):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, heads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def analyze_checkpoint(model, loader, subclasses, temperature, top_subheads):
    parent_entropies, centroid_xs, centroid_ys = [], [], []
    parents, coarse_corrects, features_all = [], [], []
    marginal_entropies, marginal_logits_centered = [], []
    subhead_js_values, subhead_valid_fractions = [], []
    subhead_entropy_by_rank = (
        [[] for _ in range(min(top_subheads, subclasses))]
        if subclasses > 1 else []
    )
    total = coarse_correct = zero_parent_cam = 0
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        logits, activations, features = forward_resnet18(model, images)
        grouped = logits.view(logits.shape[0], 10, subclasses)
        parent_logits = temperature * torch.logsumexp(grouped / temperature, dim=2)
        parent_probabilities = torch.softmax(parent_logits / temperature, dim=1)
        target_score = parent_logits.gather(1, targets[:, None]).squeeze(1)
        parent_cam = grad_cam(
            target_score, activations, retain_graph=subclasses > 1
        )
        parent_distribution, parent_valid, entropy, cx, cy = cam_statistics(parent_cam)
        predicted = parent_logits.argmax(1)
        correct = predicted.eq(targets)
        coarse_correct += correct.sum().item()
        total += images.shape[0]
        zero_parent_cam += (~parent_valid).sum().item()
        # Detach every stored statistic. A plain `.cpu()` preserves grad_fn
        # and would retain the full ResNet graph for every processed batch.
        parent_entropies.append(entropy.detach().cpu())
        centroid_xs.append(cx.detach().cpu())
        centroid_ys.append(cy.detach().cpu())
        parents.append(targets.cpu())
        coarse_corrects.append(correct.cpu())
        features_all.append(features.detach().cpu())
        marginal_entropies.append((-(
            parent_probabilities
            * parent_probabilities.clamp_min(1e-30).log()
        ).sum(1)).detach().cpu())
        centered = parent_logits - parent_logits.mean(1, keepdim=True)
        marginal_logits_centered.append(centered.detach().cpu())

        if subclasses > 1:
            batch_index = torch.arange(images.shape[0], device=images.device)
            within_parent = grouped[batch_index, targets]
            selected_heads = within_parent.topk(
                min(top_subheads, subclasses), dim=1
            ).indices
            distributions, valid_masks = [], []
            for rank in range(selected_heads.shape[1]):
                score = within_parent.gather(
                    1, selected_heads[:, rank:rank + 1]
                ).squeeze(1)
                cam = grad_cam(
                    score, activations,
                    retain_graph=rank < selected_heads.shape[1] - 1,
                )
                distribution, valid = cam_distribution(cam)
                child_entropy = torch.full(
                    (cam.shape[0],), float("nan"), dtype=torch.float64,
                    device=cam.device,
                )
                child_entropy[valid] = -(
                    distribution[valid]
                    * distribution[valid].clamp_min(1e-30).log()
                ).sum(1) / math.log(cam.shape[-2] * cam.shape[-1])
                subhead_entropy_by_rank[rank].append(
                    child_entropy.detach().cpu()
                )
                distributions.append(distribution)
                valid_masks.append(valid)
            js, valid_fraction = pairwise_js(distributions, valid_masks)
            subhead_js_values.append(js.detach().cpu())
            subhead_valid_fractions.append(valid_fraction.detach().cpu())
            del (
                distributions, valid_masks, js, valid_fraction, selected_heads,
                within_parent, batch_index, score, cam, distribution, valid,
                child_entropy,
            )
        # Explicitly release all graph-owning references before Python starts
        # evaluating the next forward pass. Assignment to the next batch only
        # rebinds names after its RHS has already allocated memory.
        del (
            logits, activations, features, images, grouped, parent_logits,
            parent_probabilities, target_score, parent_cam, parent_valid,
            parent_distribution, entropy, cx, cy, predicted, correct, centered,
            targets,
        )

    entropy = torch.cat(parent_entropies)
    cx, cy = torch.cat(centroid_xs), torch.cat(centroid_ys)
    target = torch.cat(parents)
    correct = torch.cat(coarse_corrects).bool()
    features = torch.cat(features_all)
    marginal_entropy = torch.cat(marginal_entropies)
    centered_logits = torch.cat(marginal_logits_centered)
    participation, entropy_rank = effective_ranks(features)
    result = {
        "images": total,
        "coarse_accuracy": 100.0 * coarse_correct / total,
        "parent_cam": {
            "normalized_spatial_entropy_all": nanmean(entropy),
            "normalized_spatial_entropy_coarse_correct": nanmean(entropy[correct]),
            "within_parent_centroid_variance_all": macro_centroid_variance(
                cx, cy, target, torch.ones_like(correct)
            ),
            "within_parent_centroid_variance_coarse_correct": macro_centroid_variance(
                cx, cy, target, correct
            ),
            "zero_cam_fraction": zero_parent_cam / total,
        },
        "penultimate_features": {
            "dimension": features.shape[1],
            "centered_covariance_participation_rank": participation,
            "centered_covariance_entropy_effective_rank": entropy_rank,
        },
        "marginal_labels_T20": {
            "mean_entropy": float(marginal_entropy.double().mean()),
            "effective_class_count": math.exp(float(marginal_entropy.double().mean())),
            "centered_equivalent_logit_sd": float(
                centered_logits.double().std(unbiased=False)
            ),
        },
    }
    if subclasses > 1:
        js = torch.cat(subhead_js_values)
        valid_fraction = torch.cat(subhead_valid_fractions)
        rank_entropies = [torch.cat(values) for values in subhead_entropy_by_rank]
        stacked_subhead_entropy = torch.stack(rank_entropies, 1)
        result["within_parent_top_subhead_cam"] = {
            "selected_heads": min(top_subheads, subclasses),
            "selection": "top logits within the ground-truth parent for each image",
            "mean_pairwise_js_all": nanmean(js),
            "mean_pairwise_js_coarse_correct": nanmean(js[correct]),
            "mean_valid_pair_fraction": float(valid_fraction.mean()),
            "top1_subhead_normalized_spatial_entropy_all": nanmean(
                rank_entropies[0]
            ),
            "top1_subhead_normalized_spatial_entropy_coarse_correct": nanmean(
                rank_entropies[0][correct]
            ),
            "mean_topk_individual_normalized_spatial_entropy_all": nanmean(
                stacked_subhead_entropy
            ),
            "mean_topk_individual_normalized_spatial_entropy_coarse_correct": nanmean(
                stacked_subhead_entropy[correct]
            ),
            "individual_entropy_by_rank_all": [
                nanmean(values) for values in rank_entropies
            ],
            "individual_entropy_by_rank_coarse_correct": [
                nanmean(values[correct]) for values in rank_entropies
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser("ImageNette Grad-CAM trajectory audit")
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--teacher-seed", type=int, required=True)
    parser.add_argument("--C", type=int, choices=(1, 100), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--top-subheads", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output_dir)
    per_checkpoint = output / "per_checkpoint"
    per_checkpoint.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    if len(dataset) != 3925 or len(dataset.classes) != 10:
        raise ValueError("expected full 3925-image, 10-class ImageNette test set")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=(4 if args.workers > 0 else None),
    )
    model_root = (
        Path(args.trajectory_root) / f"tseed{args.teacher_seed}" / "models"
        / f"c{args.C}_tseed{args.teacher_seed}"
    )
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for training_epoch in TRAINING_EPOCHS:
        output_path = per_checkpoint / f"epoch_{training_epoch:03d}.json"
        if output_path.is_file():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            has_current_schema = (
                args.C == 1
                or "top1_subhead_normalized_spatial_entropy_all"
                in payload.get("within_parent_top_subhead_cam", {})
            )
            if payload.get("images") == 3925 and has_current_schema:
                rows.append(payload)
                continue
        checkpoint_index = training_epoch - 1
        checkpoint = model_root / "checkpoints" / f"epoch_{checkpoint_index:03d}.pth"
        model = load_model(checkpoint, 10 * args.C)
        payload = analyze_checkpoint(
            model, loader, args.C, args.temperature, args.top_subheads
        )
        payload.update({
            "audit_schema_version": 2,
            "teacher_seed": args.teacher_seed,
            "C": args.C,
            "training_epoch": training_epoch,
            "checkpoint_epoch_index": checkpoint_index,
            "checkpoint": str(checkpoint),
            "trajectory_train_accuracy": metrics[checkpoint_index]["train_acc"],
            "trajectory_val_coarse_accuracy": metrics[checkpoint_index]["val_acc"],
            "trajectory_lr": metrics[checkpoint_index]["lr"],
        })
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append(payload)
        del model
        torch.cuda.empty_cache()
        print(json.dumps(payload), flush=True)
    result = {
        "protocol": (
            "Grad-CAM at ResNet18 layer4 for the ground-truth marginalized parent "
            "logit T*logsumexp(child_logits/T), T20; full ImageNette test"
        ),
        "teacher_seed": args.teacher_seed,
        "C": args.C,
        "training_epochs": list(TRAINING_EPOCHS),
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
