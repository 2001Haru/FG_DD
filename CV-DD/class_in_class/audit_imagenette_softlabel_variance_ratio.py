import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EPOCHS = (8, 16, 32, 64, 100, 150, 200, 250, 300)
BASE_TEMPERATURE = {1: 800.0, 100: 200.0}


def load_model(checkpoint, heads):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, heads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def variance_decomposition(vectors, targets):
    vectors = vectors.double()
    targets = targets.long()
    global_mean = vectors.mean(0)
    within_sum = between_sum = 0.0
    class_rows = []
    for class_id in range(10):
        selected = vectors[targets.eq(class_id)]
        class_mean = selected.mean(0)
        within = (selected - class_mean).square().sum().item()
        between = selected.shape[0] * (class_mean - global_mean).square().sum().item()
        within_sum += within
        between_sum += between
        class_rows.append({
            "class_id": class_id,
            "images": selected.shape[0],
            "within_trace_contribution": within / vectors.shape[0],
            "between_trace_contribution": between / vectors.shape[0],
        })
    within_trace = within_sum / vectors.shape[0]
    between_trace = between_sum / vectors.shape[0]
    total_trace = (vectors - global_mean).square().sum().item() / vectors.shape[0]
    return {
        "within_trace": within_trace,
        "between_trace": between_trace,
        "total_trace": total_trace,
        "within_plus_between_error": within_trace + between_trace - total_trace,
        "R_within_over_between": within_trace / max(between_trace, 1e-30),
        "log_R": math.log(max(within_trace / max(between_trace, 1e-30), 1e-30)),
        "within_fraction_of_total": within_trace / max(total_trace, 1e-30),
        "per_class": class_rows,
    }


@torch.inference_mode()
def collect(model, loader, subclasses, temperature):
    probabilities, equivalent_logits, targets_all = [], [], []
    correct = total = 0
    entropy_sum = 0.0
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets_gpu = targets.cuda(non_blocking=True)
        logits = model(images).view(images.shape[0], 10, subclasses)
        parent_logits = temperature * torch.logsumexp(logits / temperature, dim=2)
        q = torch.softmax(parent_logits / temperature, dim=1)
        centered_logits = parent_logits - parent_logits.mean(1, keepdim=True)
        probabilities.append(q.cpu())
        equivalent_logits.append(centered_logits.cpu())
        targets_all.append(targets)
        correct += q.argmax(1).eq(targets_gpu).sum().item()
        total += images.shape[0]
        entropy_sum += (-(q * q.clamp_min(1e-30).log()).sum(1)).sum().item()
    q = torch.cat(probabilities)
    z = torch.cat(equivalent_logits)
    targets = torch.cat(targets_all)
    return {
        "images": total,
        "coarse_accuracy": 100.0 * correct / total,
        "mean_entropy": entropy_sum / total,
        "probability_vectors": variance_decomposition(q, targets),
        "centered_equivalent_logits": variance_decomposition(z, targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--teacher-seed", type=int, required=True)
    parser.add_argument("--C", type=int, choices=(1, 100), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    if len(dataset) != 3925 or len(dataset.classes) != 10:
        raise ValueError("expected full ImageNette test set")
    options = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    if args.workers > 0:
        options["prefetch_factor"] = 4
    loader = DataLoader(dataset, **options)
    model_root = (
        Path(args.trajectory_root) / f"tseed{args.teacher_seed}" / "models"
        / f"c{args.C}_tseed{args.teacher_seed}"
    )
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf-8"))
    final_sd = float(metrics[-1]["sd_z"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in EPOCHS:
        path = output / f"epoch_{epoch:03d}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("audit_schema_version") == 1:
                rows.append(payload); continue
        index = epoch - 1
        checkpoint = model_root / "checkpoints" / f"epoch_{index:03d}.pth"
        model = load_model(checkpoint, 10 * args.C)
        predicted_temperature = (
            BASE_TEMPERATURE[args.C] * float(metrics[index]["sd_z"]) / final_sd
        )
        payload = {
            "audit_schema_version": 1,
            "teacher_seed": args.teacher_seed,
            "C": args.C,
            "training_epoch": epoch,
            "trajectory_train_accuracy": metrics[index]["train_acc"],
            "trajectory_val_accuracy": metrics[index]["val_acc"],
            "sd_z": metrics[index]["sd_z"],
            "temperatures": {
                "T20": collect(model, loader, args.C, 20.0),
                "predicted": collect(model, loader, args.C, predicted_temperature),
            },
            "predicted_temperature": predicted_temperature,
            "checkpoint": str(checkpoint),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append(payload)
        del model
        torch.cuda.empty_cache()
        print(json.dumps(payload), flush=True)
    result = {
        "protocol": (
            "Within/between-class trace ratio R on full ImageNette test "
            "marginalized 10-way soft labels"
        ),
        "teacher_seed": args.teacher_seed,
        "C": args.C,
        "training_epochs": list(EPOCHS),
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
