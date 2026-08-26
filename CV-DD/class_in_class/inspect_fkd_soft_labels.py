import argparse
import csv
import json
import math
from pathlib import Path

import torch


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imagefolder_targets(root):
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    targets, paths = [], []
    for class_id, directory in enumerate(classes):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                targets.append(class_id)
                paths.append(str(path))
    return classes, torch.tensor(targets, dtype=torch.long), paths


def epoch_permutation(size, epoch, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = None
    for _ in range(epoch + 1):
        permutation = torch.randperm(size, generator=generator)
    return permutation


def main():
    parser = argparse.ArgumentParser("Inspect student-consumed FKD soft labels")
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--fkd-root", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    synthetic_root, fkd_root = Path(args.synthetic_root), Path(args.fkd_root)
    classes, targets, image_paths = imagefolder_targets(synthetic_root)
    if len(classes) != 10:
        raise ValueError(f"expected 10 classes, found {len(classes)}")
    permutation = epoch_permutation(len(targets), args.epoch, args.sampler_seed)
    start = args.batch * args.batch_size
    config_path = fkd_root / f"epoch_{args.epoch}" / f"batch_{args.batch}.tar"
    config = torch.load(config_path, map_location="cpu", weights_only=False)
    mix_index, sampled_lam, bbox, saved_logits = config[2], config[3], config[4], config[5]
    saved_logits = saved_logits.float()
    q = torch.softmax(saved_logits / args.temperature, dim=1)
    batch_indices = permutation[start:start + q.shape[0]]
    base_targets = targets[batch_indices]
    paired_targets = base_targets[mix_index.long()]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    realized_base_fraction = 1.0 - ((x2 - x1) * (y2 - y1)) / float(224 * 224)

    rows = []
    for row_id in range(q.shape[0]):
        probabilities = q[row_id]
        order = torch.argsort(probabilities, descending=True)
        base = int(base_targets[row_id])
        paired = int(paired_targets[row_id])
        constituent = float(probabilities[base])
        if paired != base:
            constituent += float(probabilities[paired])
        entropy = float(-(
            probabilities * probabilities.clamp_min(1e-15).log()
        ).sum())
        rows.append({
            "row": row_id,
            "source_image": image_paths[int(batch_indices[row_id])],
            "base_coarse_target": base,
            "paired_coarse_target": paired,
            "mix_partner_row": int(mix_index[row_id]),
            "sampled_mix_lambda": float(sampled_lam),
            "realized_base_area_fraction": realized_base_fraction,
            "bbox": [x1, y1, x2, y2],
            "probabilities": [float(value) for value in probabilities],
            "ranked_classes": [
                {"class_id": int(index), "probability": float(probabilities[index])}
                for index in order
            ],
            "entropy": entropy,
            "effective_class_count": math.exp(entropy),
            "maximum_probability": float(probabilities[order[0]]),
            "top1_margin": float(probabilities[order[0]] - probabilities[order[1]]),
            "base_target_probability": float(probabilities[base]),
            "paired_target_probability": float(probabilities[paired]),
            "constituent_class_mass": constituent,
            "non_constituent_mass": 1.0 - constituent,
            "saved_logits_fp32": [float(value) for value in saved_logits[row_id]],
        })

    result = {
        "definition": "q = softmax(saved marginalized 10-way FKD logits / temperature)",
        "synthetic_root": str(synthetic_root),
        "fkd_root": str(fkd_root),
        "config_path": str(config_path),
        "epoch": args.epoch,
        "batch": args.batch,
        "temperature": args.temperature,
        "class_directories": [path.name for path in classes],
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "row", "base_coarse_target", "paired_coarse_target", "entropy",
            "effective_class_count", "maximum_probability", "top1_margin",
            "base_target_probability", "paired_target_probability",
            "constituent_class_mass", "non_constituent_mass",
            *[f"p_class_{index}" for index in range(10)],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in fieldnames if not key.startswith("p_class_")},
                **{
                    f"p_class_{index}": row["probabilities"][index]
                    for index in range(10)
                },
            })
    print(json.dumps(result, indent=2))
    print(f"Saved: {output} and {csv_path}")


if __name__ == "__main__":
    main()
