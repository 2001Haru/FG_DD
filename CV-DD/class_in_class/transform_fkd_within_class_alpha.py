import argparse
import json
import math
import os
from pathlib import Path

import torch


def alpha_tag(alpha):
    return f"{alpha:.3f}".replace(".", "p")


def sorted_tar_paths(root):
    root = Path(root)
    epoch_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("epoch_")),
        key=lambda path: int(path.name.split("_")[1]),
    )
    paths = []
    for epoch_dir in epoch_dirs:
        batches = sorted(
            epoch_dir.glob("batch_*.tar"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        paths.extend(batches)
    return epoch_dirs, paths


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decomposition_from_sufficient_statistics(total_sum, total_square, class_sum, counts):
    images = int(counts.sum())
    global_mean = total_sum / images
    within_sum = total_square.clone()
    between = 0.0
    for class_id in range(class_sum.shape[0]):
        count = int(counts[class_id])
        if count == 0:
            raise ValueError(f"empty argmax class {class_id}")
        mean = class_sum[class_id] / count
        within_sum -= count * mean.square().sum()
        between += count * (mean - global_mean).square().sum().item()
    within = within_sum.item() / images
    between /= images
    return {
        "within_trace": within,
        "between_trace": between,
        "R_within_over_between": within / max(between, 1e-30),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--source-temperature", type=float, default=20.0)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--output-dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    epoch_dirs, paths = sorted_tar_paths(input_root)
    if len(epoch_dirs) != 300 or len(paths) != 3000:
        raise ValueError(
            f"expected 300 epochs/3000 batches, found {len(epoch_dirs)}/{len(paths)}"
        )
    total_sum = torch.zeros(args.classes, dtype=torch.float64)
    class_sum = torch.zeros(args.classes, args.classes, dtype=torch.float64)
    counts = torch.zeros(args.classes, dtype=torch.int64)
    total_square = torch.tensor(0.0, dtype=torch.float64)
    samples = 0
    batch_size = None
    for index, path in enumerate(paths):
        config = torch.load(path, map_location="cpu", weights_only=False)
        logits = config[-1].double()
        if logits.ndim != 2 or logits.shape[1] != args.classes:
            raise ValueError(f"invalid logits {tuple(logits.shape)}: {path}")
        if batch_size is None:
            batch_size = logits.shape[0]
        centered = logits - logits.mean(1, keepdim=True)
        labels = centered.argmax(1)
        total_sum += centered.sum(0)
        total_square += centered.square().sum()
        class_sum.index_add_(0, labels, centered)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.int64))
        samples += logits.shape[0]
        if (index + 1) % 500 == 0:
            print(f"stats {index + 1}/{len(paths)}", flush=True)
    scale = math.sqrt(total_square.item() / (samples * args.classes))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid global scale {scale}")
    normalized_class_means = class_sum / counts[:, None].double() / scale
    normalized_decomposition = decomposition_from_sufficient_statistics(
        total_sum / scale,
        total_square / (scale * scale),
        class_sum / scale,
        counts,
    )
    sigma = scale / args.source_temperature
    output_dtype = torch.float16 if args.output_dtype == "fp16" else torch.float32
    roots = {
        alpha: output_base / f"alpha_{alpha_tag(alpha)}"
        for alpha in sorted(set(args.alphas))
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    softmax_error_sum = softmax_error_max = 0.0
    softmax_values = 0
    for index, path in enumerate(paths):
        config = torch.load(path, map_location="cpu", weights_only=False)
        logits = config[-1].double()
        centered = logits - logits.mean(1, keepdim=True)
        normalized = centered / scale
        labels = centered.argmax(1)
        means = normalized_class_means[labels]
        relative = path.relative_to(input_root)
        for alpha, root in roots.items():
            transformed = sigma * (means + alpha * (normalized - means))
            output_path = root / relative
            if output_path.is_file():
                saved = torch.load(
                    output_path, map_location="cpu", weights_only=False
                )[-1]
            else:
                saved = transformed.to(output_dtype)
                output_config = list(config)
                output_config[-1] = saved
                atomic_torch_save(output_config, output_path)
            if abs(alpha - 1.0) < 1e-12:
                original_q = torch.softmax(logits / args.source_temperature, dim=1)
                replay_q = torch.softmax(saved.double(), dim=1)
                difference = (original_q - replay_q).abs()
                softmax_error_sum += difference.sum().item()
                softmax_error_max = max(softmax_error_max, difference.max().item())
                softmax_values += difference.numel()
        if (index + 1) % 250 == 0:
            print(f"write {index + 1}/{len(paths)}", flush=True)
    alpha_rows = []
    for alpha, root in roots.items():
        output_files = len(list(root.glob("epoch_*/batch_*.tar")))
        if output_files != len(paths):
            raise RuntimeError(
                f"incomplete transformed FKD for alpha={alpha}: "
                f"{output_files}/{len(paths)}"
            )
        before = normalized_decomposition
        after = {
            "within_trace": sigma * sigma * alpha * alpha * before["within_trace"],
            "between_trace": sigma * sigma * before["between_trace"],
            "R_within_over_between": alpha * alpha * before["R_within_over_between"],
        }
        row = {
            "alpha": alpha,
            "tag": alpha_tag(alpha),
            "output_root": str(root),
            "batch_files": output_files,
            "expected_decomposition_after_transform": after,
            "R_ratio_observed_by_algebra": (
                after["R_within_over_between"]
                / before["R_within_over_between"]
            ),
            "R_ratio_expected": alpha * alpha,
        }
        alpha_rows.append(row)
    summary = {
        "audit_schema_version": 1,
        "definition": {
            "class_assignment": "fixed argmax parent class before alpha transform",
            "normalization": "one scalar SD over all stored crops and 10 parent logits",
            "sigma_policy": "sigma=s/source_temperature, so alpha=1 replays original q at student T=1",
            "alpha_operation": "mu_argmax + alpha*(normalized_logit-mu_argmax)",
        },
        "input_root": str(input_root),
        "epochs": len(epoch_dirs),
        "batches": len(paths),
        "label_rows": samples,
        "batch_size": batch_size,
        "source_temperature": args.source_temperature,
        "global_centered_logit_sd": scale,
        "sigma": sigma,
        "argmax_class_counts": counts.tolist(),
        "base_normalized_decomposition": normalized_decomposition,
        "alpha_rows": alpha_rows,
        "alpha1_softmax_replay": {
            "values": softmax_values,
            "mae": softmax_error_sum / max(softmax_values, 1),
            "maximum_absolute_error": softmax_error_max,
        },
    }
    atomic_json_dump(summary, output_base / "alpha_transform_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
