import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def materialize(source, destination):
    if destination.exists():
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spherical_kmeans(features, clusters, seed, iterations, restarts, device):
    x = F.normalize(features.to(device=device, dtype=torch.float32), dim=1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_centers, best_score = None, float("-inf")
    for restart in range(restarts):
        indices = torch.randperm(x.shape[0], generator=generator, device=device)[:clusters]
        centers = x[indices].clone()
        for _ in range(iterations):
            similarity = x @ centers.T
            labels = similarity.argmax(1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=clusters)
            empty = counts.eq(0)
            if empty.any():
                nearest = similarity.max(1).values
                replacements = nearest.argsort()[: int(empty.sum())]
                sums[empty] = x[replacements]
                counts[empty] = 1
            centers = F.normalize(sums / counts.unsqueeze(1), dim=1)
        score = (x @ centers.T).max(1).values.mean().item()
        if score > best_score:
            best_score, best_centers = score, centers.clone()
    return best_centers, best_score


def balanced_assignment(features, centers, refinement_iterations):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("scipy is required for exact balanced cluster assignment") from error

    x = F.normalize(features.float(), dim=1)
    centers = F.normalize(centers.float().cpu(), dim=1)
    n, clusters = x.shape[0], centers.shape[0]
    base, remainder = divmod(n, clusters)
    labels = None
    for _ in range(refinement_iterations):
        similarities = x @ centers.T
        unconstrained = similarities.argmax(1)
        unconstrained_counts = torch.bincount(unconstrained, minlength=clusters)
        extra_order = torch.argsort(unconstrained_counts, descending=True)
        capacities = torch.full((clusters,), base, dtype=torch.long)
        capacities[extra_order[:remainder]] += 1
        slot_centers = torch.repeat_interleave(torch.arange(clusters), capacities)
        cost = (-similarities[:, slot_centers]).numpy()
        rows, columns = linear_sum_assignment(cost)
        labels = torch.empty(n, dtype=torch.long)
        labels[torch.from_numpy(rows)] = slot_centers[torch.from_numpy(columns)]
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=clusters)
        centers = F.normalize(sums / counts.unsqueeze(1), dim=1)
    return labels, centers, torch.bincount(labels, minlength=clusters)


def nearest_labels(features, centers, batch_size=1024):
    features = F.normalize(features.float(), dim=1)
    centers = F.normalize(centers.float(), dim=1)
    labels = []
    for start in range(0, features.shape[0], batch_size):
        labels.append((features[start:start + batch_size] @ centers.T).argmax(1))
    return torch.cat(labels)


def valid_partition(output, expected):
    hierarchy_path = output / "hierarchy.json"
    assignment_path = output / "cluster_assignments.json"
    if not hierarchy_path.is_file() or not assignment_path.is_file():
        return False, "manifest or assignments missing"
    try:
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"manifest unreadable: {error}"
    for key, value in expected.items():
        if hierarchy.get(key) != value:
            return False, f"{key} mismatch"
    classes = int(hierarchy["num_pseudo_classes"])
    width = int(hierarchy["class_name_width"])
    names = [f"{index:0{width}d}" for index in range(classes)]
    for split in ("train", "val"):
        split_root = output / split
        actual = sorted(path.name for path in split_root.iterdir() if path.is_dir()) \
            if split_root.is_dir() else []
        if actual != names:
            return False, f"{split} directory mismatch"
        counts = hierarchy["split_counts"][split]
        for name, directory in zip(names, (split_root / name for name in names)):
            count = sum(
                path.is_file() and path.suffix.lower() in EXTENSIONS
                for path in directory.iterdir()
            )
            if count != int(counts[name.lstrip("0") or "0"]):
                return False, f"{split}/{name} count mismatch"
    return True, "valid"


def main():
    parser = argparse.ArgumentParser("Prepare balanced DINOv2-clustered ImageNette subclasses")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subclasses", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-validation-split", choices=("val", "test"), default="test")
    parser.add_argument("--cluster-device", default="cuda")
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--kmeans-restarts", type=int, default=3)
    parser.add_argument("--balance-refinements", type=int, default=2)
    parser.add_argument("--repair-invalid-output", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_root).resolve()
    feature_cache = Path(args.feature_cache).resolve()
    output = Path(args.output_dir).resolve()
    if args.subclasses < 2:
        raise ValueError("clustered subclass C must be at least 2")
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    if Path(metadata["data_root"]).resolve() != source:
        raise RuntimeError("DINO feature cache source root mismatch")
    required_splits = {"train", args.source_validation_split}
    if not required_splits.issubset(cache["splits"]):
        raise RuntimeError(f"feature cache lacks splits: {required_splits}")
    coarse_names = list(metadata["class_names"])
    if len(coarse_names) != 10:
        raise RuntimeError(f"feature cache has {len(coarse_names)} classes, expected 10")

    total_classes = 10 * args.subclasses
    width = max(3, len(str(total_classes - 1)))
    minimum_train = min(
        int((cache["splits"]["train"]["targets"] == coarse).sum())
        for coarse in range(10)
    )
    if args.subclasses > minimum_train:
        raise ValueError(f"C={args.subclasses} exceeds minimum parent train size={minimum_train}")
    cache_hash = file_sha256(feature_cache)
    expected = {
        "kind": "imagenette_balanced_dinov2_clusters",
        "source_root": str(source),
        "source_validation_split": args.source_validation_split,
        "feature_cache_sha256": cache_hash,
        "cluster_seed": args.seed,
        "subclasses_per_coarse": args.subclasses,
        "num_pseudo_classes": total_classes,
        "class_name_width": width,
    }
    if output.exists():
        valid, reason = valid_partition(output, expected)
        if valid:
            print(f"Existing DINO cluster partition is valid, reusing: {output}")
            return
        if not args.repair_invalid_output:
            raise RuntimeError(f"invalid existing output ({reason}): {output}")
        if not output.name.startswith("dinov2_cluster_c"):
            raise RuntimeError(f"refusing to archive unsafe output: {output}")
        archive = Path(str(output) + f".invalid_{time.strftime('%Y%m%d_%H%M%S')}")
        if archive.exists():
            raise RuntimeError(f"archive exists: {archive}")
        output.rename(archive)
        print(f"Archived invalid clustered partition: {output} -> {archive}")

    train = cache["splits"]["train"]
    validation = cache["splits"][args.source_validation_split]
    assignments = {"train": {}, "val": {}}
    split_counts = {
        "train": {str(index): 0 for index in range(total_classes)},
        "val": {str(index): 0 for index in range(total_classes)},
    }
    cluster_stats = {}
    started = time.time()
    for coarse_id, coarse_name in enumerate(coarse_names):
        train_indices = torch.where(train["targets"].eq(coarse_id))[0]
        val_indices = torch.where(validation["targets"].eq(coarse_id))[0]
        train_features = train["features"][train_indices]
        seed = args.seed * 1_000_003 + args.subclasses * 10_007 + coarse_id * 101
        centers, initial_score = spherical_kmeans(
            train_features, args.subclasses, seed,
            args.kmeans_iterations, args.kmeans_restarts,
            torch.device(args.cluster_device),
        )
        train_labels, centers, counts = balanced_assignment(
            train_features, centers.cpu(), args.balance_refinements
        )
        val_labels = nearest_labels(validation["features"][val_indices], centers)
        train_score = float(
            (F.normalize(train_features.float(), dim=1) * centers[train_labels]).sum(1).mean()
        )
        cluster_stats[str(coarse_id)] = {
            "coarse_name": coarse_name,
            "train_images": int(train_indices.numel()),
            "validation_images": int(val_indices.numel()),
            "minimum_cluster_size": int(counts.min()),
            "maximum_cluster_size": int(counts.max()),
            "initial_spherical_kmeans_score": initial_score,
            "balanced_assignment_score": train_score,
        }

        for output_split, split_record, indices, local_labels in (
            ("train", train, train_indices, train_labels),
            ("val", validation, val_indices, val_labels),
        ):
            for cache_index, local_label in zip(indices.tolist(), local_labels.tolist()):
                pseudo_id = coarse_id * args.subclasses + local_label
                relative = split_record["relative_paths"][cache_index]
                source_path = source / relative
                destination = output / output_split / f"{pseudo_id:0{width}d}" / source_path.name
                materialize(source_path, destination)
                assignments[output_split][relative] = pseudo_id
                split_counts[output_split][str(pseudo_id)] += 1
        print(
            f"DINO cluster C={args.subclasses} coarse={coarse_id + 1}/10 "
            f"train={train_indices.numel()} val={val_indices.numel()} "
            f"size={int(counts.min())}-{int(counts.max())} score={train_score:.6f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    for split in ("train", "val"):
        for pseudo_id in range(total_classes):
            (output / split / f"{pseudo_id:0{width}d}").mkdir(parents=True, exist_ok=True)
    assignment_path = output / "cluster_assignments.json"
    assignment_path.write_text(json.dumps(assignments, indent=2) + "\n", encoding="utf-8")
    hierarchy = {
        **expected,
        "feature_type": metadata["feature_type"],
        "dinov2_model_dir": metadata["model_dir"],
        "partition_balance": "exact floor/ceil capacity constrained",
        "validation_assignment": "nearest balanced-train centroid; no validation balancing",
        "kmeans_iterations": args.kmeans_iterations,
        "kmeans_restarts": args.kmeans_restarts,
        "balance_refinements": args.balance_refinements,
        "num_coarse_classes": 10,
        "coarse_names": coarse_names,
        "fine_to_coarse": {
            str(index): index // args.subclasses for index in range(total_classes)
        },
        "coarse_to_fine": {
            str(coarse): list(range(coarse * args.subclasses, (coarse + 1) * args.subclasses))
            for coarse in range(10)
        },
        "split_counts": split_counts,
        "source_train_images": sum(split_counts["train"].values()),
        "source_val_images": sum(split_counts["val"].values()),
        "minimum_parent_train_images": minimum_train,
        "assignments_sha256": file_sha256(assignment_path),
        "cluster_stats": cluster_stats,
    }
    (output / "hierarchy.json").write_text(
        json.dumps(hierarchy, indent=2) + "\n", encoding="utf-8"
    )
    valid, reason = valid_partition(output, expected)
    if not valid:
        raise RuntimeError(f"new DINO cluster partition failed validation: {reason}")
    print(
        f"Prepared balanced DINO clusters: C={args.subclasses}, classes={total_classes}, "
        f"train={hierarchy['source_train_images']}, val={hierarchy['source_val_images']}, "
        f"output={output}", flush=True,
    )


if __name__ == "__main__":
    main()
