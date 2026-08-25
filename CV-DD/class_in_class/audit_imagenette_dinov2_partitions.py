import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser("Audit DINOv2 clustered ImageNette partitions")
    parser.add_argument("--master-root", required=True)
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--cluster-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.master_root)
    cache_path = root / "features" / "dinov2_base_official_imagenette.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    train = cache["splits"]["train"]
    test = cache["splits"]["test"]
    train_path_to_index = {path: index for index, path in enumerate(train["relative_paths"])}
    test_path_to_index = {path: index for index, path in enumerate(test["relative_paths"])}
    device = torch.device(args.device)
    rows = []

    for c in args.c_values:
        partition = root / "data" / f"dinov2_cluster_c{c}_seed{args.cluster_seed}"
        hierarchy = json.loads((partition / "hierarchy.json").read_text(encoding="utf-8"))
        assignments = json.loads((partition / "cluster_assignments.json").read_text(encoding="utf-8"))
        nclasses = 10 * c
        mapping = hierarchy["fine_to_coarse"]
        if hierarchy["kind"] != "imagenette_balanced_dinov2_clusters":
            raise ValueError(f"C={c}: wrong partition kind")
        if len(mapping) != nclasses or any(int(mapping[str(i)]) != i // c for i in range(nclasses)):
            raise ValueError(f"C={c}: invalid fine_to_coarse mapping")
        if set(assignments["train"]) != set(train_path_to_index):
            raise ValueError(f"C={c}: train assignment path set mismatch")
        if set(assignments["val"]) != set(test_path_to_index):
            raise ValueError(f"C={c}: test assignment path set mismatch")

        parent_rows = []
        all_test_matches = 0
        all_test = 0
        for coarse in range(10):
            train_indices = torch.tensor([
                train_path_to_index[path] for path, pseudo in assignments["train"].items()
                if int(pseudo) // c == coarse
            ], dtype=torch.long)
            train_local = torch.tensor([
                int(pseudo) % c for path, pseudo in assignments["train"].items()
                if int(pseudo) // c == coarse
            ], dtype=torch.long)
            if not torch.all(train["targets"][train_indices].eq(coarse)):
                raise ValueError(f"C={c} coarse={coarse}: source coarse mismatch in train")
            counts = torch.bincount(train_local, minlength=c)
            if int(counts.max() - counts.min()) > 1 or int(counts.min()) < 1:
                raise ValueError(f"C={c} coarse={coarse}: train clusters are not balanced/nonempty")

            features = F.normalize(train["features"][train_indices].float(), dim=1)
            centers = torch.zeros(c, features.shape[1])
            centers.index_add_(0, train_local, features)
            centers = F.normalize(centers / counts.unsqueeze(1), dim=1)
            assigned_similarity = (features * centers[train_local]).sum(1)

            test_items = [
                (test_path_to_index[path], int(pseudo) % c)
                for path, pseudo in assignments["val"].items()
                if int(pseudo) // c == coarse
            ]
            test_indices = torch.tensor([item[0] for item in test_items], dtype=torch.long)
            test_local = torch.tensor([item[1] for item in test_items], dtype=torch.long)
            if not torch.all(test["targets"][test_indices].eq(coarse)):
                raise ValueError(f"C={c} coarse={coarse}: source coarse mismatch in test")
            test_features = F.normalize(test["features"][test_indices].float(), dim=1).to(device)
            predicted = (test_features @ centers.to(device).T).argmax(1).cpu()
            matches = int(predicted.eq(test_local).sum())
            all_test_matches += matches
            all_test += test_local.numel()
            parent_rows.append({
                "coarse_id": coarse,
                "train_images": int(train_indices.numel()),
                "test_images": int(test_indices.numel()),
                "minimum_train_cluster_size": int(counts.min()),
                "maximum_train_cluster_size": int(counts.max()),
                "mean_assigned_train_cosine": float(assigned_similarity.mean()),
                "test_nearest_centroid_assignment_accuracy": matches / test_local.numel(),
            })
        rows.append({
            "C": c,
            "heads": nclasses,
            "train_images": len(assignments["train"]),
            "test_images": len(assignments["val"]),
            "test_nearest_centroid_assignment_accuracy": all_test_matches / all_test,
            "minimum_train_cluster_size": min(row["minimum_train_cluster_size"] for row in parent_rows),
            "maximum_train_cluster_size": max(row["maximum_train_cluster_size"] for row in parent_rows),
            "mean_assigned_train_cosine": sum(
                row["mean_assigned_train_cosine"] * row["train_images"] for row in parent_rows
            ) / sum(row["train_images"] for row in parent_rows),
            "mean_initial_unconstrained_kmeans_cosine": sum(
                hierarchy["cluster_stats"][str(index)]["initial_spherical_kmeans_score"]
                * hierarchy["cluster_stats"][str(index)]["train_images"]
                for index in range(10)
            ) / hierarchy["source_train_images"],
            "mean_recorded_balanced_assignment_cosine": sum(
                hierarchy["cluster_stats"][str(index)]["balanced_assignment_score"]
                * hierarchy["cluster_stats"][str(index)]["train_images"]
                for index in range(10)
            ) / hierarchy["source_train_images"],
            "parents": parent_rows,
        })
        rows[-1]["balance_cosine_penalty"] = (
            rows[-1]["mean_initial_unconstrained_kmeans_cosine"]
            - rows[-1]["mean_recorded_balanced_assignment_cosine"]
        )

    result = {
        "definition": (
            "Recompute train centroids from materialized assignments; verify exact coarse mapping, "
            "balanced nonempty train clusters, complete source path coverage, and test nearest-centroid labels."
        ),
        "feature_cache": str(cache_path),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
