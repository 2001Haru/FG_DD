import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_SPECS = {
    "c1": {"C": 1, "training_epoch": 16, "temperature": 200.0},
    "c100": {"C": 100, "training_epoch": 100, "temperature": 46.0},
}
PAIR_SPECS = (
    ("c1_s43", "c1_s44", "within_c1"),
    ("c100_s43", "c100_s44", "within_c100"),
    ("c1_s43", "c100_s43", "cross_family_same_seed"),
    ("c1_s44", "c100_s44", "cross_family_same_seed"),
    ("c1_s43", "c100_s44", "cross_family_cross_seed"),
    ("c1_s44", "c100_s43", "cross_family_cross_seed"),
)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_model(checkpoint, heads, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, heads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def marginalized_probabilities(logits, subclasses, temperature):
    logits = logits.double().view(logits.shape[0], 10, subclasses)
    parent_logits = temperature * torch.logsumexp(logits / temperature, dim=2)
    return torch.softmax(parent_logits / temperature, dim=1).float()


@torch.inference_mode()
def collect(args):
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    if len(dataset) != 3925 or len(dataset.classes) != 10:
        raise ValueError(
            f"expected full 3925-image/10-class ImageNette test set: {args.test_root}"
        )
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_options)
    trajectory = Path(args.trajectory_root) / f"tseed{args.teacher_seed}" / "models"
    specs = {
        "c1": {
            "C": 1,
            "training_epoch": args.c1_training_epoch,
            "temperature": args.c1_temperature,
        },
        "c100": {
            "C": 100,
            "training_epoch": args.c100_training_epoch,
            "temperature": args.c100_temperature,
        },
    }
    loaded = {}
    spec_payload = {}
    for family, spec in specs.items():
        c = spec["C"]
        checkpoint = (
            trajectory / f"c{c}_tseed{args.teacher_seed}" / "checkpoints"
            / f"epoch_{spec['training_epoch'] - 1:03d}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        loaded[family] = load_model(checkpoint, 10 * c, device)
        spec_payload[family] = {**spec, "checkpoint": str(checkpoint)}

    probabilities = {family: [] for family in specs}
    targets_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        for family, spec in specs.items():
            logits = loaded[family](images)
            probabilities[family].append(
                marginalized_probabilities(
                    logits, spec["C"], spec["temperature"]
                ).cpu()
            )
        targets_all.append(targets.cpu())
    payload = {
        "audit_schema_version": 1,
        "definition": (
            "Deterministic center-crop soft labels on the same complete ImageNette "
            "test images; checkpoint/temperature choice is recorded in specs."
        ),
        "selection_label": args.selection_label,
        "teacher_seed": args.teacher_seed,
        "images": len(dataset),
        "class_names": list(dataset.classes),
        "sample_paths": [str(Path(path).relative_to(args.test_root)) for path, _ in dataset.samples],
        "targets": torch.cat(targets_all).long(),
        "probabilities": {
            family: torch.cat(chunks).float()
            for family, chunks in probabilities.items()
        },
        "specs": spec_payload,
    }
    output = Path(args.output)
    atomic_torch_save(payload, output)
    print(json.dumps({
        "output": str(output),
        "teacher_seed": args.teacher_seed,
        "images": len(dataset),
        "specs": spec_payload,
    }, indent=2), flush=True)


def class_means_and_within(vectors, targets):
    vectors = vectors.double()
    means = []
    residual = torch.empty_like(vectors)
    for class_id in range(10):
        mask = targets.eq(class_id)
        mean = vectors[mask].mean(0)
        means.append(mean)
        residual[mask] = vectors[mask] - mean
    return torch.stack(means), residual


def variance_summary(vectors, targets):
    vectors = vectors.double()
    class_means, residual = class_means_and_within(vectors, targets)
    global_mean = vectors.mean(0)
    within = residual.square().sum().item() / vectors.shape[0]
    between = 0.0
    for class_id in range(10):
        count = targets.eq(class_id).sum().item()
        between += count * (class_means[class_id] - global_mean).square().sum().item()
    between /= vectors.shape[0]
    centered = vectors - global_mean
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    participation_rank = (
        eigenvalues.sum().square()
        / eigenvalues.square().sum().clamp_min(1e-30)
    ).item()
    return {
        "within_trace": within,
        "between_trace": between,
        "R_within_over_between": within / max(between, 1e-30),
        "centered_label_participation_rank": participation_rank,
    }


def simplex_tangent_coordinates(vectors):
    """Project K probabilities isometrically onto the (K-1)-D simplex tangent."""
    vectors = vectors.double()
    classes = vectors.shape[1]
    basis = torch.zeros(
        classes, classes - 1, dtype=torch.double, device=vectors.device
    )
    for column in range(classes - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return vectors @ basis


def linear_cka(left, right):
    left = simplex_tangent_coordinates(left)
    right = simplex_tangent_coordinates(right)
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    cross = (left.T @ right).square().sum()
    denominator = torch.sqrt(
        (left.T @ left).square().sum()
        * (right.T @ right).square().sum()
    ).clamp_min(1e-30)
    return (cross / denominator).item()


def cca_spectrum(left, right, relative_tolerance=1e-8):
    left = simplex_tangent_coordinates(left)
    right = simplex_tangent_coordinates(right)
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    ux, sx, _ = torch.linalg.svd(left, full_matrices=False)
    uy, sy, _ = torch.linalg.svd(right, full_matrices=False)
    keep_x = sx > sx.max().clamp_min(1e-30) * relative_tolerance
    keep_y = sy > sy.max().clamp_min(1e-30) * relative_tolerance
    ux = ux[:, keep_x]
    uy = uy[:, keep_y]
    correlations = torch.linalg.svdvals(ux.T @ uy).clamp(0, 1)
    values = correlations.tolist()
    return {
        "rank_left": int(keep_x.sum()),
        "rank_right": int(keep_y.sum()),
        "canonical_correlations": values,
        "shared_dimensions_rho_ge_0p9": sum(value >= 0.9 for value in values),
        "shared_dimensions_rho_ge_0p75": sum(value >= 0.75 for value in values),
        "shared_dimensions_rho_ge_0p5": sum(value >= 0.5 for value in values),
        "sum_squared_canonical_correlations": sum(value * value for value in values),
    }


def covariance_directions(residual):
    residual = simplex_tangent_coordinates(residual)
    residual = residual - residual.mean(0, keepdim=True)
    covariance = residual.T @ residual / max(residual.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    threshold = eigenvalues.max().clamp_min(1e-30) * 1e-10
    rank = int((eigenvalues > threshold).sum())
    return eigenvalues, eigenvectors, rank


def principal_angle_report(left, right):
    eigen_left, vectors_left, rank_left = covariance_directions(left)
    eigen_right, vectors_right, rank_right = covariance_directions(right)
    top_angle = math.degrees(math.acos(float(
        torch.abs(vectors_left[:, 0] @ vectors_right[:, 0]).clamp(0, 1)
    )))
    subspaces = {}
    for requested in (1, 2, 3, 5):
        dimensions = min(requested, rank_left, rank_right)
        singular = torch.linalg.svdvals(
            vectors_left[:, :dimensions].T @ vectors_right[:, :dimensions]
        ).clamp(0, 1)
        angles = torch.rad2deg(torch.acos(singular)).tolist()
        subspaces[f"top_{requested}"] = {
            "dimensions_used": dimensions,
            "principal_angles_degrees": angles,
            "mean_angle_degrees": sum(angles) / len(angles),
            "maximum_angle_degrees": max(angles),
            "mean_squared_cosine": singular.square().mean().item(),
        }
    return {
        "rank_left": rank_left,
        "rank_right": rank_right,
        "top_eigenvector_angle_degrees": top_angle,
        "normalized_eigenvalue_spectrum_left": (
            eigen_left / eigen_left.sum().clamp_min(1e-30)
        ).tolist(),
        "normalized_eigenvalue_spectrum_right": (
            eigen_right / eigen_right.sum().clamp_min(1e-30)
        ).tolist(),
        "subspaces": subspaces,
    }


def cosine_matrix(rows):
    rows = rows.double()
    normalized = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return normalized @ normalized.T


def pearson(left, right):
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    return (
        (left @ right)
        / (left.norm() * right.norm()).clamp_min(1e-30)
    ).item()


def ranks(values):
    values = values.double()
    order = torch.argsort(values)
    result = torch.empty_like(values)
    result[order] = torch.arange(values.numel(), dtype=torch.double)
    return result


def compare_similarity_matrices(left, right):
    indices = torch.triu_indices(10, 10, offset=1)
    left_values = left[indices[0], indices[1]]
    right_values = right[indices[0], indices[1]]
    difference = left_values - right_values
    return {
        "upper_triangle_pearson": pearson(left_values, right_values),
        "upper_triangle_spearman": pearson(ranks(left_values), ranks(right_values)),
        "upper_triangle_mae": difference.abs().mean().item(),
        "upper_triangle_rmse": difference.square().mean().sqrt().item(),
    }


def agreement_metrics(left, right, targets):
    left = left.double()
    right = right.double()
    left_top1 = left.argmax(1)
    right_top1 = right.argmax(1)
    left_top5 = left.topk(5, dim=1).indices
    right_top5 = right.topk(5, dim=1).indices
    left_membership = torch.zeros_like(left, dtype=torch.bool).scatter_(1, left_top5, True)
    right_membership = torch.zeros_like(right, dtype=torch.bool).scatter_(1, right_top5, True)
    overlap = (left_membership & right_membership).sum(1).double()
    row_centered_left = left - left.mean(1, keepdim=True)
    row_centered_right = right - right.mean(1, keepdim=True)
    row_cosine = (
        (row_centered_left * row_centered_right).sum(1)
        / (
            row_centered_left.norm(dim=1)
            * row_centered_right.norm(dim=1)
        ).clamp_min(1e-30)
    )
    mixture = 0.5 * (left + right)
    js = 0.5 * (
        (left * (left.clamp_min(1e-30).log() - mixture.clamp_min(1e-30).log())).sum(1)
        + (right * (right.clamp_min(1e-30).log() - mixture.clamp_min(1e-30).log())).sum(1)
    )
    return {
        "top1_agreement_fraction": left_top1.eq(right_top1).double().mean().item(),
        "left_top1_accuracy": left_top1.eq(targets).double().mean().item(),
        "right_top1_accuracy": right_top1.eq(targets).double().mean().item(),
        "mean_top5_overlap_count": overlap.mean().item(),
        "mean_top5_jaccard": (overlap / (10.0 - overlap)).mean().item(),
        "exact_top5_set_agreement_fraction": overlap.eq(5).double().mean().item(),
        "left_target_in_top5_fraction": left_membership.gather(1, targets[:, None]).double().mean().item(),
        "right_target_in_top5_fraction": right_membership.gather(1, targets[:, None]).double().mean().item(),
        "mean_per_image_centered_cosine": row_cosine.mean().item(),
        "mean_jensen_shannon_divergence": js.mean().item(),
    }


def model_summary(probabilities, targets):
    probabilities = probabilities.double()
    entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log()
    ).sum(1)
    result = {
        "coarse_top1_accuracy": probabilities.argmax(1).eq(targets).double().mean().item(),
        "mean_entropy": entropy.mean().item(),
        "mean_max_probability": probabilities.max(1).values.mean().item(),
        "mean_target_probability": probabilities.gather(1, targets[:, None]).mean().item(),
    }
    result.update(variance_summary(probabilities, targets))
    return result


def pair_report(left_name, right_name, relation, models, targets, class_names):
    left = models[left_name]
    right = models[right_name]
    left_means, left_within = class_means_and_within(left, targets)
    right_means, right_within = class_means_and_within(right, targets)
    global_angles = principal_angle_report(left_within, right_within)
    per_class = []
    for class_id, class_name in enumerate(class_names):
        mask = targets.eq(class_id)
        class_angles = principal_angle_report(
            left_within[mask], right_within[mask]
        )
        per_class.append({
            "class_id": class_id,
            "class_name": class_name,
            **class_angles,
            "within_class_linear_CKA": linear_cka(
                left_within[mask], right_within[mask]
            ),
        })
    raw_left_similarity = cosine_matrix(left_means)
    raw_right_similarity = cosine_matrix(right_means)
    centered_left_means = left_means - left.mean(0, keepdim=True)
    centered_right_means = right_means - right.mean(0, keepdim=True)
    centered_left_similarity = cosine_matrix(
        simplex_tangent_coordinates(centered_left_means)
    )
    centered_right_similarity = cosine_matrix(
        simplex_tangent_coordinates(centered_right_means)
    )
    return {
        "left": left_name,
        "right": right_name,
        "relation": relation,
        "agreement": agreement_metrics(left, right, targets),
        "linear_CKA": {
            "globally_centered_labels": linear_cka(left, right),
            "within_class_centered_labels": linear_cka(left_within, right_within),
            "between_class_prototypes": linear_cka(left_means, right_means),
        },
        "CCA": {
            "globally_centered_labels": cca_spectrum(left, right),
            "within_class_centered_labels": cca_spectrum(left_within, right_within),
            "between_class_prototypes": cca_spectrum(left_means, right_means),
        },
        "within_class_covariance_principal_angles": {
            "pooled_within_class_residuals": global_angles,
            "per_class": per_class,
        },
        "interclass_similarity": {
            "raw_probability_cosine": {
                "left_matrix": raw_left_similarity.tolist(),
                "right_matrix": raw_right_similarity.tolist(),
                "comparison": compare_similarity_matrices(
                    raw_left_similarity, raw_right_similarity
                ),
            },
            "globally_centered_prototype_cosine": {
                "left_matrix": centered_left_similarity.tolist(),
                "right_matrix": centered_right_similarity.tolist(),
                "comparison": compare_similarity_matrices(
                    centered_left_similarity, centered_right_similarity
                ),
            },
        },
    }


def mean_and_sample_sd(values):
    values = torch.tensor(values, dtype=torch.double)
    return {
        "mean": values.mean().item(),
        "sample_sd": values.std(unbiased=True).item() if values.numel() > 1 else 0.0,
        "values": values.tolist(),
    }


def analyze(args):
    input_dir = Path(args.input_dir)
    payloads = {
        seed: torch.load(
            input_dir / f"tseed{seed}.pt", map_location="cpu", weights_only=False
        )
        for seed in (43, 44)
    }
    reference = payloads[43]
    def comparable_specs(payload):
        return {
            family: {
                "C": int(spec["C"]),
                "training_epoch": int(spec["training_epoch"]),
                "temperature": float(spec["temperature"]),
            }
            for family, spec in payload["specs"].items()
        }
    selected_specs = comparable_specs(reference)
    targets = reference["targets"].long()
    class_names = reference["class_names"]
    for seed, payload in payloads.items():
        if payload["images"] != 3925:
            raise ValueError(f"invalid image count for seed {seed}")
        if payload["sample_paths"] != reference["sample_paths"]:
            raise ValueError(f"sample order differs for seed {seed}")
        if not torch.equal(payload["targets"].long(), targets):
            raise ValueError(f"targets differ for seed {seed}")
        if comparable_specs(payload) != selected_specs:
            raise ValueError(f"checkpoint/temperature specs differ for seed {seed}")
        if payload.get("selection_label") != reference.get("selection_label"):
            raise ValueError(f"selection labels differ for seed {seed}")
    models = {}
    for seed, payload in payloads.items():
        for family in ("c1", "c100"):
            probabilities = payload["probabilities"][family].double()
            if probabilities.shape != (3925, 10):
                raise ValueError(f"invalid matrix: {family} seed={seed}")
            if not torch.allclose(
                probabilities.sum(1), torch.ones(3925, dtype=torch.double),
                atol=1e-6, rtol=0,
            ):
                raise ValueError(f"probabilities do not sum to one: {family} seed={seed}")
            models[f"{family}_s{seed}"] = probabilities

    model_summaries = {
        name: model_summary(probabilities, targets)
        for name, probabilities in models.items()
    }
    pair_reports = [
        pair_report(left, right, relation, models, targets, class_names)
        for left, right, relation in PAIR_SPECS
    ]
    paired_cross = [
        report for report in pair_reports
        if report["relation"] == "cross_family_same_seed"
    ]
    aggregate = {}
    scalar_paths = {
        "global_linear_CKA": lambda row: row["linear_CKA"]["globally_centered_labels"],
        "within_linear_CKA": lambda row: row["linear_CKA"]["within_class_centered_labels"],
        "between_prototype_linear_CKA": lambda row: row["linear_CKA"]["between_class_prototypes"],
        "top1_agreement": lambda row: row["agreement"]["top1_agreement_fraction"],
        "top5_overlap": lambda row: row["agreement"]["mean_top5_overlap_count"],
        "within_top1_direction_angle": lambda row: row["within_class_covariance_principal_angles"]["pooled_within_class_residuals"]["top_eigenvector_angle_degrees"],
        "raw_interclass_similarity_pearson": lambda row: row["interclass_similarity"]["raw_probability_cosine"]["comparison"]["upper_triangle_pearson"],
        "centered_interclass_similarity_pearson": lambda row: row["interclass_similarity"]["globally_centered_prototype_cosine"]["comparison"]["upper_triangle_pearson"],
    }
    for name, getter in scalar_paths.items():
        aggregate[name] = mean_and_sample_sd([getter(row) for row in paired_cross])
    cca_rows = [
        row["CCA"]["within_class_centered_labels"]["canonical_correlations"]
        for row in paired_cross
    ]
    common = min(map(len, cca_rows))
    aggregate["within_class_CCA_spectrum"] = [
        mean_and_sample_sd([row[index] for row in cca_rows])
        for index in range(common)
    ]
    result = {
        "audit_schema_version": 1,
        "question": (
            "Do the selected C1 and Random-C100 soft labels use the same "
            "information channel, after accounting for ordinary Teacher-seed variation?"
        ),
        "protocol": {
            "dataset": "complete 3925-image ImageNette test split",
            "preprocessing": "Resize(256)+CenterCrop(224)+ImageNet normalization",
            "selection_label": reference.get("selection_label"),
            "C1": selected_specs["c1"],
            "Random_C100": selected_specs["c100"],
            "teacher_seeds": [43, 44],
            "primary_pairs": ["c1_s43 vs c100_s43", "c1_s44 vs c100_s44"],
            "seed_variation_controls": ["c1_s43 vs c1_s44", "c100_s43 vs c100_s44"],
        },
        "class_names": class_names,
        "model_summaries": model_summaries,
        "paired_cross_family_aggregate": aggregate,
        "pair_reports": pair_reports,
    }
    output = Path(args.output)
    atomic_json_dump(result, output)

    csv_path = output.with_name("pair_scalar_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "left", "right", "relation", "global_CKA", "within_CKA",
            "between_CKA", "top1_agreement", "top5_overlap",
            "top1_within_direction_angle_degrees", "within_CCA_sum_rho2",
            "within_CCA_dims_ge_0p9", "raw_interclass_pearson",
            "centered_interclass_pearson",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_reports:
            writer.writerow({
                "left": row["left"],
                "right": row["right"],
                "relation": row["relation"],
                "global_CKA": row["linear_CKA"]["globally_centered_labels"],
                "within_CKA": row["linear_CKA"]["within_class_centered_labels"],
                "between_CKA": row["linear_CKA"]["between_class_prototypes"],
                "top1_agreement": row["agreement"]["top1_agreement_fraction"],
                "top5_overlap": row["agreement"]["mean_top5_overlap_count"],
                "top1_within_direction_angle_degrees": row["within_class_covariance_principal_angles"]["pooled_within_class_residuals"]["top_eigenvector_angle_degrees"],
                "within_CCA_sum_rho2": row["CCA"]["within_class_centered_labels"]["sum_squared_canonical_correlations"],
                "within_CCA_dims_ge_0p9": row["CCA"]["within_class_centered_labels"]["shared_dimensions_rho_ge_0p9"],
                "raw_interclass_pearson": row["interclass_similarity"]["raw_probability_cosine"]["comparison"]["upper_triangle_pearson"],
                "centered_interclass_pearson": row["interclass_similarity"]["globally_centered_prototype_cosine"]["comparison"]["upper_triangle_pearson"],
            })
    angle_csv = output.with_name("principal_angles_by_class.csv")
    with angle_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "left", "right", "relation", "class_id", "class_name",
            "top_eigenvector_angle_degrees", "top2_mean_angle_degrees",
            "top3_mean_angle_degrees", "top5_mean_angle_degrees",
            "within_class_linear_CKA",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_reports:
            per_class = row["within_class_covariance_principal_angles"]["per_class"]
            for class_row in per_class:
                writer.writerow({
                    "left": row["left"],
                    "right": row["right"],
                    "relation": row["relation"],
                    "class_id": class_row["class_id"],
                    "class_name": class_row["class_name"],
                    "top_eigenvector_angle_degrees": class_row["top_eigenvector_angle_degrees"],
                    "top2_mean_angle_degrees": class_row["subspaces"]["top_2"]["mean_angle_degrees"],
                    "top3_mean_angle_degrees": class_row["subspaces"]["top_3"]["mean_angle_degrees"],
                    "top5_mean_angle_degrees": class_row["subspaces"]["top_5"]["mean_angle_degrees"],
                    "within_class_linear_CKA": class_row["within_class_linear_CKA"],
                })
    cca_csv = output.with_name("cca_spectra.csv")
    with cca_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["left", "right", "relation", "scope", "component", "canonical_correlation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_reports:
            for scope, spectrum in row["CCA"].items():
                for component, correlation in enumerate(
                    spectrum["canonical_correlations"], start=1
                ):
                    writer.writerow({
                        "left": row["left"],
                        "right": row["right"],
                        "relation": row["relation"],
                        "scope": scope,
                        "component": component,
                        "canonical_correlation": correlation,
                    })
    print(json.dumps({
        "output": str(output),
        "pair_csv": str(csv_path),
        "principal_angle_csv": str(angle_csv),
        "cca_csv": str(cca_csv),
        "models": list(models),
        "pairs": len(pair_reports),
        "paired_cross_family_aggregate": aggregate,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--trajectory-root", required=True)
    collect_parser.add_argument("--test-root", required=True)
    collect_parser.add_argument("--teacher-seed", type=int, choices=(43, 44), required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--batch-size", type=int, default=256)
    collect_parser.add_argument("--workers", type=int, default=8)
    collect_parser.add_argument(
        "--c1-training-epoch", type=int,
        default=DEFAULT_SPECS["c1"]["training_epoch"],
    )
    collect_parser.add_argument(
        "--c1-temperature", type=float,
        default=DEFAULT_SPECS["c1"]["temperature"],
    )
    collect_parser.add_argument(
        "--c100-training-epoch", type=int,
        default=DEFAULT_SPECS["c100"]["training_epoch"],
    )
    collect_parser.add_argument(
        "--c100-temperature", type=float,
        default=DEFAULT_SPECS["c100"]["temperature"],
    )
    collect_parser.add_argument(
        "--selection-label", default="source-average best",
    )
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input-dir", required=True)
    analyze_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
