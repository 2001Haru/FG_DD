import argparse
import concurrent.futures
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch


METRIC_FIELDS = (
    "entropy",
    "effective_class_count_exp_entropy",
    "target_probability",
    "non_target_mass",
    "cutmix_constituent_class_mass",
    "cutmix_non_constituent_mass",
    "cutmix_area_weighted_constituent_probability",
    "cutmix_hard_mixture_cross_entropy",
    "cutmix_realized_base_fraction",
    "maximum_probability",
    "top1_margin",
    "argmax_matches_original_target",
    "centered_spectral_effective_rank",
    "centered_covariance_participation_rank",
    "uncentered_spectral_effective_rank",
)


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def synthetic_targets(synthetic_root):
    classes = sorted(path for path in synthetic_root.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise ValueError(f"expected 10 synthetic class directories: {synthetic_root}")
    targets = []
    for class_id, directory in enumerate(classes):
        images = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        targets.extend([class_id] * len(images))
    if len(targets) <= 0 or len(targets) % 10 != 0:
        raise ValueError(
            f"expected a non-empty, class-balanced ImageNette ImageFolder; "
            f"found {len(targets)} images: {synthetic_root}"
        )
    counts = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=10)
    if counts.numel() != 10 or not torch.all(counts.eq(counts[0])):
        raise ValueError(f"ImageNette classes are not balanced: {counts.tolist()}")
    return torch.tensor(targets, dtype=torch.long)


def effective_ranks(probabilities):
    def spectral(matrix):
        singular = torch.linalg.svdvals(matrix.double())
        total = singular.sum()
        if total <= 0:
            return 0.0
        weights = singular / total
        return float(torch.exp(-(weights * weights.clamp_min(1e-15).log()).sum()))

    centered = probabilities.double() - probabilities.double().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    participation = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-15)
    )
    return spectral(centered), participation, spectral(probabilities.double())


def analyze_root(task):
    torch.set_num_threads(1)
    (
        partition, c, teacher_seed, recovery_seed, synthetic_root, fkd_root,
        epochs, epoch_stride, temperature, sampler_seed,
    ) = task
    synthetic_root, fkd_root = Path(synthetic_root), Path(fkd_root)
    targets = synthetic_targets(synthetic_root)
    generator = torch.Generator().manual_seed(sampler_seed)
    selected_epochs = set(range(0, epochs, epoch_stride))
    probabilities, aligned_targets = [], []
    paired_targets, realized_base_fractions = [], []
    files_loaded = 0

    for epoch in range(epochs):
        permutation = torch.randperm(len(targets), generator=generator)
        if epoch not in selected_epochs:
            continue
        epoch_root = fkd_root / f"epoch_{epoch}"
        if not epoch_root.is_dir():
            raise FileNotFoundError(epoch_root)
        batch_files = sorted(
            epoch_root.glob("batch_*.tar"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        if not batch_files:
            raise ValueError(f"no FKD batches found: {epoch_root}")
        offset = 0
        for batch_file in batch_files:
            config = torch.load(batch_file, map_location="cpu", weights_only=False)
            logits = config[5].float()
            if logits.ndim != 2 or logits.shape[1] != 10:
                raise ValueError(f"expected marginalized 10-way logits: {batch_file}")
            size = logits.shape[0]
            probabilities.append(torch.softmax(logits / temperature, dim=1))
            batch_targets = targets[permutation[offset:offset + size]]
            mix_index = config[2].long()
            if mix_index.numel() != size:
                raise ValueError(f"CutMix index size mismatch: {batch_file}")
            bbox = config[4]
            if bbox is None or len(bbox) != 4:
                raise ValueError(f"expected CutMix bbox: {batch_file}")
            x1, y1, x2, y2 = [int(value) for value in bbox]
            mixed_area = max(0, x2 - x1) * max(0, y2 - y1)
            # Relabel and post-eval both operate on 224x224 ImageNette crops.
            realized_base_fraction = 1.0 - mixed_area / float(224 * 224)
            aligned_targets.append(batch_targets)
            paired_targets.append(batch_targets[mix_index])
            realized_base_fractions.append(
                torch.full((size,), realized_base_fraction, dtype=torch.float64)
            )
            offset += size
            files_loaded += 1
        if offset != len(targets):
            raise ValueError(f"epoch {epoch} covers {offset}, expected {len(targets)}")

    q = torch.cat(probabilities).double()
    target = torch.cat(aligned_targets)
    paired_target = torch.cat(paired_targets)
    base_fraction = torch.cat(realized_base_fractions)
    target_probability = q.gather(1, target[:, None]).squeeze(1)
    paired_probability = q.gather(1, paired_target[:, None]).squeeze(1)
    same_class = paired_target.eq(target)
    constituent_mass = target_probability + torch.where(
        same_class, torch.zeros_like(paired_probability), paired_probability
    )
    weighted_constituent_probability = (
        base_fraction * target_probability
        + (1.0 - base_fraction) * paired_probability
    )
    hard_mixture_ce = -(
        base_fraction * target_probability.clamp_min(1e-15).log()
        + (1.0 - base_fraction) * paired_probability.clamp_min(1e-15).log()
    )
    entropy = -(q * q.clamp_min(1e-15).log()).sum(1)
    top2 = q.topk(2, dim=1).values
    centered_rank, participation_rank, uncentered_rank = effective_ranks(q)
    result = {
        "partition": partition,
        "C": c,
        "teacher_seed": teacher_seed,
        "recovery_seed": recovery_seed,
        "epochs_total": epochs,
        "epoch_stride": epoch_stride,
        "epochs_sampled": len(selected_epochs),
        "fkd_files_loaded": files_loaded,
        "label_rows": q.shape[0],
        "temperature": temperature,
        "entropy": float(entropy.mean()),
        "entropy_sample_sd_across_rows": float(entropy.std(unbiased=True)),
        "effective_class_count_exp_entropy": float(torch.exp(entropy).mean()),
        "target_probability": float(target_probability.mean()),
        "non_target_mass": float((1.0 - target_probability).mean()),
        "cutmix_constituent_class_mass": float(constituent_mass.mean()),
        "cutmix_non_constituent_mass": float((1.0 - constituent_mass).mean()),
        "cutmix_area_weighted_constituent_probability": float(
            weighted_constituent_probability.mean()
        ),
        "cutmix_hard_mixture_cross_entropy": float(hard_mixture_ce.mean()),
        "cutmix_realized_base_fraction": float(base_fraction.mean()),
        "maximum_probability": float(q.max(1).values.mean()),
        "top1_margin": float((top2[:, 0] - top2[:, 1]).mean()),
        "argmax_matches_original_target": float(q.argmax(1).eq(target).double().mean()),
        "centered_spectral_effective_rank": centered_rank,
        "centered_covariance_participation_rank": participation_rank,
        "uncentered_spectral_effective_rank": uncentered_rank,
        "synthetic_root": str(synthetic_root),
        "fkd_root": str(fkd_root),
    }
    return result


def summarize_roots(rows, teacher_seeds, recovery_seeds):
    summary = {}
    for field in METRIC_FIELDS:
        values = [row[field] for row in rows]
        teacher_means = []
        recovery_variances = []
        for teacher in teacher_seeds:
            current = [
                row[field] for row in rows if row["teacher_seed"] == teacher
            ]
            if len(current) != len(recovery_seeds):
                raise ValueError(f"incomplete roots for Teacher {teacher}, field {field}")
            teacher_means.append(statistics.fmean(current))
            recovery_variances.append(sample_sd(current) ** 2)
        summary[field] = {
            "mean_across_teacher_recovery_roots": statistics.fmean(values),
            "sample_sd_across_six_roots": sample_sd(values),
            "pooled_recovery_seed_sd_within_teacher": (
                statistics.fmean(recovery_variances) ** 0.5
            ),
            "teacher_seed_sd_of_recovery_means": sample_sd(teacher_means),
            "by_teacher_seed_mean": {
                str(teacher): teacher_means[index]
                for index, teacher in enumerate(teacher_seeds)
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser("Audit student-consumed marginalized FKD labels")
    parser.add_argument("--random-root", required=True)
    parser.add_argument("--cluster-root", required=True)
    parser.add_argument("--c-values", nargs="+", type=int, required=True)
    parser.add_argument("--teacher-seeds", nargs="+", type=int, default=(43, 44))
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epoch-stride", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = {"random": Path(args.random_root), "cluster": Path(args.cluster_root)}
    tasks = []
    for partition, root in roots.items():
        for c in args.c_values:
            for teacher in args.teacher_seeds:
                teacher_root = root / f"tseed{teacher}"
                for recovery in args.recovery_seeds:
                    tasks.append((
                        partition, c, teacher, recovery,
                        str(teacher_root / "synthetic" / f"cic_t_c{c}_ipc10_rseed{recovery}"),
                        str(teacher_root / "fkd" / f"cic_t_c{c}_rseed{recovery}_bs10_ipc10"),
                        args.epochs, args.epoch_stride, args.temperature, args.sampler_seed,
                    ))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        roots_rows = list(executor.map(analyze_root, tasks))
    roots_rows.sort(key=lambda row: (
        row["partition"], row["C"], row["teacher_seed"], row["recovery_seed"]
    ))

    comparisons = {}
    for c in args.c_values:
        by_partition = {}
        for partition in ("random", "cluster"):
            current = [
                row for row in roots_rows
                if row["partition"] == partition and row["C"] == c
            ]
            by_partition[partition] = summarize_roots(
                current, args.teacher_seeds, args.recovery_seeds
            )
        differences = {}
        for field in METRIC_FIELDS:
            paired = []
            for teacher in args.teacher_seeds:
                for recovery in args.recovery_seeds:
                    cluster_value = next(
                        row[field] for row in roots_rows
                        if row["partition"] == "cluster" and row["C"] == c
                        and row["teacher_seed"] == teacher
                        and row["recovery_seed"] == recovery
                    )
                    random_value = next(
                        row[field] for row in roots_rows
                        if row["partition"] == "random" and row["C"] == c
                        and row["teacher_seed"] == teacher
                        and row["recovery_seed"] == recovery
                    )
                    paired.append({
                        "teacher_seed": teacher,
                        "recovery_seed": recovery,
                        "value": cluster_value - random_value,
                    })
            teacher_means = [
                statistics.fmean([
                    item["value"] for item in paired if item["teacher_seed"] == teacher
                ]) for teacher in args.teacher_seeds
            ]
            recovery_variances = [
                sample_sd([
                    item["value"] for item in paired if item["teacher_seed"] == teacher
                ]) ** 2 for teacher in args.teacher_seeds
            ]
            values = [item["value"] for item in paired]
            differences[field] = {
                "cluster_minus_random_mean": statistics.fmean(values),
                "sample_sd_across_six_paired_roots": sample_sd(values),
                "pooled_recovery_seed_sd_within_teacher": (
                    statistics.fmean(recovery_variances) ** 0.5
                ),
                "teacher_seed_sd_of_recovery_means": sample_sd(teacher_means),
                "by_teacher_seed_mean": {
                    str(teacher): teacher_means[index]
                    for index, teacher in enumerate(args.teacher_seeds)
                },
            }
        comparisons[f"C{c}"] = {
            "random": by_partition["random"],
            "cluster": by_partition["cluster"],
            "paired_cluster_minus_random": differences,
        }

    result = {
        "definition": {
            "consumed_distribution": "softmax(saved marginalized 10-way FKD logits / T20)",
            "entropy": "row-wise Shannon entropy of the consumed 10-way distribution",
            "target_probability": (
                "probability assigned to the original row's coarse class; retained only "
                "as a diagnostic, not the correct CutMix target-mass definition"
            ),
            "cutmix_constituent_mass": (
                "sum of probability on the two CutMix constituent coarse classes "
                "(counted once when both classes are equal)"
            ),
            "cutmix_area_weighted_probability": (
                "realized bbox-area weighted probability of the two constituent classes"
            ),
            "effective_rank": (
                "spectral effective rank and covariance participation rank of the "
                "collected 10-way probability matrix"
            ),
            "cutmix_note": (
                "Official/current code aliases origin_images=images, then CutMix mutates images "
                "in place. Teacher logits are therefore generated on the post-CutMix image, "
                "and student loading replays the same mix_index and bbox."
            ),
        },
        "teacher_seeds": args.teacher_seeds,
        "recovery_seeds": args.recovery_seeds,
        "c_values": args.c_values,
        "epochs": args.epochs,
        "epoch_stride": args.epoch_stride,
        "temperature": args.temperature,
        "root_metrics": roots_rows,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["C", "partition", *METRIC_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in args.c_values:
            for partition in ("random", "cluster"):
                writer.writerow({
                    "C": c,
                    "partition": partition,
                    **{
                        field: comparisons[f"C{c}"][partition][field][
                            "mean_across_teacher_recovery_roots"
                        ] for field in METRIC_FIELDS
                    },
                })
    print(json.dumps({
        "output": str(output),
        "csv": str(csv_output),
        "roots_analyzed": len(roots_rows),
        "comparisons": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
