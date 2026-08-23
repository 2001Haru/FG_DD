import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path


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


def balanced_chunks(items, groups):
    base, remainder = divmod(len(items), groups)
    chunks, start = [], 0
    for group in range(groups):
        size = base + (1 if group < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def existing_partition_is_valid(
    output, source, subclasses, seed, source_validation_split="val"
):
    manifest_path = output / "hierarchy.json"
    if not manifest_path.is_file():
        return False, "manifest missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, f"manifest unreadable: {error}"
    expected_classes = 10 * subclasses
    expected_width = max(3, len(str(expected_classes - 1)))
    checks = (
        (manifest.get("kind") == "imagenette_balanced_random_subclasses", "kind mismatch"),
        (int(manifest.get("partition_seed", -1)) == seed, "seed mismatch"),
        (int(manifest.get("subclasses_per_coarse", -1)) == subclasses, "C mismatch"),
        (int(manifest.get("num_pseudo_classes", -1)) == expected_classes,
         "class count mismatch in manifest"),
        (int(manifest.get("class_name_width", 3)) == expected_width,
         "class-name width mismatch"),
        (Path(manifest.get("source_root", "")).resolve() == source.resolve(), "source mismatch"),
        (manifest.get("source_validation_split", "val") == source_validation_split,
         "source validation split mismatch"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    for split in ("train", "val"):
        directories = sorted(path for path in (output / split).iterdir() if path.is_dir()) \
            if (output / split).is_dir() else []
        expected_names = [f"{index:0{expected_width}d}" for index in range(expected_classes)]
        if [path.name for path in directories] != expected_names:
            return False, f"{split} directory set/count mismatch"
        expected_counts = manifest.get("split_counts", {}).get(split, {})
        for index, directory in enumerate(directories):
            count = len([
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            ])
            if count != int(expected_counts.get(str(index), -1)):
                return False, f"{split}/{directory.name} image count mismatch"
    return True, "valid"


def main():
    parser = argparse.ArgumentParser("Prepare balanced random ImageNette subclasses")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subclasses", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-validation-split", choices=("val", "test"), default="val",
        help="source split materialized as output/val",
    )
    parser.add_argument("--repair-invalid-output", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.source_root), Path(args.output_dir)
    if args.subclasses < 1:
        raise ValueError("subclasses must be at least 1")
    total_classes = 10 * args.subclasses
    class_name_width = max(3, len(str(total_classes - 1)))

    coarse_dirs = sorted(path for path in (source / "train").iterdir() if path.is_dir())
    if len(coarse_dirs) != 10:
        raise RuntimeError(f"expected 10 ImageNette train classes, found {len(coarse_dirs)}")
    coarse_names = [path.name for path in coarse_dirs]
    minimum_train_images = min(
        len([path for path in directory.iterdir()
             if path.is_file() and path.suffix.lower() in EXTENSIONS])
        for directory in coarse_dirs
    )
    if args.subclasses > minimum_train_images:
        raise ValueError(
            f"C={args.subclasses} exceeds the smallest parent train class "
            f"({minimum_train_images}); empty train subclasses are not allowed"
        )
    source_validation = source / args.source_validation_split
    if sorted(path.name for path in source_validation.iterdir() if path.is_dir()) != coarse_names:
        raise RuntimeError(
            f"train/{args.source_validation_split} coarse class directories do not match"
        )

    valid, reason = existing_partition_is_valid(
        output, source, args.subclasses, args.seed, args.source_validation_split
    ) if output.exists() else (False, "output missing")
    if valid:
        print(f"Existing partition is valid, reusing: {output}")
        return
    if output.exists():
        if not args.repair_invalid_output:
            raise RuntimeError(
                f"invalid existing partition ({reason}): {output}; "
                "pass --repair-invalid-output to rebuild"
            )
        resolved_output, resolved_source = output.resolve(), source.resolve()
        if resolved_output == resolved_source or not output.name.startswith("random_c"):
            raise RuntimeError(f"refusing to remove unsafe output path: {resolved_output}")
        archive = Path(
            str(resolved_output) + f".invalid_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        if archive.exists():
            raise RuntimeError(f"invalid-partition archive already exists: {archive}")
        output.rename(archive)
        print(
            f"Archived invalid derived partition ({reason}): {resolved_output} -> {archive}",
            flush=True,
        )

    split_counts, started = {}, time.time()
    source_splits = (("train", "train"), ("val", args.source_validation_split))
    for split_index, (output_split, source_split) in enumerate(source_splits):
        per_pseudo = {}
        for coarse_id, coarse_name in enumerate(coarse_names):
            images = sorted(
                path for path in (source / source_split / coarse_name).iterdir()
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )
            rng = random.Random(
                args.seed * 1_000_003 + args.subclasses * 10_007
                + coarse_id * 101 + split_index
            )
            rng.shuffle(images)
            chunks = balanced_chunks(images, args.subclasses)
            for local_subclass, chunk in enumerate(chunks):
                pseudo_id = coarse_id * args.subclasses + local_subclass
                destination_dir = (
                    output / output_split / f"{pseudo_id:0{class_name_width}d}"
                )
                destination_dir.mkdir(parents=True, exist_ok=True)
                modes = {"hardlink": 0, "copy": 0, "existing": 0}
                for image in chunk:
                    modes[materialize(
                        image, destination_dir / image.name
                    )] += 1
                per_pseudo[str(pseudo_id)] = len(chunk)
                print(
                    f"split={output_split} source_split={source_split} "
                    f"coarse={coarse_id + 1}/10 subclass={local_subclass + 1}/"
                    f"{args.subclasses} images={len(chunk)} hardlink={modes['hardlink']} "
                    f"copy={modes['copy']} existing={modes['existing']} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        split_counts[output_split] = per_pseudo

    fine_to_coarse = {
        str(index): index // args.subclasses for index in range(total_classes)
    }
    coarse_to_fine = {
        str(coarse): list(range(
            coarse * args.subclasses, (coarse + 1) * args.subclasses
        ))
        for coarse in range(10)
    }
    manifest = {
        "kind": "imagenette_balanced_random_subclasses",
        "source_root": str(source.resolve()),
        "source_validation_split": args.source_validation_split,
        "partition_seed": args.seed,
        "num_coarse_classes": 10,
        "subclasses_per_coarse": args.subclasses,
        "num_pseudo_classes": total_classes,
        "class_name_width": class_name_width,
        "coarse_names": coarse_names,
        "fine_to_coarse": fine_to_coarse,
        "coarse_to_fine": coarse_to_fine,
        "split_counts": split_counts,
        "source_train_images": sum(split_counts["train"].values()),
        "source_val_images": sum(split_counts["val"].values()),
        "minimum_parent_train_images": minimum_train_images,
    }
    (output / "hierarchy.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    valid, reason = existing_partition_is_valid(
        output, source, args.subclasses, args.seed, args.source_validation_split
    )
    if not valid:
        raise RuntimeError(f"newly generated partition failed validation: {reason}")
    print(
        f"Prepared ImageNette random subclasses: C={args.subclasses}, "
        f"classes={total_classes}, train={manifest['source_train_images']}, "
        f"val={manifest['source_val_images']}, output={output}"
    )


if __name__ == "__main__":
    main()
