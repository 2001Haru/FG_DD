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


def main():
    parser = argparse.ArgumentParser("Prepare balanced random ImageNette subclasses")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subclasses", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = Path(args.source_root), Path(args.output_dir)
    if args.subclasses < 2:
        raise ValueError("subclasses must be at least 2")

    coarse_dirs = sorted(path for path in (source / "train").iterdir() if path.is_dir())
    if len(coarse_dirs) != 10:
        raise RuntimeError(f"expected 10 ImageNette train classes, found {len(coarse_dirs)}")
    coarse_names = [path.name for path in coarse_dirs]
    if sorted(path.name for path in (source / "val").iterdir() if path.is_dir()) != coarse_names:
        raise RuntimeError("train/val coarse class directories do not match")

    split_counts, started = {}, time.time()
    for split_index, split in enumerate(("train", "val")):
        per_pseudo = {}
        for coarse_id, coarse_name in enumerate(coarse_names):
            images = sorted(
                path for path in (source / split / coarse_name).iterdir()
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
                modes = {"hardlink": 0, "copy": 0, "existing": 0}
                for image in chunk:
                    modes[materialize(
                        image, output / split / f"{pseudo_id:03d}" / image.name
                    )] += 1
                per_pseudo[str(pseudo_id)] = len(chunk)
                print(
                    f"split={split} coarse={coarse_id + 1}/10 subclass={local_subclass + 1}/"
                    f"{args.subclasses} images={len(chunk)} hardlink={modes['hardlink']} "
                    f"copy={modes['copy']} existing={modes['existing']} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        split_counts[split] = per_pseudo

    total_classes = 10 * args.subclasses
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
        "partition_seed": args.seed,
        "num_coarse_classes": 10,
        "subclasses_per_coarse": args.subclasses,
        "num_pseudo_classes": total_classes,
        "coarse_names": coarse_names,
        "fine_to_coarse": fine_to_coarse,
        "coarse_to_fine": coarse_to_fine,
        "split_counts": split_counts,
        "source_train_images": sum(split_counts["train"].values()),
        "source_val_images": sum(split_counts["val"].values()),
    }
    (output / "hierarchy.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared ImageNette random subclasses: C={args.subclasses}, "
        f"classes={total_classes}, train={manifest['source_train_images']}, "
        f"val={manifest['source_val_images']}, output={output}"
    )


if __name__ == "__main__":
    main()
