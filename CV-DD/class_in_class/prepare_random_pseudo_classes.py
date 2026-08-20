import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def materialize(source, destination):
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def split_coarse_class(source_dir, output, split, coarse_id, groups, seed):
    images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in EXTENSIONS)
    if len(images) % groups:
        raise RuntimeError(
            f"{split} coarse class {coarse_id} has {len(images)} images, not divisible by {groups}"
        )
    rng = random.Random(seed * 1_000_003 + coarse_id * 101 + (0 if split == "train" else 1))
    rng.shuffle(images)
    per_group = len(images) // groups
    modes = {"existing": 0, "hardlink": 0, "copy": 0}
    for group_id in range(groups):
        pseudo_id = coarse_id * groups + group_id
        selected = images[group_id * per_group:(group_id + 1) * per_group]
        destination_dir = output / split / f"{pseudo_id:03d}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in selected:
            modes[materialize(source, destination_dir / source.name)] += 1
    return per_group, modes


def main():
    parser = argparse.ArgumentParser(
        "Randomly and evenly split every CIFAR-100 coarse class into five pseudo classes"
    )
    parser.add_argument("--coarse-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups-per-coarse", type=int, default=5)
    args = parser.parse_args()
    source, output = Path(args.coarse_data), Path(args.output_dir)
    groups = args.groups_per_coarse
    if groups != 5:
        raise RuntimeError("the controlled experiment requires five pseudo classes per coarse class")

    split_counts = {}
    started = time.time()
    for split, expected_per_coarse in (("train", 2500), ("test", 500)):
        split_root = source / split
        classes = sorted(path for path in split_root.iterdir() if path.is_dir())
        if len(classes) != 20:
            raise RuntimeError(f"{split}: expected 20 coarse directories, found {len(classes)}")
        per_group = None
        for coarse_id, class_dir in enumerate(classes):
            if class_dir.name != f"{coarse_id:02d}":
                raise RuntimeError(f"unexpected coarse class order/name: {class_dir}")
            count = len([path for path in class_dir.iterdir() if path.suffix.lower() in EXTENSIONS])
            if count != expected_per_coarse:
                raise RuntimeError(
                    f"{split} coarse class {coarse_id}: expected {expected_per_coarse}, found {count}"
                )
            per_group, modes = split_coarse_class(
                class_dir, output, split, coarse_id, groups, args.seed
            )
            print(
                f"random partition: split={split} coarse={coarse_id + 1}/20 "
                f"images={expected_per_coarse} hardlink={modes['hardlink']} "
                f"copy={modes['copy']} existing={modes['existing']} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        split_counts[split] = per_group

    fine_to_coarse = {
        str(pseudo_id): pseudo_id // groups for pseudo_id in range(20 * groups)
    }
    coarse_to_fine = {
        str(coarse_id): list(range(coarse_id * groups, (coarse_id + 1) * groups))
        for coarse_id in range(20)
    }
    manifest = {
        "kind": "random_balanced_pseudo_classes",
        "partition_seed": args.seed,
        "groups_per_coarse": groups,
        "source": str(source.resolve()),
        "train_images_per_pseudo_class": split_counts["train"],
        "test_images_per_pseudo_class": split_counts["test"],
        "fine_to_coarse": fine_to_coarse,
        "coarse_to_fine": coarse_to_fine,
    }
    manifest_path = output / "hierarchy.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split, expected in (("train", 500), ("test", 100)):
        counts = [
            len([path for path in (output / split / f"{pseudo_id:03d}").iterdir()
                 if path.suffix.lower() in EXTENSIONS])
            for pseudo_id in range(100)
        ]
        if counts != [expected] * 100:
            raise RuntimeError(f"{split}: pseudo-class counts are not all {expected}: {counts}")
    print(
        f"Prepared random100 seed={args.seed}: train=100x500, test=100x100, root={output}"
    )


if __name__ == "__main__":
    main()
