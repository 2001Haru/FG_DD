import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def materialize(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def source_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            stat = path.stat()
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def valid(output, expected):
    manifest = output / "selection.json"
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if any(record.get(key) != value for key, value in expected.items()):
        return False
    classes = sorted(path for path in output.iterdir() if path.is_dir())
    if len(classes) != 10:
        return False
    expected_per_class = int(expected["images_per_class"])
    return all(sum(
        path.is_file() and path.suffix.lower() in EXTENSIONS
        for path in directory.iterdir()
    ) == expected_per_class for directory in classes)


def main():
    parser = argparse.ArgumentParser("Prepare stratified random real ImageNette IPC10")
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--repair-invalid-output", action="store_true")
    args = parser.parse_args()
    source = Path(args.source_train).resolve()
    output = Path(args.output_dir).resolve()
    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise RuntimeError(f"expected 10 source classes, found {len(classes)}")
    fingerprint = source_fingerprint(source)
    expected = {
        "kind": "imagenette_stratified_random_real_subset",
        "source_train": str(source),
        "source_fingerprint": fingerprint,
        "seed": args.seed,
        "images_per_class": args.images_per_class,
        "total_images": 10 * args.images_per_class,
    }
    if output.exists() and valid(output, expected):
        print(f"Reusing valid random-real subset: {output}")
        return
    if output.exists():
        safe_derived_name = (
            output.name.startswith("tseed") or output.name.startswith("real_ipc")
        )
        if not args.repair_invalid_output or not safe_derived_name:
            raise RuntimeError(f"invalid output; refusing replacement: {output}")
        archive = Path(str(output) + f".invalid_{time.strftime('%Y%m%d_%H%M%S')}")
        output.rename(archive)
        print(f"Archived invalid subset: {output} -> {archive}")

    selected = {}
    for class_id, class_dir in enumerate(classes):
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if len(images) < args.images_per_class:
            raise RuntimeError(f"class {class_dir.name} has only {len(images)} images")
        rng = random.Random(args.seed + class_id * 1_000_003)
        chosen = rng.sample(images, args.images_per_class)
        destination = output / class_dir.name
        for image in chosen:
            materialize(image, destination / image.name)
        selected[class_dir.name] = [image.name for image in chosen]
    record = {**expected, "selected": selected}
    (output / "selection.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if not valid(output, expected):
        raise RuntimeError(f"new random-real subset failed validation: {output}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
