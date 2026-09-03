import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_id(path):
    name = path.name
    if not name.startswith("new") or not name[3:].isdigit():
        raise ValueError(f"unexpected recovery class directory: {path}")
    return int(name[3:])


def image_files(root):
    return sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_output(output, coarse_classes, images_per_coarse, expected_total):
    class_dirs = sorted(path for path in output.iterdir() if path.is_dir())
    if [path.name for path in class_dirs] != [f"{i:05d}" for i in range(coarse_classes)]:
        return False, "coarse directory set mismatch"
    counts = [len(image_files(path)) for path in class_dirs]
    if counts != [images_per_coarse] * coarse_classes:
        return False, f"per-parent image counts mismatch: {counts}"
    if sum(counts) != expected_total:
        return False, f"total image count mismatch: {sum(counts)}"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        "Collapse native subclass recovery directories into a coarse ImageFolder"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--native-classes", type=int, required=True)
    parser.add_argument("--coarse-classes", type=int, required=True)
    parser.add_argument("--images-per-native", type=int, default=1)
    args = parser.parse_args()

    source = Path(args.input_dir).resolve()
    hierarchy_path = Path(args.hierarchy).resolve()
    output = Path(args.output_dir).resolve()
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    mapping = {
        int(index): int(parent)
        for index, parent in hierarchy["fine_to_coarse"].items()
    }
    expected_native = set(range(args.native_classes))
    if set(mapping) != expected_native:
        raise RuntimeError("hierarchy does not map every native recovery class")
    if set(mapping.values()) != set(range(args.coarse_classes)):
        raise RuntimeError("hierarchy parent set does not match coarse classes")

    source_dirs = sorted(
        (path for path in source.iterdir() if path.is_dir()), key=native_id
    )
    if [native_id(path) for path in source_dirs] != list(range(args.native_classes)):
        raise RuntimeError("native recovery directory set/count mismatch")
    sources = {}
    for class_dir in source_dirs:
        class_id = native_id(class_dir)
        files = image_files(class_dir)
        if len(files) != args.images_per_native:
            raise RuntimeError(
                f"native class {class_id} has {len(files)} images; "
                f"expected {args.images_per_native}"
            )
        sources[class_id] = files

    images_per_coarse = args.images_per_native * (
        args.native_classes // args.coarse_classes
    )
    expected_total = args.native_classes * args.images_per_native
    manifest_path = output / ".collapse_manifest.json"
    expected_provenance = {
        "input_dir": str(source),
        "hierarchy": str(hierarchy_path),
        "hierarchy_sha256": sha256(hierarchy_path),
        "native_classes": args.native_classes,
        "coarse_classes": args.coarse_classes,
        "images_per_native": args.images_per_native,
        "images_per_coarse": images_per_coarse,
        "total_images": expected_total,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid, reason = validate_output(
            output, args.coarse_classes, images_per_coarse, expected_total
        )
        if existing.get("provenance") == expected_provenance and valid:
            print(json.dumps(existing, indent=2))
            return
        raise RuntimeError(f"existing collapsed output is stale/invalid: {reason}")

    output.mkdir(parents=True, exist_ok=True)
    modes = {"hardlink": 0, "copy": 0, "existing": 0}
    records = []
    for class_id in range(args.native_classes):
        parent = mapping[class_id]
        parent_dir = output / f"{parent:05d}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sources[class_id]:
            destination = parent_dir / f"subclass{class_id:05d}_{source_path.name}"
            if destination.is_file():
                mode = "existing"
            else:
                mode = link_or_copy(source_path, destination)
            modes[mode] += 1
            records.append(
                {
                    "native_class": class_id,
                    "coarse_class": parent,
                    "source": str(source_path),
                    "destination": str(destination),
                    "mode": mode,
                }
            )

    valid, reason = validate_output(
        output, args.coarse_classes, images_per_coarse, expected_total
    )
    if not valid:
        raise RuntimeError(f"collapsed output failed validation: {reason}")
    payload = {
        "schema_version": 1,
        "definition": (
            "Native K*C recovery images grouped into K parent directories; "
            "image bytes are unchanged"
        ),
        "provenance": expected_provenance,
        "materialization_modes": modes,
        "records": records,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
