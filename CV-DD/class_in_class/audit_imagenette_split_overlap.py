import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(root):
    return sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def signatures(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L").resize((9, 8))
        pixels = list(image.getdata())
    dhash = 0
    for row in range(8):
        for column in range(8):
            dhash = (dhash << 1) | (
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return {
        "path": str(path),
        "class": path.parent.name,
        "name": path.name,
        "sha256": digest.hexdigest(),
        "dhash64": dhash,
    }


def scan(root, workers):
    paths = image_paths(root)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(signatures, paths))
    class_names = sorted({record["class"] for record in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    for record in records:
        record["class_index"] = class_to_index[record["class"]]
    return records


def overlap(left, right, threshold):
    left_names = {record["name"] for record in left}
    right_names = {record["name"] for record in right}
    left_sha = {record["sha256"] for record in left}
    right_sha = {record["sha256"] for record in right}
    left_dhash = {record["dhash64"] for record in left}
    right_dhash = {record["dhash64"] for record in right}

    right_by_class = {}
    for record in right:
        right_by_class.setdefault(record["class_index"], []).append(record["dhash64"])
    left_near = 0
    for record in left:
        candidates = right_by_class.get(record["class_index"], ())
        if any((record["dhash64"] ^ candidate).bit_count() <= threshold
               for candidate in candidates):
            left_near += 1

    left_by_class = {}
    for record in left:
        left_by_class.setdefault(record["class_index"], []).append(record["dhash64"])
    right_near = 0
    for record in right:
        candidates = left_by_class.get(record["class_index"], ())
        if any((record["dhash64"] ^ candidate).bit_count() <= threshold
               for candidate in candidates):
            right_near += 1

    return {
        "left_images": len(left),
        "right_images": len(right),
        "basename_intersection": len(left_names & right_names),
        "exact_file_sha256_intersection": len(left_sha & right_sha),
        "exact_dhash64_intersection": len(left_dhash & right_dhash),
        f"left_images_with_same_class_dhash_distance_le_{threshold}": left_near,
        f"right_images_with_same_class_dhash_distance_le_{threshold}": right_near,
    }


def main():
    parser = argparse.ArgumentParser("Audit ImageNette split overlap by content")
    parser.add_argument("--vlcp-root", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = {
        "vlcp_train": Path(args.vlcp_root) / "train",
        "vlcp_val": Path(args.vlcp_root) / "val",
        "official_train": Path(args.official_root) / "train",
        "official_test": Path(args.official_root) / "test",
    }
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name}: {root}")
        print(f"Scanning {name}: {root}", flush=True)

    records = {name: scan(root, args.workers) for name, root in roots.items()}
    pairs = (
        ("vlcp_train", "official_train"),
        ("vlcp_train", "official_test"),
        ("vlcp_val", "official_train"),
        ("vlcp_val", "official_test"),
    )
    result = {
        "definitions": {
            "exact_file_sha256": "identical encoded file bytes",
            "dhash64": "64-bit grayscale difference hash after EXIF transpose",
            "near_dhash": (
                "same sorted parent-class index and Hamming distance no greater than "
                f"{args.dhash_threshold}"
            ),
        },
        "roots": {name: str(root.resolve()) for name, root in roots.items()},
        "image_counts": {name: len(value) for name, value in records.items()},
        "class_counts": {
            name: len({record["class"] for record in value})
            for name, value in records.items()
        },
        "class_names": {
            name: sorted({record["class"] for record in value})
            for name, value in records.items()
        },
        "overlaps": {
            f"{left}_vs_{right}": overlap(
                records[left], records[right], args.dhash_threshold
            )
            for left, right in pairs
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
