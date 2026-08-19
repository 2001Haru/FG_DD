import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


def decode(items):
    return [item.decode() if isinstance(item, bytes) else str(item) for item in items]


def convert(raw_file, root, split, mapping):
    payload = load(raw_file)
    images = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    for index, (image, fine, coarse) in enumerate(
        zip(images, payload[b"fine_labels"], payload[b"coarse_labels"])
    ):
        previous = mapping.setdefault(str(fine), coarse)
        if previous != coarse:
            raise RuntimeError(f"fine label {fine} maps to multiple coarse labels")
        name = f"{index:05d}.png"
        value = Image.fromarray(image.astype(np.uint8))
        for space, label, width in (("fine", fine, 3), ("coarse", coarse, 2)):
            directory = root / space / split / f"{label:0{width}d}"
            directory.mkdir(parents=True, exist_ok=True)
            value.save(directory / name)
    return len(images)


def main():
    parser = argparse.ArgumentParser("Prepare official CIFAR-100 fine/coarse ImageFolder datasets")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw, root = Path(args.raw_dir), Path(args.output_dir)
    if not all((raw / name).is_file() for name in ("train", "test", "meta")):
        raise FileNotFoundError(f"incomplete official CIFAR-100 pickle directory: {raw}")

    meta = load(raw / "meta")
    fine_names, coarse_names = decode(meta[b"fine_label_names"]), decode(meta[b"coarse_label_names"])
    mapping = {}
    train_count = convert(raw / "train", root, "train", mapping)
    test_count = convert(raw / "test", root, "test", mapping)
    inverse = {str(coarse): [] for coarse in range(20)}
    for fine in range(100):
        inverse[str(mapping[str(fine)])].append(fine)
    if any(len(values) != 5 for values in inverse.values()):
        raise RuntimeError("official hierarchy is not five fine classes per coarse class")
    manifest = {
        "fine_names": fine_names,
        "coarse_names": coarse_names,
        "fine_to_coarse": {str(fine): mapping[str(fine)] for fine in range(100)},
        "coarse_to_fine": inverse,
        "train_images": train_count,
        "test_images": test_count,
    }
    with (root / "hierarchy.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Prepared hierarchy: train={train_count}, test={test_count}, root={root}")


if __name__ == "__main__":
    main()
