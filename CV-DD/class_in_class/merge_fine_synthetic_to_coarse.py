import argparse
import json
import shutil
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser("Merge CIFAR-100 fine synthetic classes into coarse ImageFolder classes")
    parser.add_argument("--fine-dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fine-ipc", type=int, default=5)
    args = parser.parse_args()
    source, output = Path(args.fine_dir), Path(args.output_dir)
    with Path(args.mapping).open(encoding="utf-8") as handle:
        mapping = json.load(handle)["fine_to_coarse"]
    for fine in range(100):
        images = sorted(path for path in (source / f"new{fine:03d}").iterdir()
                        if path.suffix.lower() in EXTENSIONS)
        if len(images) != args.fine_ipc:
            raise RuntimeError(f"fine class {fine} contains {len(images)}, expected {args.fine_ipc}")
        coarse_dir = output / f"new{int(mapping[str(fine)]):03d}"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            shutil.copy2(image, coarse_dir / f"fine{fine:03d}_id{index:03d}{image.suffix.lower()}")
    counts = [len(list((output / f"new{coarse:03d}").glob("*"))) for coarse in range(20)]
    if counts != [25] * 20:
        raise RuntimeError(f"merged counts differ from 25/class: {counts}")
    print(f"Merged 500 images into 20 coarse classes: {output}")


if __name__ == "__main__":
    main()
