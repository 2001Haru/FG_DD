import argparse
import json
import os
import shutil
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser("Create a nested IPC subset from an ImageFolder")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ipc", type=int, required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise RuntimeError(f"expected 10 classes, found {len(classes)}: {source}")
    expected = {
        "kind": "nested_imagefolder_ipc_subset",
        "source": str(source),
        "ipc": args.ipc,
        "total_images": 10 * args.ipc,
    }
    manifest = output / "subset.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        count = sum(
            path.is_file() and path.suffix.lower() in EXTENSIONS
            for path in output.rglob("*")
        )
        if all(record.get(k) == v for k, v in expected.items()) and count == 10 * args.ipc:
            print(f"Reusing valid nested IPC subset: {output}")
            return
        raise RuntimeError(f"invalid existing nested subset: {output}")
    selected = {}
    for class_dir in classes:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        if len(images) < args.ipc:
            raise RuntimeError(f"{class_dir} has {len(images)} images, needs {args.ipc}")
        chosen = images[:args.ipc]
        destination = output / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            target = destination / image.name
            if not target.exists():
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)
        selected[class_dir.name] = [path.name for path in chosen]
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({**expected, "selected": selected}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**expected, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
