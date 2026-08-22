import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser("Validate the exact CV-DD patch tree contract")
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--patches-per-class", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.patch_dir) / args.difficulty
    errors = []
    expected = set()
    for class_id in range(args.classes):
        class_name = f"{class_id:05d}"
        for patch_id in range(args.patches_per_class):
            path = root / class_name / f"class{class_name}_id{patch_id:05d}.jpg"
            expected.add(path.resolve())
            if not path.is_file():
                errors.append(f"missing: {path}")
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"OpenCV cannot decode: {path}")
            elif image.shape[:2] != (args.image_size, args.image_size):
                errors.append(
                    f"wrong size {image.shape[:2]} (expected "
                    f"{args.image_size}x{args.image_size}): {path}"
                )

    actual = {
        path.resolve() for path in root.rglob("*.jpg") if path.is_file()
    } if root.is_dir() else set()
    for path in sorted(actual - expected):
        errors.append(f"unexpected: {path}")
    if errors:
        preview = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... {len(errors) - 20} more errors"
        raise RuntimeError(
            f"invalid CV-DD patch tree ({len(errors)} errors):\n{preview}{suffix}"
        )
    print(
        f"CV-DD patch tree valid: root={root} files={len(expected)} "
        f"size={args.image_size}x{args.image_size}"
    )


if __name__ == "__main__":
    main()
