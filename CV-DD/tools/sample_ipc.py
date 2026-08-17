import argparse
import os
import shutil


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser("Sample the first N images per class")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--classes", required=True, type=int)
    args = parser.parse_args()

    class_dirs = sorted(
        entry for entry in os.listdir(args.source)
        if os.path.isdir(os.path.join(args.source, entry))
    )
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"found {len(class_dirs)} classes, expected {args.classes}")

    copied = 0
    for class_name in class_dirs:
        source_class = os.path.join(args.source, class_name)
        images = sorted(
            name for name in os.listdir(source_class)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        if len(images) < args.ipc:
            raise RuntimeError(f"{class_name} has {len(images)} images, expected at least {args.ipc}")
        target_class = os.path.join(args.target, class_name)
        os.makedirs(target_class, exist_ok=True)
        for name in images[:args.ipc]:
            shutil.copy2(os.path.join(source_class, name), os.path.join(target_class, name))
            copied += 1

    print(f"Prepared IPC{args.ipc}: {copied} images in {args.target}")


if __name__ == "__main__":
    main()
