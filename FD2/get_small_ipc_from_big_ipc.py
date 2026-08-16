import argparse
import shutil
from pathlib import Path


def sample_images(src_dir, tgt_dir, tgt_ipc):
    src_dir = Path(src_dir)
    tgt_dir = Path(tgt_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {src_dir}")

    class_dirs = sorted(path for path in src_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"no class directories found in {src_dir}")

    copied = 0
    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ('.jpg', '.png', '.jpeg')
        )
        if len(images) < tgt_ipc:
            raise ValueError(
                f"class {class_dir.name} contains {len(images)} images; need {tgt_ipc}"
            )

        target_class_dir = tgt_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        existing_images = [
            path for path in target_class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ('.jpg', '.png', '.jpeg')
        ]
        selected_names = {path.name for path in images[:tgt_ipc]}
        unexpected_names = sorted(path.name for path in existing_images if path.name not in selected_names)
        if unexpected_names:
            raise ValueError(
                f"target class {class_dir.name} contains unexpected images: {unexpected_names}"
            )

        for source_path in images[:tgt_ipc]:
            shutil.copy2(source_path, target_class_dir / source_path.name)
            copied += 1

    final_count = sum(
        1 for path in tgt_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in ('.jpg', '.png', '.jpeg')
    )
    expected_count = len(class_dirs) * tgt_ipc
    if final_count != expected_count:
        raise ValueError(f"target contains {final_count} images; expected {expected_count}")
    print(f"IPC{tgt_ipc} dataset ready: {copied} copied, {final_count} total in {tgt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract a smaller IPC dataset from an IPC5 dataset')
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--target-ipc', required=True, type=int)
    args = parser.parse_args()
    if args.target_ipc <= 0:
        parser.error('--target-ipc must be positive')
    return args


if __name__ == '__main__':
    cli_args = parse_args()
    sample_images(cli_args.source_dir, cli_args.target_dir, cli_args.target_ipc)
