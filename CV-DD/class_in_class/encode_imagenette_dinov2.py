import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tree_fingerprint(root, splits):
    digest = hashlib.sha256()
    count = 0
    for split in splits:
        split_root = root / split
        for path in sorted(split_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
            count += 1
    return digest.hexdigest(), count


def model_fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def image_transform(processor):
    size = getattr(processor, "size", {})
    crop_size = getattr(processor, "crop_size", {})
    resize = size.get("shortest_edge", 256) if isinstance(size, dict) else int(size)
    if isinstance(crop_size, dict):
        crop = (int(crop_size.get("height", 224)), int(crop_size.get("width", 224)))
    else:
        crop = int(crop_size)
    mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.inference_mode()
def encode_split(model, dataset, loader, device, use_amp, split):
    features, targets = [], []
    started = time.time()
    seen = 0
    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(pixel_values=images)
            embedding = output.last_hidden_state[:, 0]
        embedding = F.normalize(embedding.float(), dim=1).cpu()
        features.append(embedding)
        targets.append(labels.long().cpu())
        seen += images.shape[0]
        if batch_index % 10 == 0 or seen == len(dataset):
            print(
                f"DINO encode split={split} images={seen}/{len(dataset)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return torch.cat(features), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser("Encode official ImageNette with local DINOv2")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("transformers is required for local DINOv2 encoding") from error

    data_root = Path(args.data_root).resolve()
    model_root = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    for split in args.splits:
        if not (data_root / split).is_dir():
            raise FileNotFoundError(data_root / split)
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)

    source_hash, source_images = tree_fingerprint(data_root, args.splits)
    model_hash = model_fingerprint(model_root)
    if output.is_file() and not args.force:
        cached = torch.load(output, map_location="cpu", weights_only=False)
        metadata = cached.get("metadata", {})
        if (
            metadata.get("source_fingerprint") == source_hash
            and metadata.get("model_fingerprint") == model_hash
            and metadata.get("splits") == list(args.splits)
        ):
            print(f"Valid DINO feature cache already exists: {output}")
            return
        raise RuntimeError(
            f"stale DINO feature cache: {output}; pass --force to replace it"
        )

    processor = AutoImageProcessor.from_pretrained(
        str(model_root), local_files_only=True
    )
    model = AutoModel.from_pretrained(str(model_root), local_files_only=True)
    device = torch.device(args.device)
    model.to(device).eval()
    transform = image_transform(processor)

    encoded = {}
    class_names = None
    for split in args.splits:
        dataset = datasets.ImageFolder(str(data_root / split), transform=transform)
        if len(dataset.classes) != 10:
            raise RuntimeError(f"{split} has {len(dataset.classes)} classes, expected 10")
        if class_names is None:
            class_names = dataset.classes
        elif dataset.classes != class_names:
            raise RuntimeError("ImageNette train/test class order mismatch")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        split_features, split_targets = encode_split(
            model, dataset, loader, device, not args.no_fp16, split
        )
        relative_paths = [
            Path(path).relative_to(data_root).as_posix()
            for path, _ in dataset.samples
        ]
        encoded[split] = {
            "features": split_features.contiguous(),
            "targets": split_targets.contiguous(),
            "relative_paths": relative_paths,
        }

    payload = {
        "metadata": {
            "schema_version": 1,
            "feature_type": "dinov2_cls_l2_normalized",
            "data_root": str(data_root),
            "model_dir": str(model_root),
            "source_fingerprint": source_hash,
            "source_images": source_images,
            "model_fingerprint": model_hash,
            "splits": list(args.splits),
            "class_names": class_names,
            "feature_dimension": int(next(iter(encoded.values()))["features"].shape[1]),
        },
        "splits": encoded,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    metadata_output = output.with_suffix(output.suffix + ".metadata.json")
    metadata_output.write_text(
        json.dumps(payload["metadata"], indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved DINO features: images={source_images}, dim={payload['metadata']['feature_dimension']}, "
        f"output={output}", flush=True,
    )


if __name__ == "__main__":
    main()
