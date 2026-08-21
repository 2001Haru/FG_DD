import argparse
import math
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models as torchvision_models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import ResNet18  # noqa: E402

MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]


def sample_square_crop(image, rng, scale=(0.08, 1.0)):
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = area * rng.uniform(*scale)
        side = int(round(math.sqrt(target_area)))
        if 0 < side <= width and side <= height:
            top = rng.randint(0, height - side)
            left = rng.randint(0, width - side)
            return top, left, side, side
    side = min(width, height)
    return (height - side) // 2, (width - side) // 2, side, side


def deterministic_crop_tensor(task):
    image, params, crop_size, normalize = task
    cropped = TF.resized_crop(
        image, *params, size=[crop_size, crop_size],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    return normalize(TF.to_tensor(cropped))


def main():
    parser = argparse.ArgumentParser("Generate balanced RDED medium patches for CIFAR hierarchy experiments")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-num-classes", type=int)
    parser.add_argument("--teacher-mapping",
                        help="fine_to_coarse hierarchy used to marginalize Teacher probabilities")
    parser.add_argument("--teacher-architecture", choices=("cvdd", "torchvision"), default="cvdd")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--normalization", choices=("cifar100", "imagenet"), default="cifar100")
    parser.add_argument("--scoring-batch-size", type=int, default=512)
    parser.add_argument("--crop-workers", type=int, default=0,
                        help="parallel deterministic CPU crop workers; 0 keeps legacy transform")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--patches-per-class", type=int, required=True)
    parser.add_argument("--candidate-images", type=int, default=100)
    parser.add_argument("--crops-per-image", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"))
    if len(dataset.classes) != args.num_classes:
        raise RuntimeError("teacher/data class count mismatch")
    by_class = [[] for _ in range(args.num_classes)]
    for path, target in dataset.samples:
        by_class[target].append(path)

    teacher_num_classes = args.teacher_num_classes or args.num_classes
    if args.teacher_architecture == "torchvision":
        model = torchvision_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, teacher_num_classes)
    else:
        model = ResNet18(teacher_num_classes)
    model.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    teacher_to_target = None
    if args.teacher_mapping:
        hierarchy = json.loads(Path(args.teacher_mapping).read_text(encoding="utf-8"))
        teacher_to_target = torch.tensor(
            [int(hierarchy["fine_to_coarse"][str(index)])
             for index in range(teacher_num_classes)],
            dtype=torch.long, device=device,
        )
    mean, std = (([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 if args.normalization == "imagenet" else (MEAN, STD))
    normalize = transforms.Normalize(mean, std)
    if args.image_size % 2:
        raise ValueError("2x2 patch initialization requires an even image size")
    crop_size = args.image_size // 2
    cropper = transforms.RandomResizedCrop(
        crop_size, ratio=(1.0, 1.0), antialias=True
    )

    crop_executor = (ThreadPoolExecutor(max_workers=args.crop_workers)
                     if args.crop_workers > 0 else None)
    for class_id, paths in enumerate(by_class):
        image_cache = {}
        class_dir = Path(args.output_dir) / f"{class_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for patch_id in range(args.patches_per_class):
            target_path = class_dir / f"class{class_id:05d}_id{patch_id:05d}.jpg"
            if target_path.is_file():
                continue
            rng = random.Random(args.seed + class_id * 10000 + patch_id)
            selected_paths = rng.sample(paths, min(args.candidate_images, len(paths)))
            candidates = []
            crop_started = time.perf_counter()
            cache_hits = 0
            selected_images = []
            for path in selected_paths:
                image = image_cache.get(path)
                if image is None:
                    with Image.open(path) as opened:
                        image = opened.convert("RGB").copy()
                    image_cache[path] = image
                else:
                    cache_hits += 1
                selected_images.append(image)
            if crop_executor is None:
                candidates = torch.stack([
                    torch.stack([
                        normalize(transforms.functional.to_tensor(cropper(image)))
                        for _ in range(args.crops_per_image)
                    ])
                    for image in selected_images
                ])
            else:
                crop_rng = random.Random(
                    args.seed + class_id * 10000 + patch_id + 1_000_000_007
                )
                tasks = [
                    (image, sample_square_crop(image, crop_rng), crop_size, normalize)
                    for image in selected_images
                    for _ in range(args.crops_per_image)
                ]
                transformed = list(crop_executor.map(deterministic_crop_tensor, tasks))
                candidates = torch.stack(transformed).reshape(
                    len(selected_images), args.crops_per_image, 3, crop_size, crop_size
                )
            flat = candidates.flatten(0, 1)
            losses = []
            crop_seconds = time.perf_counter() - crop_started
            scoring_started = time.perf_counter()
            with torch.no_grad():
                for start in range(0, flat.shape[0], args.scoring_batch_size):
                    current = flat[start:start + args.scoring_batch_size].to(device)
                    padding = crop_size // 2
                    images = F.pad(current, (padding, padding, padding, padding))
                    logits = model(images)
                    if teacher_to_target is None:
                        labels = torch.full(
                            (images.shape[0],), class_id, dtype=torch.long, device=device
                        )
                        batch_losses = F.cross_entropy(logits, labels, reduction="none")
                    else:
                        probabilities = logits.softmax(dim=1)
                        target_probabilities = torch.zeros(
                            images.shape[0], args.num_classes,
                            dtype=probabilities.dtype, device=device,
                        )
                        target_probabilities.scatter_add_(
                            1,
                            teacher_to_target.unsqueeze(0).expand(images.shape[0], -1),
                            probabilities,
                        )
                        batch_losses = -target_probabilities[:, class_id].clamp_min(1e-12).log()
                    losses.append(batch_losses.cpu())
            torch.cuda.synchronize()
            scoring_seconds = time.perf_counter() - scoring_started
            losses = torch.cat(losses).reshape(len(selected_paths), args.crops_per_image)
            best_crop_loss, best_crop_id = losses.min(dim=1)
            best_image_ids = best_crop_loss.argsort()[:4]
            patches = torch.stack([
                candidates[image_id, best_crop_id[image_id]] for image_id in best_image_ids
            ])
            canvas = torch.zeros(3, args.image_size, args.image_size)
            canvas[:, :crop_size, :crop_size] = patches[0]
            canvas[:, :crop_size, crop_size:] = patches[1]
            canvas[:, crop_size:, :crop_size] = patches[2]
            canvas[:, crop_size:, crop_size:] = patches[3]
            for channel, (channel_mean, channel_std) in enumerate(zip(mean, std)):
                canvas[channel].mul_(channel_std).add_(channel_mean).clamp_(0, 1)
            Image.fromarray((canvas.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(target_path)
            print(
                f"class={class_id} patch={patch_id} saved "
                f"crop_io={crop_seconds:.2f}s scoring={scoring_seconds:.2f}s "
                f"cache_hits={cache_hits}/{len(selected_paths)} "
                f"cache_size={len(image_cache)}",
                flush=True,
            )
    if crop_executor is not None:
        crop_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
