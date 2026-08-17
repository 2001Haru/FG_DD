import argparse
import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from utils_fkd import (
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
    mix_aug,
)


def main():
    parser = argparse.ArgumentParser("Restored backbone-only SRe2L++ BSSL on CUB")
    parser.add_argument("--syn-data-path", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--fkd-path", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fkd-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-scale", type=float, default=0.08)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--mix-type", choices=["cutmix", "mixup", "none"], default="cutmix")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.fkd_path, exist_ok=True)

    args.mode = "fkd_save"
    args.mix_type = None if args.mix_type == "none" else args.mix_type
    args.cutmix = 1.0
    args.mixup = 0.8

    dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(
                size=224,
                scale=(args.min_scale, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4857, 0.4994, 0.4326],
                std=[0.2260, 0.2215, 0.2595],
            ),
        ]),
    )
    generator = torch.Generator().manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True), strict=True)
    model.cuda().train()
    for parameter in model.parameters():
        parameter.requires_grad = False

    try:
        with torch.no_grad():
            for epoch in tqdm(range(args.epochs)):
                epoch_dir = os.path.join(args.fkd_path, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                for batch_idx, (images, targets, flip_status, coords_status) in enumerate(loader):
                    images = images.cuda(non_blocking=True)
                    split = images.shape[0] // 2
                    _, mix_index, mix_lam, mix_bbox = mix_aug(images, args)

                    logits = torch.cat([model(images[:split]), model(images[split:])], dim=0)
                    if args.use_fp16:
                        logits = logits.half()
                    config = [
                        coords_status,
                        flip_status,
                        mix_index,
                        mix_lam,
                        mix_bbox,
                        logits.cpu(),
                    ]
                    torch.save(config, os.path.join(epoch_dir, f"batch_{batch_idx}.tar"))
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
