import argparse
import math
import os
import sys

import torch
from torch.utils.data._utils.fetch import _MapDatasetFetcher
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from relabel.utils_fkd import (  # noqa: E402
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
)


_original_fetch = _MapDatasetFetcher.fetch


def _fkd_fetch(self, possibly_batched_index):
    if not (hasattr(self.dataset, "mode") and self.dataset.mode == "fkd_load"):
        return _original_fetch(self, possibly_batched_index)
    mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(
        possibly_batched_index[0]
    )
    if self.auto_collation:
        if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
            data = self.dataset.__getitems__(possibly_batched_index)
        else:
            data = [self.dataset[idx] for idx in possibly_batched_index]
    else:
        data = self.dataset[possibly_batched_index]
    return self.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()


_MapDatasetFetcher.fetch = _fkd_fetch


def compare(left, right, path="batch"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return f"{path}: tensor metadata differs: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}"
        if torch.equal(left, right):
            return None
        if left.is_floating_point():
            delta = (left.float() - right.float()).abs().max().item()
            return f"{path}: tensor values differ, max_abs={delta:.9g}"
        return f"{path}: tensor values differ"
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: sequence lengths differ: {len(left)} vs {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatch = compare(left_item, right_item, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, float) or isinstance(right, float):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0):
            return f"{path}: values differ: {left!r} vs {right!r}"
        return None
    if left != right:
        return f"{path}: values differ: {left!r} vs {right!r}"
    return None


def make_loader(args, persistent):
    dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode="fkd_load",
        args_epoch=args.max_epoch + 1,
        args_bs=args.batch_size,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(
                size=224,
                scale=(0.08, 1.0),
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
        pin_memory=False,
        persistent_workers=persistent and args.workers > 0,
    )
    return loader


def advance_sampler(loader):
    # RandomSampler consumes exactly one randperm per complete epoch.
    list(iter(loader.sampler))


def main():
    parser = argparse.ArgumentParser("Compare released and persistent CV-DD FKD loaders")
    parser.add_argument("--syn-data-path", required=True)
    parser.add_argument("--fkd-path", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fkd-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 50, 399])
    args = parser.parse_args()
    args.epochs = sorted(set(args.epochs))
    args.max_epoch = max(args.epochs)

    released = make_loader(args, persistent=False)
    optimized = make_loader(args, persistent=True)
    previous_epoch = -1

    for epoch in args.epochs:
        for _ in range(previous_epoch + 1, epoch):
            advance_sampler(released)
            advance_sampler(optimized)

        released.dataset.set_epoch(epoch)
        optimized.dataset.set_epoch(epoch)
        released_batches = iter(released)
        optimized_batches = iter(optimized)
        compared = 0
        while True:
            try:
                released_batch = next(released_batches)
            except StopIteration:
                try:
                    next(optimized_batches)
                except StopIteration:
                    break
                raise RuntimeError(f"epoch {epoch}: optimized loader has extra batches")
            try:
                optimized_batch = next(optimized_batches)
            except StopIteration as exc:
                raise RuntimeError(f"epoch {epoch}: optimized loader ended early") from exc

            mismatch = compare(released_batch, optimized_batch)
            if mismatch:
                raise RuntimeError(f"epoch {epoch}, batch {compared}: {mismatch}")
            compared += 1

        print(f"epoch {epoch}: PASS ({compared} batches exactly equal)", flush=True)
        previous_epoch = epoch

    print("PASS: released and persistent FKD loaders are element-wise equivalent")


if __name__ == "__main__":
    main()
