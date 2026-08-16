import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms
from torch.utils.data import BatchSampler, RandomSampler
from torchvision.transforms import InterpolationMode

from models.utils_models import load_model
from relabel.utils_fkd import (
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
    mix_aug,
)


CUB_MEAN = [0.4857, 0.4994, 0.4326]
CUB_STD = [0.2260, 0.2215, 0.2595]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit synthetic images and exact FKD crop/CutMix replay."
    )
    parser.add_argument("--syn-data-path", required=True)
    parser.add_argument("--fkd-path", required=True)
    parser.add_argument("--model-pool-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 50, 399])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_teacher(model_pool_dir, device):
    checkpoint_path = Path(model_pool_dir) / "ResNet18_M8_5e-01cal.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    errors = []
    for source in ("CVDD", "torchvision"):
        model = load_model("ResNet18", 200, source, False, False)
        try:
            model.load_state_dict(checkpoint["ResNet18"])
        except RuntimeError as exc:
            errors.append(f"{source}: {exc}")
            continue
        print(f"Teacher architecture source: {source}")
        return model.to(device)
    raise RuntimeError("teacher checkpoint matches neither architecture:\n" + "\n".join(errors))


def build_dataset(args):
    return ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode="fkd_load",
        args_epoch=max(args.epochs) + 1,
        args_bs=args.batch_size,
        root=args.syn_data_path,
        transform=ComposeWithCoords(
            transforms=[
                RandomResizedCropWithCoords(
                    size=224,
                    scale=(0.08, 1.0),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                RandomHorizontalFlipWithRes(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CUB_MEAN, std=CUB_STD),
            ]
        ),
    )


@torch.no_grad()
def synthetic_teacher_accuracy(dataset, teacher, device):
    # The recovered files are already 224x224. This test deliberately avoids
    # random crop and CutMix and checks their basic class semantics.
    plain = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=CUB_MEAN, std=CUB_STD)]
    )
    teacher.eval()
    correct = 0
    total = 0
    for start in range(0, len(dataset.samples), 64):
        chunk = dataset.samples[start : start + 64]
        images = torch.stack([plain(dataset.loader(path)) for path, _ in chunk]).to(device)
        targets = torch.tensor([target for _, target in chunk], device=device)
        correct += teacher(images).argmax(1).eq(targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / total


def epoch_first_batches(dataset, batch_size, seed, final_epoch):
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    result = {}
    for epoch in range(final_epoch + 1):
        first_batch = None
        for batch_idx, indices in enumerate(batch_sampler):
            if batch_idx == 0:
                first_batch = indices
        result[epoch] = first_batch
    return result


@torch.no_grad()
def audit_epoch(dataset, teacher, device, epoch, indices):
    dataset.set_epoch(epoch)
    mix_index, mix_lam, mix_bbox, soft_label = dataset.load_batch_config(indices[0])
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    targets = torch.tensor([sample[1] for sample in samples], device=device)

    mix_args = SimpleNamespace(mode="fkd_load", mix_type="cutmix")
    images, replay_index, _, _ = mix_aug(
        images, mix_args, rand_index=mix_index, lam=mix_lam, bbox=mix_bbox
    )

    # Relabel uses train mode and two half-batch forwards. Reproduce that
    # behavior exactly; the saved tensor is [CAL/backbone, batch, class].
    teacher.train()
    split = images.shape[0] // 2
    replay_logits = torch.cat([teacher(images[:split]), teacher(images[split:])], dim=0)
    saved_logits = soft_label[1].to(device=device, dtype=torch.float32)

    delta = replay_logits - saved_logits
    pred = saved_logits.argmax(1)
    mixed_targets = targets[replay_index.to(device)]
    hard_match = pred.eq(targets)
    either_match = hard_match | pred.eq(mixed_targets)
    cosine = torch.nn.functional.cosine_similarity(replay_logits, saved_logits, dim=1).mean()
    return {
        "mae": delta.abs().mean().item(),
        "max": delta.abs().max().item(),
        "cos": cosine.item(),
        "argmax": replay_logits.argmax(1).eq(pred).float().mean().item() * 100,
        "hard": hard_match.float().mean().item() * 100,
        "either": either_match.float().mean().item() * 100,
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = build_dataset(args)
    if len(dataset.classes) != 200:
        raise RuntimeError(f"synthetic dataset has {len(dataset.classes)} classes, expected 200")
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes))
    print(
        f"Synthetic set: {len(dataset)} images, {len(dataset.classes)} classes, "
        f"per-class min/max={counts.min().item()}/{counts.max().item()}"
    )

    teacher = load_teacher(args.model_pool_dir, args.device)
    plain_accuracy = synthetic_teacher_accuracy(dataset, teacher, args.device)
    print(f"Teacher Top1 on full recovered images: {plain_accuracy:.2f}%")

    selected_epochs = sorted(set(args.epochs))
    first_batches = epoch_first_batches(
        dataset, args.batch_size, args.seed, max(selected_epochs)
    )
    print("FKD replay (batch_0 at each selected epoch):")
    for epoch in selected_epochs:
        metrics = audit_epoch(dataset, teacher, args.device, epoch, first_batches[epoch])
        print(
            f"  epoch {epoch:3d}: MAE={metrics['mae']:.6f}, "
            f"max={metrics['max']:.6f}, cosine={metrics['cos']:.6f}, "
            f"argmax_replay={metrics['argmax']:.1f}%, "
            f"saved_vs_hard={metrics['hard']:.1f}%, "
            f"saved_vs_either_cutmix_class={metrics['either']:.1f}%"
        )

    print("Interpretation:")
    print("  - low teacher Top1 => recovered images / class mapping are the primary problem")
    print("  - replay MAE clearly above fp16 rounding (~0.01) or low cosine => FKD alignment is broken")
    print("  - both checks healthy => investigate the published student initialization/evaluation setting")


if __name__ == "__main__":
    main()
