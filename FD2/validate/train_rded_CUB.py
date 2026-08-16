import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR


CUB_MEAN = [0.4857, 0.4994, 0.4326]
CUB_STD = [0.2260, 0.2215, 0.2595]


class SelectedImageFolder(torch.utils.data.Dataset):
    """RDED's per-class random IPC selection, with optional in-memory PIL images."""

    def __init__(self, root, ipc, transform, seed=42, memory=True):
        self.root = Path(root)
        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader
        self.samples = []
        for class_id in range(200):
            class_dir = self.root / f"{class_id:05d}"
            if not class_dir.is_dir():
                raise FileNotFoundError(f"missing class directory: {class_dir}")
            files = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if len(files) < ipc:
                raise RuntimeError(
                    f"class {class_dir.name} has {len(files)} images, but IPC={ipc}"
                )
            # IPC5 generation writes classXXXXX_id00000...id00004. For an
            # IPC3 evaluation, using the first three is equivalent to having
            # generated RDED directly with patch_num=3; shuffling a five-image
            # superset would silently evaluate a different distilled set.
            for path in files[:ipc]:
                image = self.loader(path) if memory else None
                self.samples.append((path, image, class_id))

        self.memory = memory

    def __getitem__(self, index):
        path, image, target = self.samples[index]
        if not self.memory:
            image = self.loader(path)
        return self.transform(image), target

    def __len__(self):
        return len(self.samples)


class ShufflePatches(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    @staticmethod
    def _shuffle_axis(image, factor):
        patches = list(torch.tensor_split(image, factor, dim=-1))
        random.shuffle(patches)
        return torch.cat(patches, dim=-1)

    def forward(self, image):
        image = self._shuffle_axis(image, self.factor)
        image = image.transpose(1, 2)
        image = self._shuffle_axis(image, self.factor)
        return image.transpose(1, 2)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, count):
        self.sum += value * count
        self.count += count

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    prediction = output.topk(maxk, dim=1).indices.t()
    correct = prediction.eq(target.view(1, -1))
    return [correct[:k].reshape(-1).float().sum().mul_(100.0 / target.numel()) for k in topk]


def parameter_groups(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if "weight" in name and parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [dict(params=decay), dict(params=no_decay, weight_decay=0.0)]


def load_teacher(model_pool_dir, device):
    model_pool_dir = Path(model_pool_dir)
    candidates = [
        # Use the same observer checkpoint that generated the RDED patches.
        model_pool_dir / "ResNet18.pth",
        model_pool_dir / "ResNet18_M8_5e-1cal.pth",
        model_pool_dir / "ResNet18_M8_5e-01cal.pth",
    ]
    checkpoint_path = next((path for path in candidates if path.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError("teacher checkpoint not found in " + str(model_pool_dir))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ResNet18" in checkpoint:
        state_dict = checkpoint["ResNet18"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    teacher = torchvision.models.resnet18(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 200)
    teacher.load_state_dict(state_dict)
    teacher.to(device).eval()
    teacher.requires_grad_(False)
    print(f"Teacher: {checkpoint_path}")
    return teacher


def load_student(model_name, device):
    # RDED's official post-evaluation protocol trains students from scratch.
    if model_name == "ResNet18":
        student = torchvision.models.resnet18(weights=None)
    elif model_name == "ResNet50":
        student = torchvision.models.resnet50(weights=None)
    else:
        raise ValueError(f"unsupported student: {model_name}")
    student.fc = nn.Linear(student.fc.in_features, 200)
    return student.to(device)


def cutmix(images):
    permutation = torch.randperm(images.shape[0], device=images.device)
    lam = np.random.beta(1.0, 1.0)
    cut_ratio = math.sqrt(1.0 - lam)
    height, width = images.shape[-2:]
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y, center_x = np.random.randint(height), np.random.randint(width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    return images


def parse_args():
    parser = argparse.ArgumentParser("RDED CUB post-evaluation")
    parser.add_argument("--syn-data-path", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--model-pool-dir", required=True)
    parser.add_argument("--model", choices=["ResNet18", "ResNet50"], required=True)
    parser.add_argument("--ipc", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=20.0)
    return parser.parse_args()


def build_loaders(args):
    normalize = transforms.Normalize(CUB_MEAN, CUB_STD)
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ShufflePatches(2),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_set = SelectedImageFolder(
        args.syn_data_path, args.ipc, train_transform, args.seed, memory=True
    )
    val_set = torchvision.datasets.ImageFolder(args.val_dir, transform=val_transform)
    if len(val_set.classes) != 200:
        raise RuntimeError(f"validation dataset has {len(val_set.classes)} classes, expected 200")

    generator = torch.Generator().manual_seed(args.seed)
    common = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, **common
    )
    return train_loader, val_loader


def train_one_epoch(student, teacher, loader, optimizer, temperature, device):
    loss_meter = AverageMeter()
    student.train()
    teacher.eval()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        images = cutmix(images)
        with torch.no_grad():
            soft_targets = F.softmax(teacher(images) / temperature, dim=1)
        output = F.log_softmax(student(images) / temperature, dim=1)
        loss = F.kl_div(output, soft_targets, reduction="batchmean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
    return loss_meter.avg


@torch.no_grad()
def validate(student, loader, device):
    top1_meter, top5_meter = AverageMeter(), AverageMeter()
    student.eval()
    start = time.time()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = student(images)
        top1, top5 = accuracy(output, targets)
        top1_meter.update(top1.item(), images.shape[0])
        top5_meter.update(top5.item(), images.shape[0])
    return top1_meter.avg, top5_meter.avg, time.time() - start


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RDED evaluation requires CUDA")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(args)
    teacher = load_teacher(args.model_pool_dir, device)
    student = load_student(args.model, device)
    optimizer = torch.optim.AdamW(
        parameter_groups(student), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = LambdaLR(
        optimizer,
        lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs / 2.0)),
    )

    print(
        f"RDED setting: student={args.model}, IPC={args.ipc}, epochs={args.epochs}, "
        f"batch={args.batch_size}, random_init=True, T={args.temperature}"
    )
    best_top1, best_epoch = 0.0, -1
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(
            student, teacher, train_loader, optimizer, args.temperature, device
        )
        if epoch >= int(args.epochs * 0.8) and (epoch % 10 == 9 or epoch == args.epochs - 1):
            top1, top5, val_time = validate(student, val_loader, device)
            if top1 > best_top1:
                best_top1, best_epoch = top1, epoch
            print(
                f"RDED Test epoch {epoch}: loss={train_loss:.6f}, Top1={top1:.2f}, "
                f"Top5={top5:.2f}, train_time={time.time() - start - val_time:.2f}, "
                f"val_time={val_time:.2f}, best={best_top1:.2f}@{best_epoch}"
            )
        elif epoch % 25 == 0:
            print(
                f"RDED Train epoch {epoch}: loss={train_loss:.6f}, "
                f"time={time.time() - start:.2f}, lr={scheduler.get_last_lr()[0]:.6g}"
            )
        scheduler.step()
    print(f"RDED {args.model} finished: best Top1={best_top1:.2f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
