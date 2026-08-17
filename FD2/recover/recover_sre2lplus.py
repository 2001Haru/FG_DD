import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

MEAN = [0.4857, 0.4994, 0.4326]
STD = [0.2260, 0.2215, 0.2595]


class BNFeatureHook:
    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        activation = inputs[0]
        channels = activation.shape[1]
        mean = activation.mean([0, 2, 3])
        var = activation.permute(1, 0, 2, 3).contiguous().reshape(channels, -1).var(1, unbiased=False)
        self.r_feature = torch.norm(module.running_var.detach() - var, 2) + \
            torch.norm(module.running_mean.detach() - mean, 2)

    def close(self):
        self.hook.remove()


def clip(image_tensor):
    for channel, (mean, std) in enumerate(zip(MEAN, STD)):
        image_tensor[:, channel].clamp_(-mean / std, (1.0 - mean) / std)
    return image_tensor


def denormalize(image_tensor):
    for channel, (mean, std) in enumerate(zip(MEAN, STD)):
        image_tensor[:, channel].mul_(std).add_(mean).clamp_(0.0, 1.0)
    return image_tensor


def set_cosine_lr(optimizer, base_lr, iteration, iterations):
    lr = 0.5 * (1.0 + np.cos(np.pi * iteration / iterations)) * base_lr
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = lr


def output_path(root, class_id, ipc_id):
    return os.path.join(root, f"new{class_id:03d}", f"class{class_id:03d}_id{ipc_id:03d}.jpg")


def patch_path(root, class_id, ipc_id):
    return os.path.join(root, f"{class_id:05d}", f"class{class_id:05d}_id{ipc_id:05d}.jpg")


def load_patch_batch(root, class_ids, ipc_id, device):
    normalize = transforms.Normalize(MEAN, STD)
    tensors = []
    for class_id in class_ids:
        path = patch_path(root, class_id, ipc_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing deterministic IPC patch: {path}")
        image = Image.open(path).convert("RGB")
        tensors.append(normalize(transforms.functional.to_tensor(image)))
    return torch.stack(tensors).to(device).requires_grad_(True)


def save_batch(root, images, class_ids, ipc_id):
    images = denormalize(images.detach().clone())
    for image, class_id in zip(images, class_ids):
        path = output_path(root, class_id, ipc_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        array = image.cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((array * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser("Restored single-teacher SRe2L++ recovery on CUB")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--patch-dir", required=True,
                        help="directory containing 00000/class00000_id00000.jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-start", type=int, default=0)
    parser.add_argument("--class-end", type=int, default=200)
    parser.add_argument("--class-batch", type=int, default=100)
    parser.add_argument("--ipc-start", type=int, default=0)
    parser.add_argument("--ipc-end", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r-bn", type=float, default=1e-3)
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.class_start < args.class_end <= 200:
        raise ValueError("invalid CUB class range")

    seed = args.seed + args.class_start
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    state_dict = torch.load(args.teacher, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    bn_hooks = [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    criterion = nn.CrossEntropyLoss()
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])
    try:
        for ipc_id in range(args.ipc_start, args.ipc_end):
            for start in range(args.class_start, args.class_end, args.class_batch):
                end = min(start + args.class_batch, args.class_end)
                class_ids = list(range(start, end))
                if args.skip_completed and all(
                    os.path.isfile(output_path(args.output_dir, class_id, ipc_id))
                    for class_id in class_ids
                ):
                    print(f"ipc={ipc_id} classes=[{start},{end}): complete, skipping", flush=True)
                    continue

                images = load_patch_batch(args.patch_dir, class_ids, ipc_id, device)
                targets = torch.tensor(class_ids, dtype=torch.long, device=device)
                optimizer = optim.Adam([images], lr=args.lr, betas=(0.5, 0.9), eps=1e-8)

                for iteration in range(args.iterations):
                    set_cosine_lr(optimizer, args.lr, iteration, args.iterations)
                    augmented = augmentation(images)
                    off1 = random.randint(0, args.jitter)
                    off2 = random.randint(0, args.jitter)
                    augmented = torch.roll(augmented, shifts=(off1, off2), dims=(2, 3))

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(augmented)
                    ce = criterion(logits, targets)
                    scales = [args.first_bn_multiplier] + [1.0] * (len(bn_hooks) - 1)
                    bn_loss = sum(hook.r_feature * scale for hook, scale in zip(bn_hooks, scales))
                    loss = ce + args.r_bn * bn_loss
                    loss.backward()
                    optimizer.step()
                    images.data = clip(images.data)

                    if iteration % 100 == 0 or iteration == args.iterations - 1:
                        print(
                            f"ipc={ipc_id} classes=[{start},{end}) iter={iteration}/{args.iterations} "
                            f"loss={loss.item():.6f} ce={ce.item():.6f} bn={bn_loss.item():.6f}",
                            flush=True,
                        )

                save_batch(args.output_dir, images, class_ids, ipc_id)
                del images, optimizer
                torch.cuda.empty_cache()
    finally:
        for hook in bn_hooks:
            hook.close()


if __name__ == "__main__":
    main()
