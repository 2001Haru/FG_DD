import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from audit_imagenette_cam_trajectory import (
    MEAN, STD, forward_resnet18, grad_cam, load_model,
)


def denormalize(tensor):
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image, cam):
    # Grad-CAM intentionally originates from an autograd graph, but rendering
    # must not retain or convert that graph to NumPy.
    cam = cam.detach()
    cam = F.interpolate(
        cam[None, None], size=image.shape[:2], mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-12)
    heat = plt.get_cmap("jet")(cam.cpu().numpy())[..., :3]
    return np.clip(0.55 * image + 0.45 * heat, 0, 1)


def parent_and_child_cams(model, images, targets, subclasses, temperature, children):
    logits, activations, _ = forward_resnet18(model, images)
    grouped = logits.view(logits.shape[0], 10, subclasses)
    parent_logits = temperature * torch.logsumexp(grouped / temperature, dim=2)
    parent_score = parent_logits.gather(1, targets[:, None]).squeeze(1)
    parent_cam = grad_cam(parent_score, activations, retain_graph=children > 0)
    child_cams, selected = [], None
    if children:
        batch = torch.arange(images.shape[0], device=images.device)
        within = grouped[batch, targets]
        selected = within.topk(children, dim=1).indices
        for rank in range(children):
            score = within.gather(1, selected[:, rank:rank + 1]).squeeze(1)
            child_cams.append(
                grad_cam(score, activations, retain_graph=rank < children - 1)
            )
    probabilities = torch.softmax(parent_logits / temperature, dim=1)
    return parent_cam, child_cams, selected, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--teacher-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=(16, 64, 100, 300))
    parser.add_argument("--temperature", type=float, default=20.0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.test_root, transform=transform)
    chosen_classes = (0, 4, 8)
    chosen_indices = [
        next(index for index, (_, target) in enumerate(dataset.samples) if target == cls)
        for cls in chosen_classes
    ]
    images, targets = zip(*(dataset[index] for index in chosen_indices))
    images = torch.stack(images).cuda()
    targets = torch.tensor(targets, dtype=torch.long, device="cuda")
    originals = [denormalize(image) for image in images]
    root = Path(args.trajectory_root) / f"tseed{args.teacher_seed}" / "models"
    for training_epoch in args.epochs:
        index = training_epoch - 1
        c1 = load_model(
            root / f"c1_tseed{args.teacher_seed}" / "checkpoints" / f"epoch_{index:03d}.pth",
            10,
        )
        c100 = load_model(
            root / f"c100_tseed{args.teacher_seed}" / "checkpoints" / f"epoch_{index:03d}.pth",
            1000,
        )
        c1_parent, _, _, c1_probability = parent_and_child_cams(
            c1, images, targets, 1, args.temperature, 0
        )
        c100_parent, child_cams, selected, c100_probability = parent_and_child_cams(
            c100, images, targets, 100, args.temperature, 5
        )
        figure, axes = plt.subplots(len(images), 8, figsize=(22, 8.2))
        for row in range(len(images)):
            panels = [
                originals[row],
                overlay(originals[row], c1_parent[row]),
                overlay(originals[row], c100_parent[row]),
                *[overlay(originals[row], cam[row]) for cam in child_cams],
            ]
            titles = [
                f"image / parent {int(targets[row])}",
                f"C1 parent\np={c1_probability[row, targets[row]].detach().item():.3f}",
                f"C100 parent\np={c100_probability[row, targets[row]].detach().item():.3f}",
                *[
                    f"C100 child top{rank + 1}\nlocal={int(selected[row, rank])}"
                    for rank in range(5)
                ],
            ]
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axes[row, column].imshow(panel)
                axes[row, column].set_title(title, fontsize=9)
                axes[row, column].axis("off")
        figure.suptitle(
            f"ImageNette paired Grad-CAM, Teacher seed {args.teacher_seed}, "
            f"epoch {training_epoch}, T={args.temperature:g}",
            fontsize=15,
        )
        figure.tight_layout()
        figure.savefig(
            output / f"paired_cam_seed{args.teacher_seed}_epoch{training_epoch:03d}.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(figure)
        del c1, c100
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
