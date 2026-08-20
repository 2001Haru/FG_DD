import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import ResNet18  # noqa: E402
from recover.utils_recover import BNFeatureHook, clip, denormalize, lr_cosine_policy  # noqa: E402


MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]


def output_path(root, entry):
    return Path(root) / f"new{entry['class_id']:03d}" / \
        f"class{entry['class_id']:03d}_id{entry['image_id']:03d}.jpg"


def load_inputs(entries, patch_root, device):
    normalize = transforms.Normalize(MEAN, STD)
    images = []
    for entry in entries:
        class_id, patch_id = entry["class_id"], entry["patch_id"]
        path = Path(patch_root) / "medium" / f"{class_id:05d}" / \
            f"class{class_id:05d}_id{patch_id:05d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(normalize(transforms.functional.to_tensor(Image.open(path).convert("RGB"))))
    return torch.stack(images).to(device).requires_grad_(True)


def main():
    parser = argparse.ArgumentParser("CV-DD SRe2L++ recovery adapter for explicit equal-budget batch plans")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-num-classes", type=int)
    parser.add_argument("--teacher-mapping",
                        help="fine_to_coarse hierarchy used to marginalize Teacher probabilities")
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--r-bn", type=float, default=0.01)
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--diagnostics-output",
                        help="durable JSONL recovery diagnostics (in addition to stdout)")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan["batch_size"] != 100 or plan["num_batches"] != 5:
        raise RuntimeError("this controlled experiment requires exactly five BS100 batches")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    teacher_num_classes = args.teacher_num_classes or plan["num_classes"]
    model = ResNet18(teacher_num_classes)
    model.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    hooks = [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    criterion = nn.CrossEntropyLoss()
    teacher_to_target = None
    if args.teacher_mapping:
        hierarchy = json.loads(Path(args.teacher_mapping).read_text(encoding="utf-8"))
        teacher_to_target = torch.tensor(
            [int(hierarchy["fine_to_coarse"][str(index)])
             for index in range(teacher_num_classes)],
            dtype=torch.long, device=device,
        )
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
    ])
    clip_args = argparse.Namespace(mean_norm=MEAN, std_norm=STD)

    diagnostic_handle = None
    if args.diagnostics_output:
        diagnostic_path = Path(args.diagnostics_output)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        completed_batch_exists = any(
            all(output_path(args.output_dir, entry).is_file() for entry in entries)
            for entries in plan["batches"]
        )
        diagnostic_handle = diagnostic_path.open(
            "a" if completed_batch_exists else "w", encoding="utf-8"
        )

    try:
        for batch_id, entries in enumerate(plan["batches"]):
            paths = [output_path(args.output_dir, entry) for entry in entries]
            if all(path.is_file() for path in paths):
                print(f"batch={batch_id}: complete, skipping", flush=True)
                continue
            if any(path.is_file() for path in paths):
                raise RuntimeError(f"batch={batch_id} is partially complete; archive {args.output_dir}")
            inputs = load_inputs(entries, args.patch_root, device)
            targets = torch.tensor([entry["class_id"] for entry in entries], device=device)
            optimizer = torch.optim.Adam([inputs], lr=args.lr, betas=(0.5, 0.9), eps=1e-8)
            scheduler = lr_cosine_policy(args.lr, 0, args.iterations)
            started = time.time()

            for iteration in range(args.iterations):
                scheduler(optimizer, iteration, iteration)
                images = augmentation(inputs)
                images = torch.roll(images, (random.randint(0, 4), random.randint(0, 4)), (2, 3))
                optimizer.zero_grad()
                logits = model(images)
                if teacher_to_target is None:
                    ce = criterion(logits, targets)
                else:
                    probabilities = logits.softmax(dim=1)
                    target_probabilities = torch.zeros(
                        images.shape[0], plan["num_classes"],
                        dtype=probabilities.dtype, device=device,
                    )
                    target_probabilities.scatter_add_(
                        1,
                        teacher_to_target.unsqueeze(0).expand(images.shape[0], -1),
                        probabilities,
                    )
                    ce = F.nll_loss(target_probabilities.clamp_min(1e-12).log(), targets)
                scales = [args.first_bn_multiplier] + [1.0] * (len(hooks) - 1)
                bn_raw = sum(hook.r_feature * scale for hook, scale in zip(hooks, scales))
                bn_weighted = args.r_bn * bn_raw
                loss = ce + bn_weighted
                diagnose = iteration % 100 == 0 or iteration == args.iterations - 1
                if diagnose:
                    ce_grad = torch.autograd.grad(ce, inputs, retain_graph=True)[0]
                    bn_grad = torch.autograd.grad(bn_weighted, inputs, retain_graph=True)[0]
                    before = inputs.detach().clone()
                    image_rms = before.float().square().mean().sqrt()
                loss.backward()
                total_grad_rms = inputs.grad.detach().float().square().mean().sqrt() if diagnose else None
                optimizer.step()
                inputs.data = clip(inputs.data, clip_args)

                if diagnose:
                    update_rms = (inputs.detach() - before).float().square().mean().sqrt()
                    ce_grad_rms = ce_grad.float().square().mean().sqrt()
                    bn_grad_rms = bn_grad.float().square().mean().sqrt()
                    eps = 1e-12
                    record = {
                        "plan": plan["name"], "seed": args.seed, "batch_id": batch_id,
                        "iteration": iteration, "iterations": args.iterations,
                        "lr": optimizer.param_groups[0]["lr"], "r_bn": args.r_bn,
                        "ce": ce.item(), "bn_raw": bn_raw.item(), "bn_weighted": bn_weighted.item(),
                        "bn_to_ce_loss_ratio": bn_weighted.item() / max(abs(ce.item()), eps),
                        "ce_grad_rms": ce_grad_rms.item(), "bn_grad_rms": bn_grad_rms.item(),
                        "bn_to_ce_grad_ratio": bn_grad_rms.item() / max(ce_grad_rms.item(), eps),
                        "total_grad_rms": total_grad_rms.item(), "image_rms": image_rms.item(),
                        "update_rms": update_rms.item(),
                        "relative_update_rms": update_rms.item() / max(image_rms.item(), eps),
                    }
                    serialized = json.dumps(record, sort_keys=True)
                    print("RECOVERY_DIAG " + serialized, flush=True)
                    if diagnostic_handle is not None:
                        diagnostic_handle.write(serialized + "\n")
                        diagnostic_handle.flush()
                    print(f"batch={batch_id} iter={iteration} loss={loss.item():.6f} "
                          f"elapsed={time.time()-started:.1f}s", flush=True)

            saved = denormalize(inputs.detach().clone(), clip_args).cpu()
            for image, path in zip(saved, paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)
            del inputs, optimizer
            torch.cuda.empty_cache()
    finally:
        if diagnostic_handle is not None:
            diagnostic_handle.close()
        for hook in hooks:
            hook.close()


if __name__ == "__main__":
    main()
