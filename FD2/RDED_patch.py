import argparse
import os
import time
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
import torch.nn.functional as F
from models import *
from models.utils_models import load_model


class SimpleImageFolder(Dataset):
    def __init__(self, root, ipc, mode='train', memory=False, transform=None,
                 class_start=0, class_end=None):
        self.root = os.path.join(root, mode)
        self.ipc = ipc
        self.memory = memory
        self.transform = transform
        self.loader = default_loader
        classes = sorted([cls for cls in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, cls))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.class_start = class_start
        self.class_end = len(classes) if class_end is None else class_end
        if not 0 <= self.class_start < self.class_end <= len(classes):
            raise ValueError(
                f"Invalid class range [{self.class_start}, {self.class_end}) "
                f"for a dataset with {len(classes)} classes"
            )
        self.image_paths = [] 
        self.targets = []  
        self.samples = []  
        self.class_sample_indices = {}
        self._load_images()

    def _load_images(self):
        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_idx < self.class_start or cls_idx >= self.class_end:
                continue
            cls_dir = os.path.join(self.root, cls_name)
            imgs_name = sorted([img_name for img_name in os.listdir(cls_dir)
                                if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if len(imgs_name) < self.ipc:
                raise ValueError(
                    f"Class {cls_name} only contains {len(imgs_name)} images, "
                    f"but ipc={self.ipc} was requested"
                )
            first_index = len(self.targets)
            for img_name in imgs_name[:self.ipc]:
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.targets.append(cls_idx)
                if self.memory:
                    self.samples.append(self.loader(img_path))
            self.class_sample_indices[cls_idx] = list(range(first_index, first_index + self.ipc))

    def __getitem__(self, index):
        if self.memory:
            img = self.samples[index]
        else:
            img = self.loader(self.image_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.targets)


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=64, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(self.size // self.factor, ratio=(1, 1), antialias=True, )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


def pad(input_tensor, target_height, target_width=None):
    """
    Pad input tensor(shape=[batch_size, C, H, W]) to padded_tensor(shape=[batch_size, C, target_height, target_width])
    Args:
        input_tensor: shape=[batch_size, C, H, W]
        target_height: target height
        target_width: target width
    Returns: padded_tensor(shape=[batch_size, C, target_height, target_width]
    """
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)  # target_height-H
    horizontal_padding = target_width - input_tensor.size(3)  # target_width-W

    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding
    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding

    padded_tensor = F.pad(input_tensor, (left_padding, right_padding, top_padding, bottom_padding))

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)  # ipc * num_crop
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i: min(i + batch_size, total_samples)]
            output = model(batch_data)
            all_outputs.append(output)
    final_output = torch.cat(all_outputs, dim=0)
    return final_output


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def selector(best_crop_num, model, images, labels, size, device="cuda", forward_batch_size=256):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        s = images.shape  # [ipc, num_crop, 3, H, W]
        if best_crop_num > s[0]:
            raise ValueError(f"best_crop_num({best_crop_num}) can't be greater than ipc")
        images = images.permute(1, 0, 2, 3, 4)  # [num_crop, ipc, 3, H, W]
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])  # [num_crop * ipc, 3, H, W]
        labels = labels.repeat(s[1]).to(device)  # [ipc * num_crop]

        # A100-class GPUs can process all crop candidates in a few large
        # batches. The original implementation used one batch per crop
        # (batch_size=ipc), which left the GPU severely under-utilized.
        preds = batched_forward(model, pad(images, size), batch_size=forward_batch_size)

        # dist = cross_entropy(preds, labels)  # [num_crop * ipc]
        dist = F.cross_entropy(preds, labels, reduction='none')  # [num_crop * ipc]

        dist = dist.reshape(s[1], s[0])  # [num_crop, ipc]

        index = torch.argmin(dist, 0)  # [ipc]

        dist = dist[index, torch.arange(s[0])]  # [ipc]

        images = images.reshape(s[1], s[0], s[2], s[3], s[4])
        images = images[index, torch.arange(s[0])]  # [ipc, 3, H, W]

    indices = torch.argsort(dist, descending=False)[:best_crop_num]
    return images[indices].detach()


def mix_images(input_img, out_size, factor, mixed_img_num):
    patch_size = out_size // factor
    remained = out_size % factor
    k = 0
    # Keep composition on the input device and perform only one device-to-CPU
    # transfer when the completed image is saved.
    mixed_images = input_img.new_zeros((mixed_img_num, 3, out_size, out_size), requires_grad=False)
    h_loc = 0
    for i in range(factor):
        h_r = patch_size + 1 if i < remained else patch_size
        w_loc = 0
        for j in range(factor):
            w_r = patch_size + 1 if j < remained else patch_size
            img_part = F.interpolate(input_img.data[k * mixed_img_num: (k + 1) * mixed_img_num], size=(h_r, w_r))
            mixed_images.data[0:mixed_img_num, :, h_loc: h_loc + h_r, w_loc: w_loc + w_r,] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images


def image_output_path(root, class_id, img_id):
    dir_path = os.path.join(root, "{:05d}".format(class_id))
    return os.path.join(dir_path, "class{:05d}_id{:05d}.jpg".format(class_id, img_id))


def save_images(root, images, class_id, img_id):
    place_to_store = image_output_path(root, class_id, img_id)
    dir_path = os.path.dirname(place_to_store)
    os.makedirs(dir_path, exist_ok=True)
    image_np = images[0].data.cpu().numpy().transpose((1, 2, 0))
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    temporary_path = place_to_store + f".tmp.{os.getpid()}"
    pil_image.save(temporary_path, format="JPEG")
    os.replace(temporary_path, place_to_store)


def make_patch(model_name, ckpt_path, ncls, src_dir, ipc, mean_norm, std_norm, patch_num, num_crop, imsize,
               save_dir, class_start=0, class_end=None, patch_start=0, patch_end=None, workers=8,
               forward_batch_size=256, device="cuda", model_source="auto", memory=True, overwrite=False):
    state_dict = torch.load(ckpt_path, weights_only=True)
    model = None
    if model_source == "auto":
        try:
            model = load_model(model_name, ncls, "CVDD", False, False)
            model.load_state_dict(state_dict)
        except RuntimeError:
            print(f"CVDD's {model_name} can't match ckpt, next try torchvision")
            model = load_model(model_name, ncls, "torchvision", False, False)
            model.load_state_dict(state_dict)
            print(f"torchvision's {model_name} can match ckpt")
        else:
            print(f"CVDD's {model_name} can match ckpt")
    else:
        model = load_model(model_name, ncls, model_source, False, False)
        model.load_state_dict(state_dict)
        print(f"{model_source}'s {model_name} can match ckpt")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available")
    model = model.to(device).eval()
    class_end = ncls if class_end is None else class_end
    patch_end = patch_num if patch_end is None else patch_end
    if not 0 <= patch_start < patch_end <= patch_num:
        raise ValueError(f"Invalid patch range [{patch_start}, {patch_end}) for patch_num={patch_num}")

    trainset = SimpleImageFolder(
        src_dir,
        ipc=ipc,
        mode='train',
        memory=memory,
        transform=None,
        class_start=class_start,
        class_end=class_end,
    )

    trainset.transform = transforms.Compose([
        transforms.ToTensor(),
        MultiRandomCrop(num_crop=num_crop, size=imsize, factor=2),
        # transforms.Resize((64, 64)),
        transforms.Normalize(mean=mean_norm, std=std_norm),
    ])
    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / std_norm[0], 1 / std_norm[1], 1 / std_norm[2]]),
         transforms.Normalize(mean=[-mean_norm[0], -mean_norm[1], -mean_norm[2]], std=[1.0, 1.0, 1.0])])
    os.makedirs(save_dir, exist_ok=True)
    total_started_at = time.perf_counter()
    generated = 0
    for img_id in range(patch_start, patch_end):
        pending_classes = [
            class_id
            for class_id in range(class_start, class_end)
            if overwrite or not os.path.isfile(image_output_path(save_dir, class_id, img_id))
        ]
        if not pending_classes:
            print(f"patch id {img_id}: all classes already exist, skipping")
            continue

        pending_indices = [
            sample_index
            for class_id in pending_classes
            for sample_index in trainset.class_sample_indices[class_id]
        ]
        loader_kwargs = dict(
            dataset=Subset(trainset, pending_indices),
            batch_size=ipc,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.startswith("cuda"),
            drop_last=False,
        )
        if workers > 0:
            loader_kwargs.update(prefetch_factor=2)
        train_loader = torch.utils.data.DataLoader(**loader_kwargs)

        for c, (images, labels) in enumerate(train_loader):
            class_id = int(labels[0].item())
            if not torch.all(labels == class_id):
                raise RuntimeError(f"A batch contains multiple classes: {labels.tolist()}")
            item_started_at = time.perf_counter()
            images = selector(
                4,
                model,
                images,
                labels,
                size=imsize,
                device=device,
                forward_batch_size=forward_batch_size,
            )
            images = mix_images(images, imsize, factor=2, mixed_img_num=1)
            save_images(save_dir, denormalize(images), class_id, img_id)
            generated += 1
            print(
                f"patch id {img_id}, class {class_id}: saved "
                f"({time.perf_counter() - item_started_at:.2f}s, generated this run: {generated})",
                flush=True,
            )

    print(
        f"RDED patch generation finished: {generated} new files in "
        f"{time.perf_counter() - total_started_at:.1f}s",
        flush=True,
    )


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.abspath(os.path.join(script_dir, "..", "Datasets"))
    parser = argparse.ArgumentParser("Generate resumable RDED initialization patches")
    parser.add_argument("--model-name", default="ResNet18")
    parser.add_argument("--model-source", default="auto", choices=["auto", "CVDD", "torchvision"])
    parser.add_argument("--dataset-name", default="CUB_imsize224")
    parser.add_argument("--ncls", type=int, default=200)
    parser.add_argument("--ipc", type=int, default=29, help="real images loaded per class")
    parser.add_argument("--imsize", type=int, default=224)
    parser.add_argument("--patch-num", type=int, default=5)
    parser.add_argument("--num-crop", type=int, default=5)
    parser.add_argument("--src-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--class-start", type=int, default=0, help="inclusive global class id")
    parser.add_argument("--class-end", type=int, default=None, help="exclusive global class id")
    parser.add_argument("--patch-start", type=int, default=0, help="inclusive patch id")
    parser.add_argument("--patch-end", type=int, default=None, help="exclusive patch id")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--forward-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-memory-cache", action="store_true", help="load images from disk in workers")
    parser.add_argument("--overwrite", action="store_true", help="regenerate completed output files")
    args = parser.parse_args()
    args.src_dir = args.src_dir or os.path.join(default_data_root, args.dataset_name)
    args.save_dir = args.save_dir or os.path.join(default_data_root, "patches", args.dataset_name, "2")
    args.ckpt_path = args.ckpt_path or os.path.join(
        default_data_root, "pretrained_models", args.dataset_name, args.model_name + ".pth"
    )
    return args


if __name__ == '__main__':
    args = parse_args()
    mean_norm = [0.4857, 0.4994, 0.4326]
    std_norm = [0.2260, 0.2215, 0.2595]
    make_patch(
        args.model_name,
        args.ckpt_path,
        args.ncls,
        args.src_dir,
        args.ipc,
        mean_norm,
        std_norm,
        args.patch_num,
        args.num_crop,
        args.imsize,
        args.save_dir,
        class_start=args.class_start,
        class_end=args.class_end,
        patch_start=args.patch_start,
        patch_end=args.patch_end,
        workers=args.workers,
        forward_batch_size=args.forward_batch_size,
        device=args.device,
        model_source=args.model_source,
        memory=not args.no_memory_cache,
        overwrite=args.overwrite,
    )
