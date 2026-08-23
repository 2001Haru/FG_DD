from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


class EncodedSubclassFolder(Dataset):
    """ImageFolder-compatible reader that permits empty numeric class folders."""

    def __init__(self, root, num_classes, transform=None):
        self.root = str(root)
        self.transform = transform
        self.target_transform = None
        self.loader = default_loader
        class_name_width = max(3, len(str(num_classes - 1)))
        self.classes = [
            f"{index:0{class_name_width}d}" for index in range(num_classes)
        ]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        root = Path(root)
        actual_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
        if actual_dirs != self.classes:
            raise RuntimeError(
                f"numeric subclass directory mismatch: expected {num_classes}, "
                f"found {len(actual_dirs)} in {root}"
            )
        extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        self.samples = []
        for target, class_name in enumerate(self.classes):
            for path in sorted((root / class_name).iterdir()):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.samples.append((str(path), target))
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
