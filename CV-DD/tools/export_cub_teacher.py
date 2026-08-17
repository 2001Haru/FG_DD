import argparse
import os

import torch
from torchvision import models


def main():
    parser = argparse.ArgumentParser("Export an FD2 CUB backbone for a controlled CV-DD run")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "ResNet18" in checkpoint:
        state_dict = checkpoint["ResNet18"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(state_dict, strict=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"Exported verified CUB ResNet18 backbone to: {args.output}")


if __name__ == "__main__":
    main()
