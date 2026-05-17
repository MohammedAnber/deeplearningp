"""
models.py — Model architecture definitions.

Three models with distinct scientific motivations:

1. BaselineCNN   — 3-conv scratch model. Simple comparison target.
                   Grad-CAM maps will be coarse/blurry (limited feature depth).

2. ResNet18      — Used for both scratch and pretrained variants.
                   Pretrained: carries ImageNet knowledge → cleaner maps earlier.
                   Scratch:    shows what architecture alone buys you.

The key insight for Grad-CAM: the target layer determines what the heatmap shows.
  layer2 → edges and textures
  layer3 → object parts and regions
  layer4 → full semantic objects  ← this is what we want
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# 1. Baseline CNN

class BaselineCNN(nn.Module):
    """
    Simple 3-layer convolutional network.
    Input:  3 × 32 × 32 (CIFAR-10)
    Output: 10-class logits

    Architecture choices:
    - BatchNorm after each conv: stabilizes training, acts as regularizer
    - MaxPool halves spatial dims each time: 32 → 16 → 8 → 4
    - Dropout(0.5) before classifier: reduces overfitting on small dataset
    - last conv is `layer3` — hook Grad-CAM here
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32 → 16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16 → 8
        )
        # layer3 is the Grad-CAM target for this model
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


# 2. ResNet18 (pretrained or scratch)

def build_resnet18(pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    """
    Build ResNet18 adapted for CIFAR-10.

    Two adaptations from the ImageNet version:
    1. Replace the final FC layer (1000 classes → 10 classes).
    2. Modify the first conv layer: original is 7×7 stride-2 (designed for 224×224).
       For 32×32 CIFAR images, we use 3×3 stride-1 to avoid shrinking spatial dims too fast.
       We also remove the initial MaxPool for the same reason.

    pretrained=True:  loads ImageNet weights → better maps from the start
    pretrained=False: random init → shows what architecture alone contributes
    """
    if pretrained:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        model = tv_models.resnet18(weights=weights)
        print("[model] ResNet18 loaded with ImageNet pretrained weights")
    else:
        model = tv_models.resnet18(weights=None)
        print("[model] ResNet18 initialized from scratch (random weights)")

    # Adapt first conv for 32×32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove initial MaxPool — would shrink 32×32 to 16×16 immediately
    model.maxpool = nn.Identity()

    # Replace final classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# Model factory

def build_model(model_name: str, cfg: dict) -> nn.Module:
    """
    Factory function. model_name options:
      "baseline_cnn"
      "resnet18_pretrained"
      "resnet18_scratch"
    """
    num_classes = 10

    if model_name == "baseline_cnn":
        dropout = cfg["models"]["baseline_cnn"]["dropout"]
        return BaselineCNN(num_classes=num_classes, dropout=dropout)

    elif model_name == "resnet18_pretrained":
        return build_resnet18(pretrained=True, num_classes=num_classes)

    elif model_name == "resnet18_scratch":
        return build_resnet18(pretrained=False, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: baseline_cnn, resnet18_pretrained, resnet18_scratch")


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters. Useful for reporting in paper."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
