"""
gradcam.py — Grad-CAM implementation from scratch.

Mathematical pipeline (be able to say this out loud):
1. Forward pass — save feature maps at target layer via hook
2. Compute raw logit score for target class (NOT softmax probability)
3. Backward pass — save gradients of that score w.r.t. feature maps via hook
4. Importance weights: average gradients over spatial dimensions (H, W)
5. Weighted sum of feature map channels
6. Apply ReLU — only care about positively contributing features
7. Upsample to input size (32×32 for CIFAR-10)

Why raw logit, not softmax?
  Softmax introduces competition between classes. If class A has a high score,
  softmax suppresses class B even if the image contains clear features of B.
  The raw logit isolates the score for ONE class without this suppression.

Why ReLU at the end?
  Feature map channels can negatively contribute (suppress) the target class.
  We only want pixels that HELP identify the class, not pixels that hurt.
  ReLU zeroes out the negative contributions.

Validation: compare against jacobgil/pytorch-grad-cam using Spearman rank correlation.
  r > 0.95 confirms correct implementation.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr

from utils import get_logger, denormalize

logger = get_logger("gradcam")


# Core Grad-CAM class (from scratch)

class GradCAM:
    """
    Grad-CAM from scratch using PyTorch hooks.

    Args:
        model:        trained model (eval mode recommended)
        target_layer: the nn.Module to attach hooks to.
                      For ResNet18: model.layer4[-1]
                      For BaselineCNN: model.layer3[-1] or model.layer3
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer

        # These will be populated by hooks during forward/backward
        self._feature_maps: torch.Tensor | None = None
        self._gradients:    torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks on the target layer.

        Forward hook captures: the output feature maps (activations).
        Backward hook captures: the gradients flowing back through that layer.
        """
        def forward_hook(module, input, output):
            # output shape: (batch, channels, H, W)
            self._feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (batch, channels, H, W)
            self._gradients = grad_output[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self) -> None:
        """Always call this when done to avoid memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        input_size: tuple[int, int] = (32, 32),
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) — single image, normalized
            target_class: class index to explain. If None, uses predicted class.
            input_size:   (H, W) to upsample heatmap to. Default 32×32 for CIFAR.

        Returns:
            heatmap: (H, W) float array in [0, 1]
        """
        self.model.eval()

        # ── Step 1: Forward pass ──────────────────────────────────
        # zero_grad to ensure no residual gradients from previous calls
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1, num_classes)

        # ── Step 2: Select target class ───────────────────────────
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # ── Step 3: Compute score for target class (raw logit) ────
        # We take the raw logit, NOT softmax. See module docstring for why.
        score = logits[0, target_class]

        # ── Step 4: Backward pass ─────────────────────────────────
        # This populates self._gradients via the backward hook
        score.backward()

        # ── Step 5: Importance weights ────────────────────────────
        # Average gradients over spatial dimensions H, W
        # Shape: (channels,) — one scalar weight per feature map channel
        gradients    = self._gradients    # (1, C, H, W)
        feature_maps = self._feature_maps # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3))  # global average pooling over H, W → (1, C)

        # ── Step 6: Weighted combination of feature maps ──────────
        # Sum channels, weighted by their importance
        # weights[:, :, None, None] broadcasts to (1, C, H, W)
        cam = (weights[:, :, None, None] * feature_maps).sum(dim=1)  # (1, H, W)
        cam = cam.squeeze(0)  # (H, W)

        # ── Step 7: ReLU ──────────────────────────────────────────
        # Remove negative values — only positive contributions matter for this class
        cam = F.relu(cam)

        # ── Step 8: Upsample to input resolution ──────────────────
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H_feat, W_feat)
        cam = F.interpolate(
            cam, size=input_size, mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # ── Step 9: Normalize to [0, 1] ───────────────────────────
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# Target layer resolver

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the correct target layer for Grad-CAM based on model type.

    ResNet18:    layer4[-1] — final residual block, highest semantics
    BaselineCNN: layer3     — final conv block
    """
    if "resnet18" in model_name:
        # layer4 is a Sequential of BasicBlocks; [-1] gets the last one
        return model.layer4[-1]
    elif "baseline_cnn" in model_name:
        return model.layer3
    else:
        raise ValueError(f"Don't know target layer for model: {model_name}")



# Visualization

def overlay_heatmap(
    image_np: np.ndarray,   # HWC uint8
    cam: np.ndarray,        # HW float [0,1]
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    Returns HWC uint8 composite.
    """
    # Apply jet colormap to CAM
    heatmap_colored = cm.jet(cam)[:, :, :3]       # drop alpha channel → HW3 float
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    image_float   = image_np.astype(np.float32) / 255.0
    heatmap_float = heatmap_colored.astype(np.float32) / 255.0
    blended = (1 - alpha) * image_float + alpha * heatmap_float
    return (blended * 255).astype(np.uint8)


def save_heatmap_grid(
    model: nn.Module,
    model_name: str,
    dataset,
    cfg: dict,
    save_path: str,
    images_per_class: int = 1,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Save a grid showing Grad-CAM overlays for all 10 CIFAR-10 classes.
    Columns: [original image, Grad-CAM overlay]
    Rows: one per class (one example each)
    """
    from data import CLASSES

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]

    target_layer = get_target_layer(model, model_name)

    # Collect one image per class
    class_images = {}
    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]
        if label not in class_images:
            class_images[label] = (img_tensor, label)
        if len(class_images) == 10:
            break

    num_classes = 10
    fig, axes = plt.subplots(num_classes, 2, figsize=(5, num_classes * 2.5))
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=11, fontweight="bold")

    with GradCAM(model, target_layer) as gcam:
        for class_idx in range(num_classes):
            img_tensor, label = class_images[class_idx]
            img_np = denormalize(img_tensor, mean, std)

            # Compute CAM
            input_t = img_tensor.unsqueeze(0).to(device)
            cam = gcam(input_t, target_class=class_idx)

            overlay = overlay_heatmap(img_np, cam, alpha=0.5)

            # Plot
            axes[class_idx, 0].imshow(img_np)
            axes[class_idx, 0].set_ylabel(CLASSES[class_idx], fontsize=7, rotation=0,
                                           labelpad=40, va="center")
            axes[class_idx, 0].axis("off")

            axes[class_idx, 1].imshow(overlay)
            axes[class_idx, 1].axis("off")

    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 1].set_title("Grad-CAM", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[gradcam] Heatmap grid saved → {save_path}")


# Library parity check

def verify_against_library(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    target_class: int,
    cfg: dict,
    threshold: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Compare our Grad-CAM implementation against pytorch-grad-cam library.

    Method: compute Spearman rank correlation between heatmap values.
    If r > threshold (default 0.95), implementation is verified correct.

    This goes in the report as a methodological validation step.
    """
    try:
        from pytorch_grad_cam import GradCAM as LibGradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.warning(
            "[parity] pytorch-grad-cam not installed. "
            "Run: pip install grad-cam  — then re-run this check."
        )
        return float("nan")

    target_layer = get_target_layer(model, model_name)
    input_t = image_tensor.unsqueeze(0).to(device)

    # Our implementation
    with GradCAM(model, target_layer) as gcam:
        our_cam = gcam(input_t, target_class=target_class)

    # Library implementation
    lib_gcam = LibGradCAM(model=model, target_layers=[target_layer])
    targets  = [ClassifierOutputTarget(target_class)]
    lib_cam  = lib_gcam(input_tensor=input_t, targets=targets)[0]

    # Spearman correlation on flattened arrays
    r, p = spearmanr(our_cam.flatten(), lib_cam.flatten())
    status = "✓ PASSED" if r >= threshold else "✗ FAILED"

    logger.info(
        f"[parity] Library parity check: Spearman r = {r:.4f} (p={p:.4e}) "
        f"| threshold = {threshold} | {status}"
    )

    if r < threshold:
        logger.warning(
            "[parity] Low correlation. Common causes: wrong target layer, "
            "wrong backward hook, averaging over wrong dims."
        )

    return r
