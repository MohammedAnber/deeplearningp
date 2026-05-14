"""
sanity_checks.py — Adebayo et al. (2018) randomization tests for Grad-CAM.

Paper: "Sanity Checks for Saliency Maps" (Adebayo et al., NeurIPS 2018)
https://arxiv.org/abs/1810.03292

Two independent tests that MUST be run before trusting any Grad-CAM result:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST 1 — MODEL PARAMETER RANDOMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hypothesis: if Grad-CAM truly reflects learned model behavior,
then randomizing model weights should produce visually different heatmaps.

Protocol (cascading, top-down):
  1. Start from a trained model
  2. Re-initialize the LAST layer's weights to random
  3. Compute heatmap → should differ from trained
  4. Re-initialize second-to-last layer → differ more
  5. Continue until ALL layers are random

If heatmaps DON'T change → the attribution method is not sensitive to model
learning → it is producing edge-detector artifacts, not learned explanations.

PASS criterion: SSIM < 0.5 or Spearman r < 0.5 between trained and
fully-randomized heatmaps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST 2 — DATA RANDOMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hypothesis: if the explanation reflects class-specific learning,
then a model trained on randomly shuffled labels should produce
different heatmaps (because it hasn't learned meaningful features).

Protocol:
  1. Train a model on CIFAR-10 with SHUFFLED labels (pure memorization)
  2. Train an identical model on correct labels
  3. Compare Grad-CAM heatmaps for the same images

If heatmaps are SIMILAR → attribution is insensitive to what the model learned
→ not a meaningful explanation.

PASS criterion: SSIM < 0.5 between label-shuffled and correctly-trained maps.

Usage:
  python sanity_checks.py --model resnet18_pretrained --test model_randomization
  python sanity_checks.py --model resnet18_pretrained --test data_randomization
  python sanity_checks.py --model resnet18_pretrained --test both
"""

from __future__ import annotations
import argparse
import copy
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn(
        "scikit-image not found — SSIM disabled. "
        "Run: pip install scikit-image"
    )

from utils import set_seed, load_config, get_logger, get_device, load_checkpoint
from data import get_dataloaders, CLASSES
from models import build_model
from gradcam import GradCAM, get_target_layer, overlay_heatmap
from utils import denormalize

logger = get_logger("sanity_checks", log_file="outputs/sanity_checks.log")

# ─────────────────────────────────────────────
# Similarity metrics
# ─────────────────────────────────────────────

def spearman_r(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Spearman rank correlation between two flattened heatmaps."""
    r, _ = spearmanr(map_a.flatten(), map_b.flatten())
    return float(r)


def ssim_score(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """
    Structural Similarity Index between two heatmaps.
    Returns NaN if scikit-image is not installed.
    """
    if not HAS_SKIMAGE:
        return float("nan")
    # SSIM expects values in same range; both are [0,1]
    return float(ssim(map_a, map_b, data_range=1.0))


# ─────────────────────────────────────────────
# Helper: compute a single Grad-CAM heatmap
# ─────────────────────────────────────────────

def compute_cam(
    model: nn.Module,
    model_name: str,
    img_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute a single normalized Grad-CAM heatmap.
    Returns: (32, 32) float array in [0, 1].
    """
    target_layer = get_target_layer(model, model_name)
    input_t = img_tensor.unsqueeze(0).to(device)
    with GradCAM(model, target_layer) as gcam:
        cam = gcam(input_t, target_class=target_class)
    return cam  # already [0,1] normalized in GradCAM.__call__


# ─────────────────────────────────────────────
# Layer enumeration for cascading randomization
# ─────────────────────────────────────────────

def get_named_layers_top_down(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Returns list of (name, module) pairs for all layers that HAVE parameters,
    ordered from LAST to FIRST (top-down cascading randomization).

    We only include leaf modules with weights to avoid double-randomizing
    sub-modules of Sequential blocks.
    """
    named = [
        (name, module)
        for name, module in model.named_modules()
        if len(list(module.parameters(recurse=False))) > 0
    ]
    # Reverse so we start from the classifier head
    return list(reversed(named))


def randomize_layer(module: nn.Module) -> None:
    """
    Re-initialize all parameters of a single module using Kaiming uniform.
    Mimics standard PyTorch default initialization.
    """
    for param in module.parameters(recurse=False):
        if param.dim() >= 2:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        else:
            # bias: uniform in [-1/fan_in, 1/fan_in]
            fan_in = param.shape[0] if param.dim() == 1 else param.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(param, -bound, bound)


# ─────────────────────────────────────────────
# TEST 1: Model Parameter Randomization
# ─────────────────────────────────────────────

def test_model_randomization(
    model: nn.Module,
    model_name: str,
    img_tensor: torch.Tensor,
    target_class: int,
    cfg: dict,
    device: torch.device,
    save_dir: str,
    pass_threshold_r: float = 0.5,
    pass_threshold_ssim: float = 0.5,
) -> dict:
    """
    Cascading top-down model randomization test (Adebayo et al. §3.1).

    For each step in the cascade:
      - Randomize one more layer (from output → input)
      - Recompute Grad-CAM
      - Measure similarity to TRAINED heatmap

    A well-behaved attribution method should show DECREASING similarity
    as more layers are randomized.

    Returns: dict with per-step spearman_r, ssim, and overall PASS/FAIL.
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Model Parameter Randomization (Adebayo §3.1)")
    logger.info("=" * 60)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]

    # 1. Baseline: heatmap from fully TRAINED model
    trained_cam = compute_cam(model, model_name, img_tensor, target_class, device)
    img_np = denormalize(img_tensor, mean, std)

    # Work on a deep copy so we don't destroy the trained model
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    # 2. Get layers in top-down order
    layers = get_named_layers_top_down(model_copy)
    logger.info(f"Cascading randomization over {len(layers)} parameterized layers.")

    results = []
    cams_for_plot = [("trained", trained_cam)]

    cumulative_randomized = 0

    for layer_name, layer_module in layers:
        randomize_layer(layer_module)
        cumulative_randomized += 1

        cam = compute_cam(model_copy, model_name, img_tensor, target_class, device)

        r    = spearman_r(trained_cam, cam)
        s    = ssim_score(trained_cam, cam)

        results.append({
            "layer": layer_name,
            "n_randomized": cumulative_randomized,
            "spearman_r": r,
            "ssim": s,
        })

        logger.info(
            f"  [{cumulative_randomized:2d}/{len(layers)}] Randomized '{layer_name}' | "
            f"Spearman r={r:.4f} | SSIM={s:.4f}"
        )

        # Save intermediate cam for every 3rd step and the final one
        if cumulative_randomized % max(1, len(layers) // 4) == 0 or cumulative_randomized == len(layers):
            label = f"rand_{cumulative_randomized}"
            cams_for_plot.append((label, cam))

    # 3. Final similarity (fully randomized) — this is the key number
    final_r    = results[-1]["spearman_r"]
    final_ssim = results[-1]["ssim"]

    passed_r    = final_r    < pass_threshold_r
    passed_ssim = final_ssim < pass_threshold_ssim or not HAS_SKIMAGE
    passed      = passed_r and passed_ssim

    status = "✓ PASSED" if passed else "✗ FAILED"
    logger.info(
        f"\n[Test 1 Result] Fully randomized: "
        f"Spearman r={final_r:.4f} (threshold<{pass_threshold_r}), "
        f"SSIM={final_ssim:.4f} (threshold<{pass_threshold_ssim}) "
        f"→ {status}"
    )
    if not passed:
        logger.warning(
            "[Test 1] FAILED: Heatmaps barely changed after full weight randomization. "
            "This suggests Grad-CAM is responding to image structure (edges), "
            "NOT to learned model features. Review target layer selection."
        )

    # 4. Plot: heatmap progression + similarity curve
    _plot_randomization_cascade(
        img_np=img_np,
        cams_for_plot=cams_for_plot,
        results=results,
        model_name=model_name,
        save_dir=save_dir,
        test_name="model_randomization",
    )

    return {
        "test": "model_randomization",
        "passed": passed,
        "final_spearman_r": final_r,
        "final_ssim": final_ssim,
        "per_layer": results,
    }


# ─────────────────────────────────────────────
# TEST 2: Data Randomization
# ─────────────────────────────────────────────

def _train_shuffled_model(
    model_name: str,
    cfg: dict,
    device: torch.device,
    n_epochs: int = 10,
) -> nn.Module:
    """
    Train a model on CIFAR-10 with RANDOMLY SHUFFLED labels.

    The model will achieve ~100% train accuracy by memorizing random label
    assignments, but ~10% val accuracy (chance). This is intentional.
    A model trained this way cannot have learned meaningful visual features.

    Only 10 epochs: enough to memorize on a small subset, fast enough to run
    in a sanity-check script.
    """
    logger.info("[Data Rand] Training a SHUFFLED-LABEL model (10 epochs, for test only)...")

    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as T

    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Load raw dataset
    raw = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"], train=True, download=True, transform=transform
    )

    # Shuffle labels on a subset (2000 samples keeps it fast)
    subset_size = 2000
    indices = np.random.choice(len(raw), subset_size, replace=False).tolist()
    shuffled_labels = torch.randperm(subset_size) % 10  # random labels 0-9

    class ShuffledSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, fake_labels):
            self.dataset     = dataset
            self.indices     = indices
            self.fake_labels = fake_labels
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, _ = self.dataset[self.indices[i]]
            return img, int(self.fake_labels[i])

    shuffled_ds     = ShuffledSubset(raw, indices, shuffled_labels)
    shuffled_loader = DataLoader(shuffled_ds, batch_size=64, shuffle=True,
                                 num_workers=0, pin_memory=True)

    model   = build_model(model_name, cfg).to(device)
    optim_  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    crit    = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for imgs, labels in shuffled_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optim_.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            optim_.step()
            total_loss += loss.item()
        logger.info(f"  [shuffle-train] Epoch {epoch}/{n_epochs} | "
                    f"avg loss={total_loss/len(shuffled_loader):.4f}")

    logger.info("[Data Rand] Shuffled-label model trained.")
    return model


def test_data_randomization(
    trained_model: nn.Module,
    model_name: str,
    img_tensor: torch.Tensor,
    target_class: int,
    cfg: dict,
    device: torch.device,
    save_dir: str,
    n_shuffle_epochs: int = 10,
    pass_threshold_r: float = 0.5,
    pass_threshold_ssim: float = 0.5,
) -> dict:
    """
    Data randomization test (Adebayo et al. §3.2).

    Compares Grad-CAM from:
      A) model trained on correct labels
      B) model trained on randomly shuffled labels (pure memorization)

    If A ≈ B → attribution is insensitive to what was actually learned.
    If A ≠ B → attribution reflects genuine class-specific patterns.
    """
    logger.info("=" * 60)
    logger.info("TEST 2: Data Randomization (Adebayo §3.2)")
    logger.info("=" * 60)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]

    # A: trained model heatmap
    trained_cam = compute_cam(trained_model, model_name, img_tensor, target_class, device)

    # B: shuffled-label model heatmap
    shuffled_model = _train_shuffled_model(model_name, cfg, device, n_epochs=n_shuffle_epochs)
    shuffled_model.eval()
    shuffled_cam = compute_cam(shuffled_model, model_name, img_tensor, target_class, device)

    r    = spearman_r(trained_cam, shuffled_cam)
    s    = ssim_score(trained_cam, shuffled_cam)

    passed_r    = r    < pass_threshold_r
    passed_ssim = s    < pass_threshold_ssim or not HAS_SKIMAGE
    passed      = passed_r and passed_ssim

    status = "✓ PASSED" if passed else "✗ FAILED"
    logger.info(
        f"\n[Test 2 Result] Spearman r={r:.4f} (threshold<{pass_threshold_r}), "
        f"SSIM={s:.4f} (threshold<{pass_threshold_ssim}) → {status}"
    )

    if not passed:
        logger.warning(
            "[Test 2] FAILED: Trained and shuffled-label models produce similar heatmaps. "
            "The attribution is not capturing class-specific features. "
            "Consider a different attribution method."
        )

    # Plot side-by-side comparison
    img_np = denormalize(img_tensor, mean, std)
    _plot_data_randomization(
        img_np=img_np,
        trained_cam=trained_cam,
        shuffled_cam=shuffled_cam,
        model_name=model_name,
        target_class_name=CLASSES[target_class],
        r=r,
        ssim_val=s,
        passed=passed,
        save_dir=save_dir,
    )

    return {
        "test": "data_randomization",
        "passed": passed,
        "spearman_r": r,
        "ssim": s,
    }


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def _plot_randomization_cascade(
    img_np: np.ndarray,
    cams_for_plot: list[tuple[str, np.ndarray]],
    results: list[dict],
    model_name: str,
    save_dir: str,
    test_name: str,
) -> None:
    """Save cascade heatmap grid + similarity-vs-layers curve."""
    n_cams = len(cams_for_plot)

    # ── Figure 1: heatmap progression ──────────
    fig, axes = plt.subplots(2, n_cams, figsize=(3 * n_cams, 6))
    fig.suptitle(
        f"Model Randomization — {model_name}\n"
        "Top: original | Bottom: Grad-CAM overlay",
        fontsize=10, fontweight="bold"
    )

    for col, (label, cam) in enumerate(cams_for_plot):
        overlay = _blend(img_np, cam)
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(label, fontsize=7)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    plt.tight_layout()
    grid_path = f"{save_dir}/{test_name}_heatmap_cascade_{model_name}.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[plot] Cascade heatmaps → {grid_path}")

    # ── Figure 2: similarity curve ─────────────
    ns  = [r["n_randomized"] for r in results]
    rs  = [r["spearman_r"]   for r in results]
    ss  = [r["ssim"]         for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, rs, label="Spearman r", color="#2563EB", marker="o", markersize=3)
    if HAS_SKIMAGE:
        ax.plot(ns, ss, label="SSIM",       color="#DC2626", marker="s", markersize=3,
                linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Pass threshold (0.5)")
    ax.set_xlabel("# Layers Randomized (from output)")
    ax.set_ylabel("Similarity to Trained Heatmap")
    ax.set_title(f"Model Randomization Sensitivity — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    curve_path = f"{save_dir}/{test_name}_similarity_curve_{model_name}.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[plot] Similarity curve → {curve_path}")


def _plot_data_randomization(
    img_np: np.ndarray,
    trained_cam: np.ndarray,
    shuffled_cam: np.ndarray,
    model_name: str,
    target_class_name: str,
    r: float,
    ssim_val: float,
    passed: bool,
    save_dir: str,
) -> None:
    """Side-by-side comparison for data randomization test."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    status_str = "✓ PASSED" if passed else "✗ FAILED"
    fig.suptitle(
        f"Data Randomization Test — {model_name} — class: {target_class_name}\n"
        f"Spearman r={r:.4f} | SSIM={ssim_val:.4f} | {status_str}",
        fontsize=10, fontweight="bold"
    )

    # Row 0: trained model
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(trained_cam, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title("Trained — CAM", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_blend(img_np, trained_cam))
    axes[0, 2].set_title("Trained — Overlay", fontsize=9)
    axes[0, 2].axis("off")
    axes[0, 0].set_ylabel("Correct labels", fontsize=8, labelpad=5)

    # Row 1: shuffled model
    axes[1, 0].imshow(img_np)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(shuffled_cam, cmap="jet", vmin=0, vmax=1)
    axes[1, 1].set_title("Shuffled — CAM", fontsize=9)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(_blend(img_np, shuffled_cam))
    axes[1, 2].set_title("Shuffled — Overlay", fontsize=9)
    axes[1, 2].axis("off")
    axes[1, 0].set_ylabel("Random labels", fontsize=8, labelpad=5)

    plt.tight_layout()
    path = f"{save_dir}/data_randomization_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[plot] Data randomization comparison → {path}")


def _blend(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Simple blend helper — avoids importing overlay_heatmap redundantly."""
    import matplotlib.cm as cm
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    blended = ((1 - alpha) * img_np + alpha * heatmap).clip(0, 255).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table of all test results."""
    logger.info("\n" + "=" * 60)
    logger.info("ADEBAYO SANITY CHECK SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for res in results:
        passed_str = "✓ PASSED" if res["passed"] else "✗ FAILED"
        all_passed = all_passed and res["passed"]

        if res["test"] == "model_randomization":
            logger.info(
                f"  Model Randomization : {passed_str} "
                f"| final Spearman r={res['final_spearman_r']:.4f} "
                f"| final SSIM={res['final_ssim']:.4f}"
            )
        elif res["test"] == "data_randomization":
            logger.info(
                f"  Data Randomization  : {passed_str} "
                f"| Spearman r={res['spearman_r']:.4f} "
                f"| SSIM={res['ssim']:.4f}"
            )

    overall = "✓ ALL PASSED — Grad-CAM is trustworthy for this model." \
              if all_passed else \
              "✗ ONE OR MORE CHECKS FAILED — review attribution before trusting results."
    logger.info(f"\n  Overall: {overall}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adebayo et al. sanity checks for Grad-CAM"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["baseline_cnn", "resnet18_pretrained", "resnet18_scratch"],
        help="Which model to evaluate"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth). "
             "Defaults to models/{model_name}_best.pth"
    )
    parser.add_argument(
        "--test", type=str, default="both",
        choices=["model_randomization", "data_randomization", "both"],
        help="Which test(s) to run"
    )
    parser.add_argument(
        "--class-idx", type=int, default=0,
        help="CIFAR-10 class index to use for the test image (0=airplane)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--shuffle-epochs", type=int, default=10,
        help="Epochs to train shuffled-label model (Test 2)"
    )
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────
    cfg    = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    # ── Load trained model ──────────────────────
    model = build_model(args.model, cfg).to(device)
    ckpt_path = args.checkpoint or f"{cfg['models']['save_dir']}/{args.model}_best.pth"
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # ── Grab one test image of the target class ─
    _, val_loader, _ = get_dataloaders(cfg)
    img_tensor, label = None, None
    for imgs, labels in val_loader:
        for i in range(len(labels)):
            if labels[i].item() == args.class_idx:
                img_tensor = imgs[i]
                label = labels[i].item()
                break
        if img_tensor is not None:
            break

    if img_tensor is None:
        raise RuntimeError(f"No image found for class {args.class_idx} in val set.")

    logger.info(
        f"Running sanity check(s) on: model={args.model}, "
        f"class={CLASSES[label]} (idx={label})"
    )

    save_dir = cfg["paths"]["sanity_checks"]
    results  = []

    # ── Run selected tests ─────────────────────
    if args.test in ("model_randomization", "both"):
        result = test_model_randomization(
            model=model,
            model_name=args.model,
            img_tensor=img_tensor,
            target_class=label,
            cfg=cfg,
            device=device,
            save_dir=save_dir,
        )
        results.append(result)

    if args.test in ("data_randomization", "both"):
        result = test_data_randomization(
            trained_model=model,
            model_name=args.model,
            img_tensor=img_tensor,
            target_class=label,
            cfg=cfg,
            device=device,
            save_dir=save_dir,
            n_shuffle_epochs=args.shuffle_epochs,
        )
        results.append(result)

    print_summary(results)
