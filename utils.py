"""
utils.py — shared utilities across the project.

Covers: reproducibility, config loading, logging helpers, checkpoint I/O.
"""

import os
import random
import logging
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on server
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix all sources of randomness for full reproducibility.
    Called once at the top of every script before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops — slight speed penalty, full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] All RNG sources fixed to {seed}")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Returns a logger that writes to stdout and optionally to a file.
    Use this instead of bare print() so output is timestamped and filterable.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    """Save model checkpoint. state should include model, optimizer, epoch, metrics."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device="cpu"):
    """
    Load checkpoint into model (and optionally optimizer).
    Returns the full state dict so callers can read epoch/metrics.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"[checkpoint] Loaded ← {path}  (epoch {state.get('epoch', '?')})")
    return state


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """
    Save a 1×2 figure: loss curve + accuracy curve.
    The generalization gap (val_loss - train_loss) is what you monitor.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_losses, label="Train", color="#2563EB")
    axes[0].plot(epochs, val_losses,   label="Val",   color="#DC2626", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_accs, label="Train", color="#2563EB")
    axes[1].plot(epochs, val_accs,   label="Val",   color="#DC2626", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Training curves saved → {save_path}")


def denormalize(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    Undo normalization for visualization.
    tensor: CHW float tensor.
    Returns HWC uint8 numpy array.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU. Print which one is being used."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device
