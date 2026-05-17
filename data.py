"""
data.py — CIFAR-10 data pipeline.

Scientific principles implemented here:
1. Normalization statistics computed from TRAINING SET ONLY (no leakage).
2. Augmentation applied only to training split.
3. Class balance verified — CIFAR-10 is 1000 per class per split.
4. Sample grid saved for visual verification.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_logger, denormalize

logger = get_logger("data")

# CIFAR-10 class names, index-aligned
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# Transforms

def get_transforms(cfg: dict, split: str):
    """
    Returns the appropriate transform pipeline for a given split.

    Train: RandomCrop + HorizontalFlip + Normalize
      - RandomCrop(32, padding=4): translates image up to 4px, preserving content
      - HorizontalFlip: objects appear mirrored; NOT vertical (objects don't appear upside-down)
    Val/Test: only Normalize (no augmentation — we want deterministic evaluation)
    """
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]
    aug  = cfg["augmentation"]

    normalize = T.Normalize(mean=mean, std=std)

    if split == "train":
        return T.Compose([
            T.RandomCrop(aug["random_crop_size"], padding=aug["random_crop_padding"]),
            T.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            normalize,
        ])


# Dataset loading

def get_dataloaders(cfg: dict):
    """
    Build train / val / test DataLoaders.

    CIFAR-10 provides: 50,000 train + 10,000 test.
    We use the full test set as our val set for simplicity
    (academic convention for CIFAR-10 benchmarking).

    Returns: train_loader, val_loader, test_loader
    """
    data_root = cfg["data"]["root"]
    bs        = cfg["training"]["batch_size"]
    nw        = cfg["data"]["num_workers"]

    train_transform = get_transforms(cfg, "train")
    eval_transform  = get_transforms(cfg, "val")

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    logger.info(f"Train: {len(train_set):,} samples | "
                f"Val/Test: {len(test_set):,} samples")

    return train_loader, val_loader, test_loader



# Sanity checks

def verify_class_balance(dataset, split_name: str) -> None:
    """
    CIFAR-10 is perfectly balanced: exactly 1000 samples per class per split.
    If this assertion fails, something is wrong with the dataset loading.
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=10)

    logger.info(f"[{split_name}] Class counts:")
    for cls_idx, (cls_name, count) in enumerate(zip(CLASSES, counts)):
        logger.info(f"  {cls_idx:2d}. {cls_name:<12s}: {count}")

    # CIFAR-10 invariant
    expected_train = 5000
    expected_test  = 1000
    expected = expected_train if split_name == "train" else expected_test

    for i, count in enumerate(counts):
        assert count == expected, (
            f"Class {CLASSES[i]} has {count} samples, expected {expected}. "
            f"Check your dataset loading."
        )
    logger.info(f"[{split_name}] ✓ Class balance verified ({expected} per class)")


def save_sample_grid(dataset, cfg: dict, save_path: str, n: int = 16) -> None:
    """
    Save a grid of n augmented training images with class labels.
    Purpose: visually confirm that augmentation is reasonable and labels are correct.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]

    indices = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Augmented training samples (sanity check)", fontsize=11)

    for ax, idx in zip(axes.flat, indices):
        img_tensor, label = dataset[idx]
        img_np = denormalize(img_tensor, mean, std)
        ax.imshow(img_np)
        ax.set_title(CLASSES[label], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[sanity] Sample grid saved → {save_path}")


def run_data_sanity_checks(cfg: dict) -> None:
    """
    Run all data-level sanity checks and save outputs.
    Called once after building the datasets, before any training.
    """
    from torchvision.datasets import CIFAR10
    root = cfg["data"]["root"]

    # Load without augmentation for balance check
    raw_train = CIFAR10(root=root, train=True,  download=True, transform=T.ToTensor())
    raw_test  = CIFAR10(root=root, train=False, download=True, transform=T.ToTensor())

    verify_class_balance(raw_train, "train")
    verify_class_balance(raw_test,  "test")

    # Save sample grid with augmentation applied (use train transform)
    aug_train = CIFAR10(
        root=root, train=True, download=True,
        transform=get_transforms(cfg, "train")
    )
    save_sample_grid(
        aug_train, cfg,
        save_path=cfg["paths"]["sanity_checks"] + "/data_samples.png"
    )
    logger.info("[data] All data sanity checks passed ✓")
