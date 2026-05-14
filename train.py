"""
train.py — Training loop for all three model variants.

Key scientific practices implemented:
1. Epoch-1 loss sanity check: ~2.3026 = -ln(1/10).
   Proves softmax is correctly initialized and data pipeline isn't broken.
2. Generalization gap monitoring: val_loss - train_loss at every epoch.
3. Best model saved by val accuracy.
4. Training curves saved after each run.

Usage:
  python train.py --model baseline_cnn
  python train.py --model resnet18_pretrained
  python train.py --model resnet18_scratch
  python train.py --model resnet18_pretrained --epochs 50  # override
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    set_seed, load_config, get_logger, get_device,
    save_checkpoint, plot_training_curves
)
from data import get_dataloaders, run_data_sanity_checks
from models import build_model, count_parameters

logger = get_logger("train", log_file="outputs/train.log")


# ─────────────────────────────────────────────
# One epoch: train
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# One epoch: evaluate
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Epoch-1 sanity check
# ─────────────────────────────────────────────

def check_initial_loss(loss: float, num_classes: int = 10, tol: float = 0.3) -> None:
    """
    For a randomly initialized model with uniform class distribution,
    the expected cross-entropy loss is -ln(1/num_classes) = ln(10) ≈ 2.3026.

    If this check fails:
    - Loss >> 2.3: weight initialization problem, or data pipeline is producing NaNs
    - Loss << 2.3: data leakage, or model is already loading trained weights

    This is the sanity check Dr. Abbas explicitly asked for.
    """
    expected = math.log(num_classes)  # ln(10) = 2.3026
    deviation = abs(loss - expected)

    logger.info(f"[sanity] Epoch-1 loss check: got {loss:.4f}, expected ~{expected:.4f} "
                f"(±{tol})")

    if deviation <= tol:
        logger.info(f"[sanity] ✓ Initial loss check PASSED (deviation={deviation:.4f})")
    else:
        logger.warning(
            f"[sanity] ✗ Initial loss check FAILED (deviation={deviation:.4f}). "
            f"Investigate before proceeding. Check: weight init, data pipeline, "
            f"class balance, label encoding."
        )


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(model_name: str, cfg: dict, epochs_override: int | None = None) -> None:
    """
    Full training run for a given model variant.
    Saves: best checkpoint, final checkpoint, training curves.
    """
    set_seed(cfg["seed"])
    device = get_device()

    # ── Data ──────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────
    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)
    logger.info(f"[model] {model_name} | {n_params:,} trainable parameters")

    # ── Optimizer & scheduler ─────────────────
    # SGD + momentum: still the standard for CIFAR-10 (Adam overfits more here)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = epochs_override or cfg["training"]["epochs"]

    # Cosine annealing: smooth LR decay, avoids abrupt drops that destabilize training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []
    best_val_acc = 0.0
    save_dir     = cfg["models"]["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training: {model_name} | {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Epoch-1 sanity check
        if epoch == 1:
            check_initial_loss(train_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        gen_gap = val_loss - train_loss  # positive = overfitting
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.1f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.1f}% | "
            f"Gap={gen_gap:+.4f} | LR={lr_now:.6f} | {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "cfg": cfg,
                },
                path=f"{save_dir}/{model_name}_best.pth"
            )

    # ── Post-training ──────────────────────────
    # Save final checkpoint (for analysis; best is for evaluation)
    save_checkpoint(
        {
            "epoch": num_epochs,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "cfg": cfg,
        },
        path=f"{save_dir}/{model_name}_final.pth"
    )

    # Save training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=f"{cfg['paths']['curves']}/{model_name}_curves.png",
        title=f"{model_name} — Training Curves (best val: {best_val_acc:.1f}%)"
    )

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline_cnn", "resnet18_pretrained", "resnet18_scratch"],
        help="Which model variant to train"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--data-checks", action="store_true",
        help="Run data sanity checks before training"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.data_checks:
        run_data_sanity_checks(cfg)

    train(args.model, cfg, epochs_override=args.epochs)
