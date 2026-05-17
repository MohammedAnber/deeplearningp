"""
main.py — Master orchestrator for the CIFAR-10 Grad-CAM research pipeline.

Run order (the scientific story):
  Stage 1 — Data checks:      verify pipeline is sound before training
  Stage 2 — Training:         train all three model variants
  Stage 3 — Evaluation:       accuracy + confusion matrices
  Stage 4 — Grad-CAM:         generate heatmap grids for all models
  Stage 5 — Library parity:   validate our Grad-CAM vs pytorch-grad-cam
  Stage 6 — Adebayo checks:   model & data randomization sanity tests

Each stage can be run independently via --stages flag, or all in sequence.

Usage:
  python main.py                                     # run everything
  python main.py --stages data train                 # only data + train
  python main.py --stages gradcam sanity             # only GradCAM + checks
  python main.py --stages eval --models baseline_cnn # evaluate one model
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# ── Bootstrap: make sure output dirs exist before any import logs to them
for d in ["outputs/heatmaps", "outputs/curves", "outputs/sanity_checks",
          "outputs/eval", "models", "data"]:
    Path(d).mkdir(parents=True, exist_ok=True)

from utils import set_seed, load_config, get_logger, get_device, load_checkpoint
from data import get_dataloaders, run_data_sanity_checks, CLASSES
from models import build_model, count_parameters

logger = get_logger("main", log_file="outputs/main.log")

ALL_MODELS  = ["baseline_cnn", "resnet18_pretrained", "resnet18_scratch"]
ALL_STAGES  = ["data", "train", "eval", "gradcam", "parity", "sanity"]


# Stage 1 — Data sanity checks

def stage_data(cfg: dict) -> None:
    logger.info("━" * 60)
    logger.info("STAGE 1 — Data Sanity Checks")
    logger.info("━" * 60)
    t0 = time.time()
    run_data_sanity_checks(cfg)
    logger.info(f"Stage 1 complete in {time.time()-t0:.1f}s")


# Stage 2 — Training


def stage_train(cfg: dict, models: list[str], epochs_override: int | None) -> None:
    logger.info("━" * 60)
    logger.info(f"STAGE 2 — Training: {models}")
    logger.info("━" * 60)

    from train import train
    for model_name in models:
        logger.info(f"\n▶ Training: {model_name}")
        t0 = time.time()
        train(model_name, cfg, epochs_override=epochs_override)
        logger.info(f"  Done in {(time.time()-t0)/60:.1f} min")


# Stage 3 — Evaluation

def stage_eval(cfg: dict, models: list[str], device) -> dict:
    """
    Evaluate all trained models on the test set.
    Saves: per-model accuracy, confusion matrix, class-level accuracy.
    Returns: dict mapping model_name → {accuracy, per_class_acc}
    """
    logger.info("━" * 60)
    logger.info(f"STAGE 3 — Evaluation: {models}")
    logger.info("━" * 60)

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    _, _, test_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    results   = {}

    for model_name in models:
        ckpt_path = f"{cfg['models']['save_dir']}/{model_name}_best.pth"
        if not Path(ckpt_path).exists():
            logger.warning(f"  Checkpoint not found: {ckpt_path} — skipping {model_name}")
            continue

        model = build_model(model_name, cfg).to(device)
        state = load_checkpoint(ckpt_path, model, device=device)
        model.eval()

        # ── Inference ──────────────────────────
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        n          = len(all_labels)
        accuracy   = 100.0 * (all_preds == all_labels).sum() / n
        avg_loss   = total_loss / n

        # Per-class accuracy
        per_class = {}
        for cls_idx, cls_name in enumerate(CLASSES):
            mask = all_labels == cls_idx
            per_class[cls_name] = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()

        results[model_name] = {
            "accuracy": accuracy,
            "avg_loss": avg_loss,
            "per_class": per_class,
            "best_epoch": state.get("epoch", "?"),
        }

        logger.info(
            f"  {model_name:<26s} | "
            f"Acc={accuracy:.2f}% | Loss={avg_loss:.4f} | "
            f"BestEpoch={state.get('epoch', '?')}"
        )

        # ── Confusion matrix ────────────────────
        _save_confusion_matrix(
            preds=all_preds, labels=all_labels,
            model_name=model_name,
            save_path=f"outputs/eval/{model_name}_confusion.png"
        )

    # ── Summary table ───────────────────────────
    logger.info("\n  ── Accuracy Summary ──")
    logger.info(f"  {'Model':<26s} | {'Accuracy':>10s} | {'Avg Loss':>10s}")
    logger.info("  " + "-" * 52)
    for mn, r in results.items():
        logger.info(f"  {mn:<26s} | {r['accuracy']:>9.2f}% | {r['avg_loss']:>10.4f}")

    _save_accuracy_bar(results, save_path="outputs/eval/accuracy_comparison.png")

    return results


def _save_confusion_matrix(
    preds: "np.ndarray",
    labels: "np.ndarray",
    model_name: str,
    save_path: str,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    # Row-normalize to show percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% of true class")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")

    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_norm[i, j] > 60 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.0f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [eval] Confusion matrix → {save_path}")


def _save_accuracy_bar(results: dict, save_path: str) -> None:
    import matplotlib.pyplot as plt

    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=["#2563EB", "#16A34A", "#DC2626"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Accuracy Comparison — CIFAR-10", fontweight="bold")
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, label="90% reference")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(fontsize=8)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [eval] Accuracy comparison bar → {save_path}")


# Stage 4 — Grad-CAM heatmap grids

def stage_gradcam(cfg: dict, models: list[str], device) -> None:
    logger.info("━" * 60)
    logger.info(f"STAGE 4 — Grad-CAM Heatmaps: {models}")
    logger.info("━" * 60)

    import torchvision
    import torchvision.transforms as T
    from gradcam import save_heatmap_grid

    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    val_set = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"], train=False, download=True, transform=transform
    )

    for model_name in models:
        ckpt_path = f"{cfg['models']['save_dir']}/{model_name}_best.pth"
        if not Path(ckpt_path).exists():
            logger.warning(f"  No checkpoint for {model_name} — skipping Grad-CAM")
            continue

        model = build_model(model_name, cfg).to(device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()

        save_path = f"{cfg['paths']['heatmaps']}/{model_name}_gradcam_grid.png"
        logger.info(f"  ▶ {model_name} → {save_path}")

        save_heatmap_grid(
            model=model,
            model_name=model_name,
            dataset=val_set,
            cfg=cfg,
            save_path=save_path,
            device=device,
        )


# Stage 5 — Library parity check

def stage_parity(cfg: dict, models: list[str], device) -> None:
    """
    Validate our Grad-CAM against pytorch-grad-cam library.
    Requires: pip install grad-cam
    """
    logger.info("━" * 60)
    logger.info(f"STAGE 5 — Library Parity Check: {models}")
    logger.info("━" * 60)

    import torchvision
    import torchvision.transforms as T
    from gradcam import verify_against_library

    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    val_set = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"], train=False, download=True, transform=transform
    )

    # Use one image from class 0 (airplane) for the parity check
    img_tensor, label = val_set[0]
    threshold = cfg["evaluation"]["library_parity_threshold"]

    for model_name in models:
        ckpt_path = f"{cfg['models']['save_dir']}/{model_name}_best.pth"
        if not Path(ckpt_path).exists():
            logger.warning(f"  No checkpoint for {model_name} — skipping parity")
            continue

        model = build_model(model_name, cfg).to(device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()

        r = verify_against_library(
            model=model,
            model_name=model_name,
            image_tensor=img_tensor,
            target_class=label,
            cfg=cfg,
            threshold=threshold,
            device=device,
        )
        logger.info(f"  {model_name}: Spearman r = {r:.4f}")


# Stage 6 — Adebayo sanity checks

def stage_sanity(
    cfg: dict,
    models: list[str],
    device,
    test: str = "both",
    class_idx: int = 0,
    shuffle_epochs: int = 10,
) -> None:
    logger.info("━" * 60)
    logger.info(f"STAGE 6 — Adebayo Sanity Checks: {models} | test={test}")
    logger.info("━" * 60)

    import torchvision
    import torchvision.transforms as T
    from sanity_checks import (
        test_model_randomization, test_data_randomization, print_summary
    )

    mean = cfg["data"]["mean"]
    std  = cfg["data"]["std"]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    val_set = torchvision.datasets.CIFAR10(
        root=cfg["data"]["root"], train=False, download=True, transform=transform
    )

    # Find one image of the target class
    img_tensor = None
    for img, lbl in val_set:
        if lbl == class_idx:
            img_tensor = img
            break

    save_dir = cfg["paths"]["sanity_checks"]

    for model_name in models:
        ckpt_path = f"{cfg['models']['save_dir']}/{model_name}_best.pth"
        if not Path(ckpt_path).exists():
            logger.warning(f"  No checkpoint for {model_name} — skipping sanity")
            continue

        model = build_model(model_name, cfg).to(device)
        load_checkpoint(ckpt_path, model, device=device)
        model.eval()

        results = []
        logger.info(f"\n  ▶ {model_name}")

        if test in ("model_randomization", "both"):
            r = test_model_randomization(
                model=model, model_name=model_name,
                img_tensor=img_tensor, target_class=class_idx,
                cfg=cfg, device=device, save_dir=save_dir,
            )
            results.append(r)

        if test in ("data_randomization", "both"):
            r = test_data_randomization(
                trained_model=model, model_name=model_name,
                img_tensor=img_tensor, target_class=class_idx,
                cfg=cfg, device=device, save_dir=save_dir,
                n_shuffle_epochs=shuffle_epochs,
            )
            results.append(r)

        print_summary(results)


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Grad-CAM — Master Experiment Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                               # Full pipeline, all models
  python main.py --stages data train           # Data checks + training only
  python main.py --stages eval gradcam         # Evaluate + heatmaps
  python main.py --stages sanity --sanity-test model_randomization
  python main.py --models resnet18_pretrained  # One model, all stages
  python main.py --stages train --epochs 50   # Quick training run
        """
    )
    parser.add_argument(
        "--stages", nargs="+", default=ALL_STAGES,
        choices=ALL_STAGES,
        metavar="STAGE",
        help=f"Stages to run (default: all). Choices: {ALL_STAGES}"
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help=f"Model variants to use (default: all). Choices: {ALL_MODELS}"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs from config (useful for quick tests)"
    )
    parser.add_argument(
        "--sanity-test", type=str, default="both",
        choices=["model_randomization", "data_randomization", "both"],
        help="Which Adebayo test(s) to run in sanity stage"
    )
    parser.add_argument(
        "--sanity-class", type=int, default=0,
        help="CIFAR-10 class index for sanity check image (0=airplane)"
    )
    parser.add_argument(
        "--shuffle-epochs", type=int, default=10,
        help="Epochs for shuffled-label model in data randomization test"
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device()
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("CIFAR-10 Grad-CAM — Research Pipeline")
    logger.info(f"  Stages : {args.stages}")
    logger.info(f"  Models : {args.models}")
    logger.info(f"  Device : {device}")
    logger.info("=" * 60)

    t_total = time.time()

    if "data"   in args.stages:
        stage_data(cfg)

    if "train"  in args.stages:
        stage_train(cfg, args.models, epochs_override=args.epochs)

    if "eval"   in args.stages:
        stage_eval(cfg, args.models, device)

    if "gradcam" in args.stages:
        stage_gradcam(cfg, args.models, device)

    if "parity" in args.stages:
        stage_parity(cfg, args.models, device)

    if "sanity" in args.stages:
        stage_sanity(
            cfg, args.models, device,
            test=args.sanity_test,
            class_idx=args.sanity_class,
            shuffle_epochs=args.shuffle_epochs,
        )

    elapsed = (time.time() - t_total) / 60
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All stages complete. Total wall time: {elapsed:.1f} min")
    logger.info(f"Outputs written to: outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
