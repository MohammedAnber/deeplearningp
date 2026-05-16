# CIFAR-10 Grad-CAM Research

Grad-CAM implementation from scratch, trained on CIFAR-10 across three model variants.

## Models
- `baseline_cnn` — 3-layer custom CNN
- `resnet18_scratch` — ResNet18, random init
- `resnet18_pretrained` — ResNet18, ImageNet pretrained

## Quick Start
```bash
pip install -r requirements.txt
python main.py --stages data          # verify data pipeline
python main.py --stages train         # train all models
python main.py --stages eval gradcam  # evaluate + heatmaps
python main.py                        # full pipeline
```

## Project Structure
```
main.py            — master orchestrator
train.py           — training loop
models.py          — BaselineCNN + ResNet18 variants
data.py            — CIFAR-10 data pipeline
gradcam.py         — Grad-CAM from scratch
sanity_checks.py   — Adebayo randomization tests
utils.py           — shared helpers
configs/config.yaml
outputs/           — heatmaps, curves, eval plots
LOG.md             — research diary
```
=======
# deeplearningp

