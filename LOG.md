# Research Log — CIFAR-10 Grad-CAM Project


## Project Overview

**Goal:** Implement Grad-CAM from scratch, train three CIFAR-10 model variants (BaselineCNN, ResNet18-scratch, ResNet18-pretrained), validate the implementation against the `pytorch-grad-cam` library (Spearman r > 0.95), and analyse what the heatmaps reveal about each model's learned representations.

**Models:**
- `baseline_cnn` — 3-layer custom CNN, trained from scratch
- `resnet18_scratch` — ResNet18 architecture, random init
- `resnet18_pretrained` — ResNet18, ImageNet weights → fine-tuned

**Stack:** Python 3.10+, PyTorch, torchvision, CIFAR-10



# — Project Setup & Repository Init

**What I did:**
- Created repo structure, added all source files
- Reviewed all modules: `data.py`, `models.py`, `train.py`, `gradcam.py`, `utils.py`, `sanity_checks.py`, `main.py`
- Confirmed config values in `config.yaml` (seed=42, lr=0.01, epochs=100, SGD+cosine)

**Key design decisions already baked in:**
- Normalization stats computed from training set only (no leakage)
- No vertical flip augmentation (objects don't appear upside-down in nature)
- Raw logit used in Grad-CAM backward pass, not softmax (avoids class competition)
- ReLU applied after weighted feature map sum (only positive contributions matter)

**Status:** ✅ Codebase understood. Ready to run data checks.

---

# — Data Pipeline Verification

**Command run:**
```bash
python main.py --stages data
```

**What to check:**
- [ ] Class balance: exactly 5000 train / 1000 test per class
- [ ] Sample grid saved to `outputs/sanity_checks/data_samples.png` — visually inspect labels
- [ ] Augmentation looks reasonable (random crops, horizontal flips, no distortion)

**Expected log output:**
```
[train] Class counts: airplane: 5000, automobile: 5000, ...
[train] ✓ Class balance verified (5000 per class)
[test]  ✓ Class balance verified (1000 per class)
[sanity] Sample grid saved → outputs/sanity_checks/data_samples.png
```

**Results:** _(fill in after running)_

**Decision log:**
- Using full 10k test set as validation (standard CIFAR-10 academic convention)
- `num_workers=4` in config — reduce to 0 if multiprocessing errors on Windows

---

# — Baseline CNN Training

**Command run:**
```bash
python main.py --stages train --models baseline_cnn
```

**Epoch-1 loss sanity check (critical):**
Expected: `~2.3026` (= ln(10), cross-entropy for uniform 10-class random output)
- If loss >> 2.3: weight init problem or NaN in data pipeline
- If loss << 2.3: data leakage or model already loaded trained weights

**Results:** _(fill in)_

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|-----------|----------|---------|---------|
| 1     |           |          |         |         |
| 50    |           |          |         |         |
| 100   |           |          |         |         |

**Best val accuracy:** _%

**Observations:** _(overfitting? underfitting? when did val acc plateau?)_

---

# — ResNet18 Scratch Training

**Command run:**
```bash
python main.py --stages train --models resnet18_scratch
```

**Key architecture difference from ImageNet ResNet18:**
- First conv: 7×7 stride-2 → **3×3 stride-1** (32×32 images can't afford spatial shrinkage)
- Initial MaxPool: removed (replaced with `nn.Identity()`)
- Final FC: 1000 → **10 classes**

**Results:** _(fill in)_

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|-----------|----------|---------|---------|
| 1     |           |          |         |         |
| 50    |           |          |         |         |
| 100   |           |          |         |         |

**Best val accuracy:** _%

**Observations:**

---

# — ResNet18 Pretrained Training

**Command run:**
```bash
python main.py --stages train --models resnet18_pretrained
```

**Hypothesis:** Should converge faster and reach higher accuracy than scratch variant.
Transfer learning hypothesis: ImageNet features (edges, textures, shapes) are reusable for CIFAR-10.

**Results:** _(fill in)_

| Epoch | Train Loss | Val Loss | Val Acc | Gen Gap |
|-------|-----------|----------|---------|---------|
| 1     |           |          |         |         |
| 50    |           |          |         |         |
| 100   |           |          |         |         |

**Best val accuracy:** _%

**Pretrained vs Scratch gap:** _% (pretrained higher by this much)

**Observations:**


# — Evaluation & Accuracy Comparison

**Command run:**
```bash
python main.py --stages eval
```

**Outputs generated:**
- `outputs/eval/baseline_cnn_confusion.png`
- `outputs/eval/resnet18_scratch_confusion.png`
- `outputs/eval/resnet18_pretrained_confusion.png`
- `outputs/eval/accuracy_comparison.png`

**Final accuracy table:**

| Model | Test Accuracy | Avg Loss | Best Epoch |
|-------|--------------|----------|------------|
| baseline_cnn | | | |
| resnet18_scratch | | | |
| resnet18_pretrained | | | |

**Confusion matrix observations:**
- Which classes are most confused with each other?
- Cat/dog confusion expected (visually similar)
- Automobile/truck confusion expected (same category family)

**Key finding:** _(fill in after running)_

---

# — Grad-CAM Heatmap Generation

**Command run:**
```bash
python main.py --stages gradcam
```

**Outputs:** `outputs/heatmaps/{model_name}_gradcam_grid.png` for all 3 models

**What to analyse in heatmaps:**

| Model | Expected heatmap quality | Reason |
|-------|--------------------------|--------|
| baseline_cnn | Coarse, diffuse | Shallow features, 4×4 final feature map |
| resnet18_scratch | Moderate localisation | Deeper hierarchy, but no prior knowledge |
| resnet18_pretrained | Sharp, semantically meaningful | ImageNet priors encode object structure |

**Observations per class:** _(fill in after visual inspection)_

- airplane: Does the map focus on fuselage/wings?
- cat/dog: Does it activate on the animal body vs background?
- automobile: Does it highlight the vehicle shape?

**Surprising findings:** _(note anything unexpected)_

---

# — Library Parity Validation

**Command run:**
```bash
pip install grad-cam
python main.py --stages parity
```

**Method:** Spearman rank correlation between our Grad-CAM and `pytorch-grad-cam` library output on the same image/model/class.

**Threshold:** r > 0.95 (set in `config.yaml → evaluation.library_parity_threshold`)

**Results:**

| Model | Spearman r | p-value | Pass? |
|-------|-----------|---------|-------|
| baseline_cnn | | | |
| resnet18_scratch | | | |
| resnet18_pretrained | | | |

**If r < 0.95 — debug checklist:**
1. Wrong target layer selected?
2. `register_full_backward_hook` vs `register_backward_hook` mismatch?
3. Gradients averaged over wrong dimensions?
4. ReLU applied at wrong step?

**Conclusion:** _(fill in — does our implementation match the reference?)_

---

# — Adebayo Sanity Checks

**What these tests prove:**
Adebayo et al. (2018) "Sanity Checks for Saliency Maps" showed that many saliency methods produce visually plausible maps even on randomly initialised models — which means they may be measuring data statistics, not model internals.

We run both tests to verify our Grad-CAM is actually sensitive to the model:

**Test 1 — Model Randomization:** Progressively randomise model weights layer by layer (top→down). If Grad-CAM is meaningful, maps should change as weights are randomised.

**Test 2 — Data Randomization:** Re-train model on shuffled labels. If maps look identical to a properly-trained model, they're not capturing learned representations.

**Command run:**
```bash
python main.py --stages sanity --sanity-test both --sanity-class 0
```

**Results:**

| Model | Model Rand. Result | Data Rand. Result |
|-------|-------------------|-------------------|
| baseline_cnn | | |
| resnet18_scratch | | |
| resnet18_pretrained | | |

**Interpretation:** _(do maps change meaningfully? what does this imply?)_

---

# — Analysis & Write-up

**Key questions to answer in report:**

1. Does pretrained ResNet18 produce more interpretable Grad-CAM maps than scratch? Why?
2. What do the heatmaps reveal about what each model has learned to look at?
3. Does our Grad-CAM implementation satisfy the Adebayo sanity checks?
4. How does accuracy correlate with heatmap quality?

**Generalization gap summary across models:**

| Model | Train Acc | Val Acc | Gap (overfit) |
|-------|-----------|---------|---------------|
| baseline_cnn | | | |
| resnet18_scratch | | | |
| resnet18_pretrained | | | |




## Decisions Log

| Decision | Reason |
|----------|--------|
| seed=42 everywhere | Full reproducibility |
| SGD not Adam | Adam overfits more on CIFAR-10 per literature |
| Cosine annealing LR | Avoids abrupt drops; smooth convergence |
| Raw logit in Grad-CAM backward | Avoids softmax class competition |
| No vertical flip | Objects don't appear upside-down in natural images |
| | |

---

## References

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Adebayo et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS.
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Krizhevsky (2009). *Learning Multiple Layers of Features from Tiny Images.* (CIFAR-10 dataset paper)
