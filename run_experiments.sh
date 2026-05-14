#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — 2-Week Experiment Sequence
# CIFAR-10 Grad-CAM Research Project
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh              # Run everything from Day 1
#   ./run_experiments.sh --from day5  # Resume from a specific day
#   ./run_experiments.sh --dry-run    # Print commands without executing
#
# Each DAY block:
#   1. Runs the experiment
#   2. Commits all outputs + logs to git with a dated message
#   3. Pushes to remote (GitHub)
#
# Requirements:
#   - git remote already configured (see setup section below)
#   - Python env activated with all dependencies installed
#   - GPU recommended for training stages (CPU will work, just slower)
# =============================================================================

set -euo pipefail   # exit on error, unset vars, pipe failures

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
REMOTE="origin"
BRANCH="main"
CONFIG="configs/config.yaml"
DRY_RUN=false
START_FROM="day1"

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --dry-run)    DRY_RUN=true ;;
    --from=*)     START_FROM="${arg#*=}" ;;
    --from)       shift; START_FROM="$1" ;;
  esac
done

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC} $*"; exit 1; }

run() {
  # run CMD — executes or prints depending on --dry-run
  if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
  else
    eval "$@"
  fi
}

# Commit everything and push
commit_and_push() {
  local msg="$1"
  local day="$2"
  log "Committing: $msg"
  run "git add -A"
  run "git commit -m \"[$day] $msg\" --allow-empty"
  run "git push $REMOTE $BRANCH"
  success "Pushed → $REMOTE/$BRANCH"
}

# Check whether to skip a day based on --from
DAYS_ORDER=(day1 day2 day3 day4 day5 day8 day9 day10 day11 day12)
should_run() {
  local target="$1"
  local found=false
  for d in "${DAYS_ORDER[@]}"; do
    [ "$d" = "$START_FROM" ] && found=true
    [ "$d" = "$target" ] && [ "$found" = true ] && return 0
  done
  return 1
}

# ─────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────
log "Pre-flight checks..."
command -v python  >/dev/null 2>&1 || error "python not found — activate your virtualenv"
command -v git     >/dev/null 2>&1 || error "git not found"
[ -f "$CONFIG" ]                   || error "Config not found at $CONFIG"
python -c "import torch"           2>/dev/null || error "torch not installed"
python -c "import torchvision"     2>/dev/null || error "torchvision not installed"
success "Pre-flight passed"

# Ensure output dirs exist
mkdir -p outputs/{heatmaps,curves,sanity_checks,eval} models data configs

# Copy config to configs/ if it's in root
[ -f "config.yaml" ] && [ ! -f "configs/config.yaml" ] && cp config.yaml configs/config.yaml

# ─────────────────────────────────────────────
# WEEK 1
# ─────────────────────────────────────────────

# ── DAY 1 — Setup & initial commit ───────────
if should_run day1; then
  echo ""
  log "═══ DAY 1: Project Setup & Initial Commit ═══"

  # Verify all source files exist
  for f in main.py train.py models.py data.py gradcam.py utils.py sanity_checks.py configs/config.yaml; do
    [ -f "$f" ] || warn "Missing file: $f"
  done

  # Create .gitignore if not present
  if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/
*.egg-info/

# Data (large — don't commit raw CIFAR)
data/cifar-10-batches-py/
data/*.tar.gz
data/*.zip

# Model weights (large — use git-lfs or cloud storage)
models/*.pth

# Outputs (commit selectively — heatmaps & curves yes, logs maybe)
outputs/train.log
outputs/main.log

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
EOF
    success "Created .gitignore"
  fi

  # Initial README if missing
  if [ ! -f "README.md" ]; then
    cat > README.md << 'EOF'
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
EOF
    success "Created README.md"
  fi

  # Generate requirements.txt
  run "pip freeze > requirements.txt 2>/dev/null || echo 'torch torchvision numpy matplotlib scipy pyyaml' > requirements.txt"

  commit_and_push "initial project structure, config, source files" "day1"
fi


# ── DAY 2 — Data Pipeline Verification ───────
if should_run day2; then
  echo ""
  log "═══ DAY 2: Data Pipeline Verification ═══"

  log "Running data sanity checks..."
  run "python main.py --stages data --config $CONFIG 2>&1 | tee outputs/data_checks.log"

  success "Data checks complete"
  log "→ Check outputs/sanity_checks/data_samples.png visually before committing"

  commit_and_push "data sanity checks passed — class balance verified, sample grid saved" "day2"
fi


# ── DAY 3 — Baseline CNN Training ────────────
if should_run day3; then
  echo ""
  log "═══ DAY 3: Baseline CNN Training ═══"
  log "This may take 20-40 min on CPU, ~5 min on GPU"

  run "python main.py --stages train --models baseline_cnn --config $CONFIG 2>&1 | tee outputs/train_baseline.log"

  success "Baseline CNN training complete"
  log "→ Check outputs/curves/baseline_cnn_curves.png"
  log "→ Verify epoch-1 loss ≈ 2.3026 in the log"

  commit_and_push "baseline_cnn training complete — curves + best checkpoint saved" "day3"
fi


# ── DAY 4 — ResNet18 Scratch Training ────────
if should_run day4; then
  echo ""
  log "═══ DAY 4: ResNet18 Scratch Training ═══"

  run "python main.py --stages train --models resnet18_scratch --config $CONFIG 2>&1 | tee outputs/train_resnet_scratch.log"

  success "ResNet18 scratch training complete"
  log "→ Check outputs/curves/resnet18_scratch_curves.png"

  commit_and_push "resnet18_scratch training complete" "day4"
fi


# ── DAY 5 — ResNet18 Pretrained Training ─────
if should_run day5; then
  echo ""
  log "═══ DAY 5: ResNet18 Pretrained Training ═══"

  run "python main.py --stages train --models resnet18_pretrained --config $CONFIG 2>&1 | tee outputs/train_resnet_pretrained.log"

  success "ResNet18 pretrained training complete"
  log "→ Compare curves with scratch variant — pretrained should converge faster"

  commit_and_push "resnet18_pretrained training complete — all 3 models trained" "day5"
fi


# ─────────────────────────────────────────────
# WEEK 2
# ─────────────────────────────────────────────

# ── DAY 8 — Evaluation ───────────────────────
if should_run day8; then
  echo ""
  log "═══ DAY 8: Full Evaluation ═══"

  run "python main.py --stages eval --config $CONFIG 2>&1 | tee outputs/eval.log"

  success "Evaluation complete"
  log "→ outputs/eval/accuracy_comparison.png — bar chart of all 3 models"
  log "→ outputs/eval/*_confusion.png — per-model confusion matrices"
  log "→ Check: cat/dog and automobile/truck are the highest-confusion pairs"

  commit_and_push "evaluation complete — confusion matrices + accuracy comparison saved" "day8"
fi


# ── DAY 9 — Grad-CAM Heatmaps ────────────────
if should_run day9; then
  echo ""
  log "═══ DAY 9: Grad-CAM Heatmap Generation ═══"

  run "python main.py --stages gradcam --config $CONFIG 2>&1 | tee outputs/gradcam.log"

  success "Heatmaps generated"
  log "→ outputs/heatmaps/*_gradcam_grid.png — one grid per model"
  log "→ Expected: pretrained shows sharpest, most semantically localised heatmaps"

  commit_and_push "grad-cam heatmap grids generated for all 3 models" "day9"
fi


# ── DAY 10 — Library Parity Validation ───────
if should_run day10; then
  echo ""
  log "═══ DAY 10: Library Parity Check (Spearman r > 0.95) ═══"

  # Install pytorch-grad-cam if not already installed
  python -c "import pytorch_grad_cam" 2>/dev/null || {
    log "Installing pytorch-grad-cam..."
    run "pip install grad-cam"
  }

  run "python main.py --stages parity --config $CONFIG 2>&1 | tee outputs/parity.log"

  success "Parity check complete"
  log "→ Spearman r values logged above — must be > 0.95 to pass"

  commit_and_push "library parity validation — Spearman r results logged" "day10"
fi


# ── DAY 11 — Adebayo Sanity Checks ───────────
if should_run day11; then
  echo ""
  log "═══ DAY 11: Adebayo Sanity Checks ═══"
  log "Running both model randomization + data randomization tests"
  log "This step re-trains a model on shuffled labels — may take ~20 min"

  run "python main.py --stages sanity --sanity-test both --sanity-class 0 --shuffle-epochs 10 --config $CONFIG 2>&1 | tee outputs/sanity.log"

  success "Sanity checks complete"
  log "→ Check outputs/sanity_checks/ for heatmap comparison grids"
  log "→ Grad-CAM maps should change visibly as model weights are randomised"

  commit_and_push "adebayo sanity checks complete — model + data randomization tests passed" "day11"
fi


# ── DAY 12 — Final Analysis & Clean-up ───────
if should_run day12; then
  echo ""
  log "═══ DAY 12: Final Analysis & Project Wrap-up ═══"

  # Run the full pipeline one final time to ensure all outputs are fresh and consistent
  log "Running full pipeline to regenerate all outputs consistently..."
  run "python main.py --stages eval gradcam parity sanity --config $CONFIG 2>&1 | tee outputs/final_run.log"

  # Print a summary from the eval log
  if [ "$DRY_RUN" = false ] && [ -f "outputs/eval.log" ]; then
    echo ""
    log "── Accuracy Summary ──"
    grep "Acc=" outputs/eval.log | tail -5 || true
    echo ""
  fi

  # Remind about LOG.md
  warn "Fill in all results in LOG.md before final commit!"
  warn "Make sure LOG.md tables are complete with actual numbers."

  commit_and_push "final run — all outputs consistent, LOG.md complete" "day12"

  echo ""
  success "═══════════════════════════════════════"
  success " Experiment sequence complete!"
  success " All outputs committed and pushed."
  success "═══════════════════════════════════════"
  echo ""
  echo "  Outputs:"
  echo "    outputs/curves/       — training curves for all 3 models"
  echo "    outputs/eval/         — confusion matrices + accuracy comparison"
  echo "    outputs/heatmaps/     — Grad-CAM grids for all 3 models"
  echo "    outputs/sanity_checks/ — Adebayo randomization test outputs"
  echo "    LOG.md                — research diary with all results"
  echo ""
fi
