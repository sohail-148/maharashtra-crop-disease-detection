#!/bin/bash
# =============================================================================
# aws_setup.sh
# Bootstrap script for AWS g4dn.xlarge (Deep Learning AMI — Ubuntu)
#
# Run this ONCE immediately after SSHing into a fresh instance.
# It installs dependencies, uploads your project files from GitHub,
# downloads datasets, and prepares everything so run_all_experiments.sh
# can start immediately.
#
# Assumed AMI: AWS Deep Learning AMI GPU TensorFlow 2.x (Ubuntu 20.04/22.04)
# These AMIs ship with: Python 3.10+, TensorFlow 2.x, CUDA, cuDNN pre-installed.
# No conda environment needed — use the system Python.
#
# Usage:
#   chmod +x aws_setup.sh
#   ./aws_setup.sh
#
# After completion, run:
#   ./run_all_experiments.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP $(date '+%H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[SETUP $(date '+%H:%M:%S')] WARNING: $*${NC}"; }

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

GITHUB_REPO="https://github.com/sohail-148/maharashtra-crop-disease-detection.git"
PROJECT_DIR="/home/ubuntu/maharashtra-crop-disease-detection"
DATASET_DIR="/data/datasets"

# Kaggle credentials — set these or export before running this script.
# You can also place kaggle.json at ~/.kaggle/kaggle.json manually.
KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"
KAGGLE_KEY="${KAGGLE_KEY:-}"

# =============================================================================
# STEP 1 — System packages
# =============================================================================
log "Step 1: Updating system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
    git curl wget unzip p7zip-full \
    libgl1-mesa-glx libglib2.0-0   # required by OpenCV

# =============================================================================
# STEP 2 — Python packages
# =============================================================================
log "Step 2: Installing Python packages..."

# The Deep Learning AMI includes TensorFlow — just add missing packages.
pip install --quiet --upgrade \
    scikit-learn==1.5.2 \
    pandas \
    matplotlib \
    Pillow \
    opencv-python-headless \
    kaggle

log "Python packages ready."

# =============================================================================
# STEP 3 — Clone project repository
# =============================================================================
log "Step 3: Cloning project repository..."

if [ -d "$PROJECT_DIR" ]; then
    warn "Project directory already exists — pulling latest changes."
    cd "$PROJECT_DIR" && git pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
log "Repository ready at: $PROJECT_DIR"

# =============================================================================
# STEP 4 — Prepare dataset directory
# =============================================================================
log "Step 4: Creating dataset directory..."
sudo mkdir -p "$DATASET_DIR"
sudo chown ubuntu:ubuntu "$DATASET_DIR"

# =============================================================================
# STEP 5 — Download datasets via Kaggle CLI
# =============================================================================
log "Step 5: Setting up Kaggle credentials..."

mkdir -p ~/.kaggle

if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    log "Kaggle credentials written from environment variables."
elif [ -f ~/.kaggle/kaggle.json ]; then
    log "Kaggle credentials already present at ~/.kaggle/kaggle.json"
else
    warn "No Kaggle credentials found."
    warn "Either:"
    warn "  1. Export KAGGLE_USERNAME and KAGGLE_KEY before running this script, OR"
    warn "  2. Upload kaggle.json manually: scp kaggle.json ubuntu@<ip>:~/.kaggle/"
    warn "Skipping dataset download. Upload datasets manually to $DATASET_DIR."
    warn "Expected structure:"
    warn "  $DATASET_DIR/grape_niphad/"
    warn "  $DATASET_DIR/grape_2024/"
    warn "  $DATASET_DIR/chilli_cold/"
    warn "  $DATASET_DIR/sugarcane_maharashtra/"
    warn "  $DATASET_DIR/sugarcane_large/"
fi

# ---- Dataset download function ----
download_dataset() {
    local dataset_slug="$1"    # e.g. "user/dataset-name"
    local target_dir="$2"      # e.g. /data/datasets/grape_niphad
    local zip_name="$3"        # zip filename Kaggle will produce

    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
        log "  Dataset already exists: $target_dir  (skipping download)"
        return 0
    fi

    log "  Downloading: $dataset_slug → $target_dir"
    mkdir -p "$target_dir"
    cd /tmp

    kaggle datasets download -d "$dataset_slug" --quiet

    log "  Extracting $zip_name..."
    unzip -q "$zip_name" -d "$target_dir"
    rm -f "$zip_name"

    log "  Done: $target_dir"
    cd "$PROJECT_DIR"
}

if [ -f ~/.kaggle/kaggle.json ]; then
    log "Downloading datasets..."

    # NOTE: Replace the Kaggle dataset slugs below with the exact ones
    # from the dataset URLs on kaggle.com.
    # Format: owner/dataset-name  (from the URL: kaggle.com/datasets/owner/dataset-name)
    #
    # These are placeholders — verify against the actual Kaggle dataset pages.

    download_dataset \
        "your-kaggle-user/grape-leaf-disease-niphad" \
        "$DATASET_DIR/grape_niphad" \
        "grape-leaf-disease-niphad.zip"

    download_dataset \
        "your-kaggle-user/grape-disease-2024" \
        "$DATASET_DIR/grape_2024" \
        "grape-disease-2024.zip"

    download_dataset \
        "your-kaggle-user/chilli-cold-2024" \
        "$DATASET_DIR/chilli_cold" \
        "chilli-cold-2024.zip"

    download_dataset \
        "your-kaggle-user/maharashtra-sugarcane-leaf-disease" \
        "$DATASET_DIR/sugarcane_maharashtra" \
        "maharashtra-sugarcane-leaf-disease.zip"

    download_dataset \
        "your-kaggle-user/sugarcane-leaf-image-dataset" \
        "$DATASET_DIR/sugarcane_large" \
        "sugarcane-leaf-image-dataset.zip"

    log "All datasets downloaded."
else
    warn "Skipped dataset downloads (no Kaggle credentials)."
    warn "Upload datasets manually before running experiments."
fi

# =============================================================================
# STEP 6 — Fix split CSV paths for Linux
# =============================================================================
log "Step 6: Rewriting Windows paths in split CSVs..."

cd "$PROJECT_DIR"
python fix_split_paths.py --dataset-root "$DATASET_DIR"

log "Split paths updated."

# =============================================================================
# STEP 7 — Verify GPU
# =============================================================================
log "Step 7: Verifying GPU visibility..."

python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        print(f'  Found GPU: {g}')
    print('  TensorFlow GPU ready.')
else:
    print('  WARNING: No GPU found. Check CUDA installation.')
"

# =============================================================================
# STEP 8 — Quick sanity check
# =============================================================================
log "Step 8: Sanity check — verifying files..."

for f in train_experiment.py run_all_experiments.sh fix_split_paths.py; do
    [ -f "$f" ] && log "  OK: $f" || warn "  MISSING: $f"
done

for exp in grape_niphad grape_2024 chilli_cold sugarcane_maharashtra sugarcane_large; do
    csv="splits/$exp/train.csv"
    [ -f "$csv" ] && log "  OK: $csv" || warn "  MISSING: $csv"
done

for ds in grape_niphad grape_2024 chilli_cold sugarcane_maharashtra sugarcane_large; do
    [ -d "$DATASET_DIR/$ds" ] && log "  OK: $DATASET_DIR/$ds" \
        || warn "  MISSING: $DATASET_DIR/$ds  — upload before running experiments"
done

# =============================================================================
# Done
# =============================================================================
echo ""
log "================================================================="
log "  Setup complete."
log "  Project : $PROJECT_DIR"
log "  Datasets: $DATASET_DIR"
log ""
log "  To start training all 5 experiments:"
log "    cd $PROJECT_DIR"
log "    chmod +x run_all_experiments.sh"
log "    ./run_all_experiments.sh"
log ""
log "  To run a single experiment:"
log "    python train_experiment.py --experiment G1 --dataset-root $DATASET_DIR"
log ""
log "  To download results after training:"
log "    scp -r ubuntu@<instance-ip>:$PROJECT_DIR/results ./results_aws"
log "    scp -r ubuntu@<instance-ip>:$PROJECT_DIR/models ./models_aws"
log "================================================================="
