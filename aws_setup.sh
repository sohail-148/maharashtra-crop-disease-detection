#!/bin/bash
# =============================================================================
# aws_setup.sh
# Bootstrap script for AWS g4dn.xlarge (Deep Learning AMI — Ubuntu)
#
# Tested AMI: "AWS Deep Learning AMI GPU TensorFlow 2.x (Ubuntu 22.04)"
# - Find it in EC2 launch wizard under "AWS Marketplace AMIs"
# - Search: "Deep Learning AMI GPU TensorFlow"
# - This AMI ships TensorFlow 2.x inside a conda environment named
#   "tensorflow2_p310" (Python 3.10). Do NOT use the system Python.
#
# Run this ONCE immediately after SSHing into a fresh instance:
#   chmod +x aws_setup.sh
#   ./aws_setup.sh
#
# BEFORE running this script, upload your datasets from your local machine:
#   (see aws_training_plan.md Step 3 for scp commands)
#
# After this script completes, run:
#   ./run_all_experiments.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP $(date '+%H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[SETUP $(date '+%H:%M:%S')] WARNING: $*${NC}"; }
fail() { echo -e "${RED}[SETUP $(date '+%H:%M:%S')] ERROR: $*${NC}"; exit 1; }

# =============================================================================
# CONFIGURATION — edit if needed
# =============================================================================

GITHUB_REPO="https://github.com/sohail-148/maharashtra-crop-disease-detection.git"
PROJECT_DIR="/home/ubuntu/maharashtra-crop-disease-detection"
DATASET_DIR="/data/datasets"

# The conda environment name on the AWS Deep Learning AMI (Ubuntu 22.04).
# Run `conda env list` on the instance to confirm the exact name.
CONDA_ENV="tensorflow2_p310"

# =============================================================================
# STEP 1 — Activate the correct conda environment
# =============================================================================
log "Step 1: Activating conda environment: $CONDA_ENV"

# Source conda so it is available in this non-interactive shell
CONDA_BASE=$(conda info --base 2>/dev/null) || fail "conda not found. Is this the AWS Deep Learning AMI?"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Confirm Python and TF are visible
PYTHON=$(which python)
log "  Python : $PYTHON"
python --version

python -c "import tensorflow as tf; print('  TensorFlow:', tf.__version__)" \
    || fail "TensorFlow not found in conda env '$CONDA_ENV'. Check AMI and env name."

# Confirm GPU
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: print(f'  GPU: {g}')
    print('  TensorFlow GPU ready.')
else:
    print('  WARNING: No GPU detected. Check CUDA / instance type.')
"

# =============================================================================
# STEP 2 — System packages
# =============================================================================
log "Step 2: Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q git libgl1-mesa-glx libglib2.0-0

# =============================================================================
# STEP 3 — Python packages (into the active conda env via pip)
# =============================================================================
log "Step 3: Installing/upgrading Python packages..."

# Install into the conda env using its pip.
# Do NOT use --upgrade on existing packages to avoid breaking TF dependencies.
pip install --quiet \
    "scikit-learn>=1.3,<2.0" \
    "pandas>=2.0" \
    "matplotlib>=3.7" \
    "Pillow>=10.0" \
    "opencv-python-headless>=4.8"

python -c "
import sklearn, pandas, matplotlib, PIL, cv2
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  matplotlib   : {matplotlib.__version__}')
print(f'  Pillow       : {PIL.__version__}')
print(f'  opencv       : {cv2.__version__}')
"

# =============================================================================
# STEP 4 — Create dataset directory
# =============================================================================
log "Step 4: Creating dataset directory: $DATASET_DIR"

# This must exist BEFORE datasets are uploaded via scp.
# If you are running this script after the scp upload, the directory
# already exists and this is a no-op.
sudo mkdir -p "$DATASET_DIR"
sudo chown ubuntu:ubuntu "$DATASET_DIR"
log "  $DATASET_DIR is ready."

# =============================================================================
# STEP 5 — Clone project repository
# =============================================================================
log "Step 5: Cloning project repository..."

if [ -d "$PROJECT_DIR/.git" ]; then
    warn "Repository already exists — pulling latest changes."
    git -C "$PROJECT_DIR" pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
fi

log "  Repository ready at: $PROJECT_DIR"
cd "$PROJECT_DIR"

# =============================================================================
# STEP 6 — Verify datasets are present
# =============================================================================
log "Step 6: Verifying datasets..."

DATASETS="grape_niphad grape_2024 chilli_cold sugarcane_maharashtra sugarcane_large"
ALL_OK=true

for ds in $DATASETS; do
    ds_path="$DATASET_DIR/$ds"
    if [ -d "$ds_path" ] && [ "$(ls -A "$ds_path" 2>/dev/null)" ]; then
        count=$(find "$ds_path" -type f | wc -l)
        log "  OK: $ds_path  ($count files)"
    else
        warn "  MISSING or EMPTY: $ds_path"
        warn "  Upload it from your local machine:"
        warn "    scp -i your-key.pem -r /path/to/$ds ubuntu@<ip>:$DATASET_DIR/"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    warn "One or more datasets are missing."
    warn "Upload them and re-run this step, or proceed and experiments will fail."
fi

# =============================================================================
# STEP 7 — Fix split CSV paths for Linux
# =============================================================================
log "Step 7: Rewriting Windows paths in split CSVs..."

python fix_split_paths.py --dataset-root "$DATASET_DIR"

log "  Split paths updated."

# =============================================================================
# STEP 8 — Verify split CSVs and spot-check a path
# =============================================================================
log "Step 8: Spot-checking rewritten paths..."

python - <<'PYEOF'
import pandas as pd, os, sys

datasets = {
    "grape_niphad":         "splits/grape_niphad/train.csv",
    "grape_2024":           "splits/grape_2024/train.csv",
    "chilli_cold":          "splits/chilli_cold/train.csv",
    "sugarcane_maharashtra":"splits/sugarcane_maharashtra/train.csv",
    "sugarcane_large":      "splits/sugarcane_large/train.csv",
}

all_ok = True
for name, csv_path in datasets.items():
    if not os.path.exists(csv_path):
        print(f"  MISSING CSV: {csv_path}")
        all_ok = False
        continue
    df = pd.read_csv(csv_path)
    sample = df["file_path"].iloc[0]
    exists = os.path.exists(sample)
    status = "OK  " if exists else "MISS"
    print(f"  [{status}] {name}: {sample}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n  Some paths are missing. Check dataset upload and --dataset-root.")
    sys.exit(1)
else:
    print("\n  All sample paths verified.")
PYEOF

# =============================================================================
# STEP 9 — Make scripts executable
# =============================================================================
log "Step 9: Setting script permissions..."
chmod +x run_all_experiments.sh aws_setup.sh

# =============================================================================
# Done
# =============================================================================
echo ""
log "=============================================================="
log "  Setup complete."
log "  Conda env : $CONDA_ENV"
log "  Python    : $(which python)"
log "  Project   : $PROJECT_DIR"
log "  Datasets  : $DATASET_DIR"
log ""
log "  To train all 5 experiments (~25-35 min on g4dn.xlarge):"
log "    cd $PROJECT_DIR"
log "    ./run_all_experiments.sh"
log ""
log "  To run a single experiment:"
log "    python train_experiment.py --experiment G1 \\"
log "           --dataset-root $DATASET_DIR --batch-size 64"
log ""
log "  To download results back to your local machine:"
log "    scp -i your-key.pem -r ubuntu@<ip>:$PROJECT_DIR/results ./results_aws"
log "    scp -i your-key.pem -r ubuntu@<ip>:$PROJECT_DIR/models  ./models_aws"
log ""
log "  IMPORTANT: Terminate the instance when done to stop billing."
log "  EC2 Console -> Instances -> select -> Instance State -> Terminate"
log "=============================================================="
