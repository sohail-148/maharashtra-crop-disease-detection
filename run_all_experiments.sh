#!/bin/bash
# =============================================================================
# run_all_experiments.sh
# Sequential runner for G1, G2, C1, S1, S2 MobileNetV2 baseline experiments
#
# Usage on AWS g4dn.xlarge (Deep Learning AMI):
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Override dataset root if datasets are not in the default location:
#   DATASET_ROOT=/data/datasets ./run_all_experiments.sh
#
# Override batch size (default 64 for GPU; drop to 32 if OOM):
#   BATCH_SIZE=32 ./run_all_experiments.sh
#
# Run a single experiment instead of all five:
#   EXPERIMENTS="G1" ./run_all_experiments.sh
# =============================================================================

set -euo pipefail   # exit on any error

# --------------------------------------------------------------------------
# Configuration — override via environment variables
# --------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EXPERIMENTS="${EXPERIMENTS:-G1 G2 C1 S1 S2}"

# Conda environment on the AWS Deep Learning AMI.
# Override with: CONDA_ENV=my_env ./run_all_experiments.sh
CONDA_ENV="${CONDA_ENV:-tensorflow2_p310}"

# --------------------------------------------------------------------------
# Colours for readability
# --------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Colour

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $*${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAILED: $*${NC}"; exit 1; }

# --------------------------------------------------------------------------
# Activate conda environment
# --------------------------------------------------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null) || fail "conda not found. Is this the AWS Deep Learning AMI?"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PYTHON=$(which python)

# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------
log "=== AWS GPU Training Run ==="
log "Project dir   : $PROJECT_DIR"
log "Dataset root  : $DATASET_ROOT"
log "Batch size    : $BATCH_SIZE"
log "Experiments   : $EXPERIMENTS"
log "Conda env     : $CONDA_ENV"
log "Python        : $PYTHON"
echo ""

cd "$PROJECT_DIR"

# Check Python
$PYTHON --version || fail "Python not found. Set PYTHON= env var."

# Check TensorFlow GPU visibility
log "Checking GPU availability..."
$PYTHON -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  GPU detected: {gpus}')
else:
    print('  WARNING: No GPU detected — training will be slow on CPU.')
"

# Check split CSVs exist
for exp in $EXPERIMENTS; do
    case $exp in
        G1) dir="grape_niphad" ;;
        G2) dir="grape_2024" ;;
        C1) dir="chilli_cold" ;;
        S1) dir="sugarcane_maharashtra" ;;
        S2) dir="sugarcane_large" ;;
    esac
    csv="splits/$dir/train.csv"
    [ -f "$csv" ] || fail "Split CSV not found: $csv"
done
log "All split CSVs found."

# Check dataset root
[ -d "$DATASET_ROOT" ] || fail "Dataset root does not exist: $DATASET_ROOT"
log "Dataset root exists."

echo ""

# --------------------------------------------------------------------------
# Summary log file
# --------------------------------------------------------------------------
SUMMARY_FILE="$PROJECT_DIR/results/aws_run_summary.txt"
mkdir -p "$PROJECT_DIR/results"
echo "AWS GPU Training Run — $(date)" > "$SUMMARY_FILE"
echo "Experiments: $EXPERIMENTS" >> "$SUMMARY_FILE"
echo "Dataset root: $DATASET_ROOT" >> "$SUMMARY_FILE"
echo "Batch size: $BATCH_SIZE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# --------------------------------------------------------------------------
# Run experiments sequentially
# --------------------------------------------------------------------------
OVERALL_START=$(date +%s)
FAILED_EXPS=""

for EXP in $EXPERIMENTS; do
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Starting experiment: $EXP"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EXP_START=$(date +%s)

    if $PYTHON train_experiment.py \
        --experiment "$EXP" \
        --dataset-root "$DATASET_ROOT" \
        --batch-size "$BATCH_SIZE"; then

        EXP_END=$(date +%s)
        EXP_MINS=$(( (EXP_END - EXP_START) / 60 ))
        EXP_SECS=$(( (EXP_END - EXP_START) % 60 ))
        log "Experiment $EXP DONE in ${EXP_MINS}m ${EXP_SECS}s"
        echo "$EXP: DONE in ${EXP_MINS}m ${EXP_SECS}s" >> "$SUMMARY_FILE"
    else
        warn "Experiment $EXP FAILED — continuing with remaining experiments."
        echo "$EXP: FAILED" >> "$SUMMARY_FILE"
        FAILED_EXPS="$FAILED_EXPS $EXP"
    fi

    echo ""
done

# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
OVERALL_END=$(date +%s)
TOTAL_MINS=$(( (OVERALL_END - OVERALL_START) / 60 ))
TOTAL_SECS=$(( (OVERALL_END - OVERALL_START) % 60 ))

echo "" >> "$SUMMARY_FILE"
echo "Total wall-clock time: ${TOTAL_MINS}m ${TOTAL_SECS}s" >> "$SUMMARY_FILE"

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ALL EXPERIMENTS COMPLETE"
log "Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
log "Summary written to: $SUMMARY_FILE"

if [ -n "$FAILED_EXPS" ]; then
    warn "Failed experiments:$FAILED_EXPS"
    exit 1
else
    log "All experiments succeeded."
fi
