# AWS GPU Training Plan — Experiments G1–S2

## Instance recommendation

| Field | Value |
|---|---|
| Instance type | `g4dn.xlarge` |
| GPU | NVIDIA T4 (16 GB VRAM) |
| vCPU | 4 |
| RAM | 16 GB |
| Region | ap-south-1 (Mumbai) |
| AMI | AWS Deep Learning AMI GPU TensorFlow 2.x (Ubuntu 22.04) |
| On-demand price | ~$0.58/hr (~Rs. 49/hr) |
| Spot price | ~$0.13–0.16/hr (~Rs. 11–14/hr) |

### Finding the correct AMI

In EC2 launch wizard:
1. Click "Browse more AMIs"
2. Select "AWS Marketplace AMIs"
3. Search: `Deep Learning AMI GPU TensorFlow`
4. Choose the **Ubuntu 22.04** variant with TensorFlow 2.x
5. The AMI name will look like:  
   `Deep Learning AMI GPU TensorFlow 2.16 (Ubuntu 22.04)`

### Important: this AMI uses conda, not system Python

TensorFlow is installed inside a conda environment called **`tensorflow2_p310`**
(Python 3.10). The system Python has no TensorFlow. All scripts in this project
activate this environment automatically. If you need to run commands manually,
activate it first:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tensorflow2_p310
python --version   # should show 3.10.x
```

To confirm the exact environment name on your instance:
```bash
conda env list
```

---

## Dataset sizes (measured from actual files)

| Exp | Dataset | Images | Disk size |
|---|---|---:|---:|
| G1 | grape_niphad | 2,727 | 24.8 MB |
| G2 | grape_2024 | 3,477 | 860.0 MB |
| C1 | chilli_cold | 1,932 | 60.8 MB |
| S1 | sugarcane_maharashtra | 2,521 | 159.9 MB |
| S2 | sugarcane_large | 6,405 | 727.4 MB |
| **Total** | | **17,062** | **1.83 GB** |

T1 Tomato is already trained locally — do NOT upload it to AWS.

---

## Split sizes

| Exp | Train | Val | Test | Classes |
|---|---:|---:|---:|---:|
| G1 | 1,908 | 409 | 409 | 4 |
| G2 | 2,433 | 522 | 522 | 4 |
| C1 | 1,352 | 290 | 290 | 5 |
| S1 | 1,764 | 378 | 379 | 5 |
| S2 | 4,483 | 961 | 961 | 10 |

---

## Training time estimates on g4dn.xlarge (T4 GPU)

T1 Tomato (10,170 train images, 10 classes) took 207.5 min on CPU at ~1 s/step.
On a T4 GPU with batch size 64, typical speedup for MobileNetV2 is 15–25x.

| Exp | Train images | Steps/epoch (bs=64) | Est. sec/epoch | Est. epochs | Est. time |
|---|---:|---:|---:|---:|---:|
| G1 | 1,908 | 30 | ~6 s | 15–25 | 3–4 min |
| G2 | 2,433 | 38 | ~8 s | 15–25 | 3–5 min |
| C1 | 1,352 | 22 | ~5 s | 15–20 | 2–3 min |
| S1 | 1,764 | 28 | ~6 s | 15–20 | 3–4 min |
| S2 | 4,483 | 70 | ~14 s | 20–30 | 7–10 min |
| **Total** | | | | | **~18–26 min** |

Add ~5–10 min for TF startup, conda activation, dataset loading, evaluation.
**Realistic total: 25–35 minutes. Comfortably within 1 hour.**

---

## Required storage on the AWS instance

| Component | Size |
|---|---|
| 5 datasets (images) | 1.83 GB |
| Project repo (code + splits) | ~15 MB |
| Saved models (5 x ~9.4 MB) | ~50 MB |
| Results (CSVs + PNGs) | ~10 MB |
| **Total** | **~2 GB** |

The default EBS root volume on the Deep Learning AMI is **~100 GB** —
far more than enough. No extra volume needed.

---

## File layout on the instance

```
/home/ubuntu/maharashtra-crop-disease-detection/   <- cloned from GitHub
    train_experiment.py
    run_all_experiments.sh
    aws_setup.sh
    fix_split_paths.py
    splits/
        grape_niphad/    train.csv  val.csv  test.csv  class_index.csv
        grape_2024/      ...
        chilli_cold/     ...
        sugarcane_maharashtra/  ...
        sugarcane_large/ ...
    results/             <- created during training
    models/              <- created during training

/data/datasets/                                    <- uploaded via scp
    grape_niphad/
    grape_2024/
    chilli_cold/
    sugarcane_maharashtra/
    sugarcane_large/
```

---

## Step-by-step launch instructions

### Step 1 — Launch the instance (AWS Console)

1. Go to EC2 → Launch Instance
2. Name: `crop-disease-training`
3. AMI: search "Deep Learning AMI GPU TensorFlow" → select Ubuntu 22.04 variant
4. Instance type: `g4dn.xlarge`
5. Key pair: select or create one, download the `.pem` file
6. Security group: allow SSH (port 22) from your IP only
7. Storage: leave default (~100 GB) — no changes needed
8. Launch

---

### Step 2 — SSH into the instance

```bash
# Linux / macOS / Git Bash on Windows:
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<public-ip>

# PowerShell on Windows:
ssh -i your-key.pem ubuntu@<public-ip>
```

---

### Step 3 — Create the dataset directory on the instance

Run this on the instance BEFORE uploading datasets:

```bash
sudo mkdir -p /data/datasets
sudo chown ubuntu:ubuntu /data/datasets
```

This must be done first so the scp commands in the next step have a destination.

---

### Step 4 — Upload datasets from your local machine

Run these from your **local machine** (not the instance).

**PowerShell (Windows):**
```powershell
$KEY  = "C:\path\to\your-key.pem"
$IP   = "<public-ip>"
$SRC  = "D:\CropDiseaseProject"
$DEST = "ubuntu@${IP}:/data/datasets/"

scp -i $KEY -r "$SRC\grape_niphad"          $DEST
scp -i $KEY -r "$SRC\grape_2024"            $DEST
scp -i $KEY -r "$SRC\chilli_cold"           $DEST
scp -i $KEY -r "$SRC\sugarcane_maharashtra" $DEST
scp -i $KEY -r "$SRC\sugarcane_large"       $DEST
```

**Git Bash or Linux/macOS:**
```bash
KEY="your-key.pem"
IP="<public-ip>"
SRC="/d/CropDiseaseProject"       # Git Bash path to D:\CropDiseaseProject
# On Linux/macOS use the actual path, e.g.: SRC="/mnt/d/CropDiseaseProject"

scp -i $KEY -r $SRC/grape_niphad           ubuntu@$IP:/data/datasets/
scp -i $KEY -r $SRC/grape_2024             ubuntu@$IP:/data/datasets/
scp -i $KEY -r $SRC/chilli_cold            ubuntu@$IP:/data/datasets/
scp -i $KEY -r $SRC/sugarcane_maharashtra  ubuntu@$IP:/data/datasets/
scp -i $KEY -r $SRC/sugarcane_large        ubuntu@$IP:/data/datasets/
```

Total upload size: **~1.83 GB**. Expected time: 5–20 min depending on connection.

---

### Step 5 — Run the setup script (on the instance)

```bash
# Clone the project
git clone https://github.com/sohail-148/maharashtra-crop-disease-detection.git
cd maharashtra-crop-disease-detection

# Run setup: activates conda, installs packages, fixes paths, verifies GPU
chmod +x aws_setup.sh
./aws_setup.sh
```

The setup script will:
- Activate the `tensorflow2_p310` conda environment
- Install missing Python packages into that environment
- Fix the Windows absolute paths in the split CSVs to Linux paths
- Verify all dataset folders are present
- Confirm TensorFlow can see the GPU

---

### Step 6 — Train all 5 experiments

```bash
cd ~/maharashtra-crop-disease-detection
./run_all_experiments.sh
```

Expected output: G1 → G2 → C1 → S1 → S2 run sequentially.
Total time: ~25–35 minutes. A summary is written to `results/aws_run_summary.txt`.

To run a single experiment:
```bash
python train_experiment.py \
    --experiment G1 \
    --dataset-root /data/datasets \
    --batch-size 64
```

---

### Step 7 — Download results to your local machine

Run from your **local machine**:

**PowerShell:**
```powershell
$KEY = "C:\path\to\your-key.pem"
$IP  = "<public-ip>"
$REMOTE = "ubuntu@${IP}:/home/ubuntu/maharashtra-crop-disease-detection"

scp -i $KEY -r "${REMOTE}/results" ./results_aws
scp -i $KEY -r "${REMOTE}/models"  ./models_aws
```

**Git Bash / Linux / macOS:**
```bash
KEY="your-key.pem"
IP="<public-ip>"
REMOTE="ubuntu@$IP:/home/ubuntu/maharashtra-crop-disease-detection"

scp -i $KEY -r $REMOTE/results ./results_aws
scp -i $KEY -r $REMOTE/models  ./models_aws
```

---

### Step 8 — Terminate the instance

**Do this immediately after downloading results.**  
Stopping (not terminating) still incurs EBS storage charges.

AWS Console → EC2 → Instances → select instance → Instance State → **Terminate**

---

## Cost summary

| Scenario | Duration | On-demand (~Rs. 49/hr) | Spot (~Rs. 14/hr) |
|---|---|---|---|
| Training only (~30 min) | 0.5 hr | ~Rs. 25 | ~Rs. 7 |
| With upload + setup (worst case) | 1 hr | ~Rs. 49 | ~Rs. 14 |

---

## Methodology consistency with T1 Tomato

`train_experiment.py` uses exactly the same methodology as `train_tomato.py`:

| Parameter | Value |
|---|---|
| Base model | MobileNetV2, ImageNet weights, frozen |
| Head | GAP → BatchNorm → Dropout(0.3) → Dense(N) |
| Optimiser | Adam, LR=1e-3 |
| Early stopping | patience=5, monitor val_loss |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=3 |
| Max epochs | 30 |
| Augmentation | flip LR/UD, rot90, random crop/zoom (train only) |
| Splits | 70/15/15 stratified from Phase 6 |
| Seed | 42 |
| Outputs | training_history.csv/.png, test_results.txt, test_metrics.csv, confusion_matrix.csv/.png, model_summary.txt |

The only intentional differences:
- `batch_size` defaults to **64** on GPU (was 32 on CPU) — passed explicitly, configurable
- `num_classes` is read dynamically from `class_index.csv` (not hardcoded)
- Chart titles and filenames use the experiment ID
- Path rewriting handles Windows → Linux translation at runtime
