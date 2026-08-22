# AWS GPU Training Plan — Experiments G1–S2

## Instance recommendation

| Field | Value |
|---|---|
| Instance type | `g4dn.xlarge` |
| GPU | NVIDIA T4 (16 GB VRAM) |
| vCPU | 4 |
| RAM | 16 GB |
| Region | ap-south-1 (Mumbai) |
| AMI | AWS Deep Learning AMI GPU TensorFlow 2.x (Ubuntu) |
| On-demand price | ~$0.58/hr ≈ ₹49/hr |
| Spot price | ~$0.13–0.16/hr ≈ ₹11–14/hr |

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

Tomato (T1, already trained locally) is NOT uploaded to AWS.

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

The T1 Tomato baseline (10,170 train images, 10 classes) took **207.5 min on CPU**,
approximately **~39 sec/epoch** at 318 steps × 1s/step.

On a T4 GPU with batch size 64, typical speedup is **15–25×** for MobileNetV2.
Expected per-epoch time: **2–4 seconds**.

Using batch_size=64 (double the CPU baseline of 32), steps per epoch halve,
so the per-epoch wall-clock time is roughly:

| Exp | Train images | Steps/epoch (bs=64) | Est. sec/epoch | Est. epochs | Est. time |
|---|---:|---:|---:|---:|---:|
| G1 | 1,908 | 30 | ~6s | 15–25 | **3–4 min** |
| G2 | 2,433 | 38 | ~8s | 15–25 | **3–5 min** |
| C1 | 1,352 | 22 | ~5s | 15–20 | **2–3 min** |
| S1 | 1,764 | 28 | ~6s | 15–20 | **3–4 min** |
| S2 | 4,483 | 70 | ~14s | 20–30 | **7–10 min** |
| **Total** | | | | | **~18–26 min** |

Add ~5 min for TF startup, dataset loading, and evaluation.
**Realistic total: 25–35 minutes wall-clock, well within 1 hour.**

---

## Required storage on AWS instance

| Component | Size |
|---|---|
| 5 datasets (images) | 1.83 GB |
| Project repo (code + splits) | ~15 MB |
| Saved models (5 × ~9.4 MB) | ~50 MB |
| Results (CSVs + PNGs) | ~10 MB |
| TF/Python overhead | included in AMI |
| **Total EBS needed** | **~2 GB usable** |

The default `g4dn.xlarge` root volume is **50 GB EBS** — far more than enough.
No extra volume needed.

---

## File layout on AWS instance

```
/home/ubuntu/maharashtra-crop-disease-detection/   ← cloned from GitHub
    train_experiment.py
    run_all_experiments.sh
    aws_setup.sh
    fix_split_paths.py
    splits/
        grape_niphad/    train.csv  val.csv  test.csv  class_index.csv
        grape_2024/      ...
        chilli_cold/     ...
        sugarcane_maharashtra/ ...
        sugarcane_large/ ...
    results/             ← created during training
    models/              ← created during training

/data/datasets/                                    ← datasets uploaded/downloaded
    grape_niphad/
    grape_2024/
    chilli_cold/
    sugarcane_maharashtra/
    sugarcane_large/
```

---

## Step-by-step launch instructions

### 1. Launch instance (AWS Console)

- AMI: **AWS Deep Learning AMI GPU TensorFlow 2.x (Ubuntu 22.04)**
- Instance type: `g4dn.xlarge`
- Storage: default 50 GB (no changes needed)
- Security group: allow SSH (port 22) from your IP only
- Key pair: select or create one

### 2. SSH in

```bash
ssh -i your-key.pem ubuntu@<public-ip>
```

### 3. Upload datasets (from your Windows machine)

```bash
# Run this from your local Windows machine (PowerShell or Git Bash)
# Total upload: ~1.83 GB — takes 5–15 min depending on connection

scp -i your-key.pem -r D:\CropDiseaseProject\grape_niphad     ubuntu@<ip>:/data/datasets/
scp -i your-key.pem -r D:\CropDiseaseProject\grape_2024        ubuntu@<ip>:/data/datasets/
scp -i your-key.pem -r D:\CropDiseaseProject\chilli_cold       ubuntu@<ip>:/data/datasets/
scp -i your-key.pem -r D:\CropDiseaseProject\sugarcane_maharashtra ubuntu@<ip>:/data/datasets/
scp -i your-key.pem -r D:\CropDiseaseProject\sugarcane_large   ubuntu@<ip>:/data/datasets/
```

Alternatively, if datasets are on Kaggle:
- Add your Kaggle API key, then run `aws_setup.sh` which downloads them automatically.
- Update the Kaggle dataset slugs in `aws_setup.sh` with the correct URLs first.

### 4. Run setup script (on the instance)

```bash
# Clone repo
git clone https://github.com/sohail-148/maharashtra-crop-disease-detection.git
cd maharashtra-crop-disease-detection

# Run setup (installs packages, fixes paths, verifies GPU)
chmod +x aws_setup.sh
./aws_setup.sh
```

### 5. Start training

```bash
# All 5 experiments sequentially (~25–35 min total)
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# Or run a single experiment
python train_experiment.py --experiment G1 --dataset-root /data/datasets --batch-size 64
```

### 6. Download results (from your Windows machine)

```bash
# Download only results and models (small — a few MB)
scp -i your-key.pem -r ubuntu@<ip>:/home/ubuntu/maharashtra-crop-disease-detection/results ./results_aws
scp -i your-key.pem -r ubuntu@<ip>:/home/ubuntu/maharashtra-crop-disease-detection/models  ./models_aws
```

### 7. Terminate the instance

**Important: terminate (not stop) the instance once done to avoid further charges.**

In AWS Console → EC2 → Instances → select instance → Instance State → Terminate.

---

## Cost summary

| Scenario | Duration | Cost (on-demand) | Cost (spot) |
|---|---|---|---|
| All 5 experiments | 35 min | ~$0.34 (~₹29) | ~$0.08 (~₹7) |
| With upload/setup overhead | 1 hour max | ~$0.58 (~₹49) | ~$0.16 (~₹14) |

**Worst case: ₹49 on-demand for 1 full hour.**

---

## Notes on methodology consistency

`train_experiment.py` uses the exact same:
- MobileNetV2 frozen base + GAP + BN + Dropout(0.3) + Dense(N)
- Adam LR=1e-3, EarlyStopping(patience=5), ReduceLROnPlateau(patience=3)
- 70/15/15 stratified splits from Phase 6
- On-the-fly augmentation (train only): flip, rot90, random crop/zoom
- Identical output structure: training_history.csv/.png, test_results.txt,
  test_metrics.csv, confusion_matrix.csv/.png, model_summary.txt
- Seed=42 throughout

The only differences from train_tomato.py are:
- `num_classes` is read from class_index.csv dynamically (not hardcoded)
- `batch_size` defaults to 64 on GPU (vs 32 on CPU) — configurable
- Chart title and file names use the experiment ID/dataset name
- Path rewriting handles the Windows→Linux path translation
