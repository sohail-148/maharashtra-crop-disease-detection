# Crop Disease Detection Project — Master Handoff Package

**Document Version:** 1.2  
**Project Workspace:** `D:\CropDiseaseProject`  
**Git Repository:** `https://github.com/sohail-148/maharashtra-crop-disease-detection` (Branch: `main`)  
**Status as of Handoff:** Completed Phase 1–7 & T1 Tomato Baseline (90.23%); Approved 4-Crop Production Architecture locked (Tomato, Grape G1+G2, Chilli C1, Sugarcane S1+S2); Web App updated to 4 crops; 3 remaining training jobs prepared for Kaggle/Colab GPU.

---

## 1. Project Title & Objective

- **Full Project Title:**  
  *Image-Based Disease Detection of Maharashtra-Relevant Crops*
- **Subtitle:**  
  *An AI-Based Web Application for Crop Disease Detection Using MobileNetV2 Transfer Learning*
- **Primary Objective:**  
  Build and deploy a complete, accessible web-based application that allows agricultural users to upload or capture a leaf photograph, classifies disease conditions using a lightweight MobileNetV2 transfer-learning model across four Maharashtra-relevant crops (Tomato, Grape, Chilli, Sugarcane), provides visual interpretability via Grad-CAM heatmaps, displays prediction confidence, and logs inference history to a local SQLite database.

---

## 2. Research Questions & Methodology

### Central Research Question
> *How effectively can a lightweight MobileNetV2 transfer-learning framework classify diseases across multiple Maharashtra-relevant crop datasets?*

### Supporting Research Questions
1. How accurately does MobileNetV2 classify diseases for each crop?
2. How does performance vary between distinct datasets for the same crop (Niphad Grape vs Grape 2024, Maharashtra Sugarcane vs Large Sugarcane)?
3. Which disease classes are difficult to classify, and what visual or data factors cause the confusion?
4. How does class imbalance affect class-level precision, recall, and F1-scores?
5. Can Grad-CAM provide meaningful, agronomically plausible visual explanations of model focus?
6. Can the trained transfer-learning models be successfully integrated into an efficient, responsive web application?

### Key Contributions & Research Defense
We do **not** claim individual novelty in MobileNetV2, transfer learning, or Grad-CAM. The defensible contribution is the rigorous combination and comparative evaluation:
$$\text{Maharashtra-relevant crop focus} + \text{6 independent public/regional datasets} + \text{Uniform MobileNetV2 pipeline} + \text{Class-level error analysis} + \text{Grad-CAM interpretability} + \text{End-to-end Web Application}$$

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Core Language** | Python 3.11.9 |
| **Deep Learning** | TensorFlow 2.18.1, Keras 3.15.1 (MobileNetV2 pretrained on ImageNet) |
| **Image Processing** | OpenCV 4.10.0, Pillow 12.3.0 |
| **Evaluation Metrics** | scikit-learn 1.5.2, pandas 3.0.5, matplotlib 3.11.1 |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Backend Web Server** | Flask 3.1.1, Werkzeug 3.1.8 |
| **Frontend UI** | Semantic HTML5, Vanilla CSS3 (Mobile-First, Responsive), JavaScript (ES6+, Camera `getUserMedia` API) |
| **Database** | SQLite 3 (stored locally in `instance/predictions.db`) |

---

## 4. Datasets, Research Baselines & Approved 4-Crop Architecture

### Dual-Layer Strategy
1. **Research Baseline Layer (6 Independent Experiments):**
   To evaluate dataset-specific performance and domain properties, the original 6 experiments (T1, G1, G2, C1, S1, S2) and their stratified 70/15/15 splits are permanently preserved in `splits/`.
2. **Production Deployment Layer (4 Crop-Level Models):**
   For real-world usability and seamless farmer experience, the web application maps the 4 crops directly to 4 dedicated models:
   - **🍅 Tomato:** T1 Baseline Model (10 classes | 14,529 images — *Trained & Ready: 90.23% accuracy*)
   - **🍇 Grape:** Grape Unified G1+G2 Model (7 canonical classes | 6,203 images)
   - **🌶️ Chilli:** Chilli C1 Model (5 canonical classes | 1,932 images)
   - **🌾 Sugarcane:** Sugarcane Unified S1+S2 Model (11 canonical classes | 8,926 images)
   - *Total Deployment Scope:* **33 canonical classes across 31,590 images**. All unified splits are generated with zero cross-dataset image overlap or leakage.

### Verified Dataset Summary (Total: 31,590 Images)

| Exp ID | Crop | Dataset Name & Relevance | Classes | Verified Images | Disk Size | Local Path |
|---|---|---|---:|---:|---:|---|
| **T1** | Tomato | PlantVillage Tomato subset | 10 | 14,529 | 219.9 MB | `tomato_plantvillage/` |
| **G1** | Grape | Niphad Grape (Nashik / Maharashtra) | 4 | 2,726 | 24.8 MB | `grape_niphad/` |
| **G2** | Grape | Mendeley Grape 2024 | 4 | 3,477 | 860.0 MB | `grape_2024/` |
| **C1** | Chilli | COLD 2024 (resized raw subset) | 5 | 1,932 | 60.8 MB | `chilli_cold/` |
| **S1** | Sugarcane | Maharashtra Sugarcane Dataset | 5 | 2,521 | 159.9 MB | `sugarcane_maharashtra/` |
| **S2** | Sugarcane | Large Sugarcane Leaf Dataset | 10 | 6,405 | 727.4 MB | `sugarcane_large/` |
| **Total**| — | **6 Datasets Across 4 Crops** | **38** | **31,590** | **~2.05 GB** | — |

*Note on `grape_niphad`:* Contains a hidden Windows `desktop.ini` system file in `Downy Mildew/` (raw count 2,727). It is automatically filtered out by image extension parsing (`.jpg`, `.jpeg`, `.png`) and excluded from split CSVs. Actual valid image count is exactly **2,726**.

---

### Class Breakdown per Experiment

#### T1 — Tomato (10 Classes | 14,529 images)
- Bacterial Spot: 1,702
- Early Blight: 800
- Healthy: 1,273
- Late Blight: 1,527
- Leaf Mold: 761
- Septoria Leaf Spot: 1,417
- Spider Mites (Two-spotted spider mite): 1,341
- Target Spot: 1,123
- Tomato Mosaic Virus: 299
- Tomato Yellow Leaf Curl Virus: 4,286

#### G1 — Niphad Grape (4 Classes | 2,726 images)
- Bacterial Leaf Spot: 100
- Downy Mildew: 966
- Healthy Leaves: 1,254
- Powdery Mildew: 406

#### G2 — Grape 2024 (4 Classes | 3,477 images)
- Black Rot: 808
- Esca (Black Measles): 888
- Healthy: 1,109
- Leaf Blight (Isariopsis Clavispora): 672

#### C1 — Chilli COLD (5 Classes | 1,932 images)
- Cerocospora: 899
- Healthy: 329
- Murda Complex: 275
- Nutritional Deficiency: 267
- Powdery Mildew: 162

#### S1 — Maharashtra Sugarcane (5 Classes | 2,521 images)
- Healthy: 522
- Mosaic: 462
- RedRot: 518
- Rust: 514
- Yellow: 505

#### S2 — Large Sugarcane (10 Classes | 6,405 images)
- Banded Chlorosis: 471
- Brown Spot: 1,722
- Brown Rust: 314
- Grassy Shoot: 346
- Healthy Leaves: 430
- Pokkah Boeng: 297
- Sett Rot: 652
- Smut: 316
- Viral Disease: 663
- Yellow Leaf: 1,194

---

## 5. Dataset Splitting Details (70 / 15 / 15 Stratified)

Splits were generated using stratified random sampling (`seed=42`) via `dataset_split.py` / `dataset_preparation.py` without physical image duplication. Reference CSV files are located in `splits/<dataset_id>/`:
- `train.csv` (70%)
- `val.csv` (15%)
- `test.csv` (15%)
- `class_index.csv` (mapping string class label $\to$ integer class index)

| Experiment | Training Set (70%) | Validation Set (15%) | Test Set (15%) | Total Images | Classes |
|---|---:|---:|---:|---:|---:|
| **T1 — Tomato** | 10,170 | 2,179 | 2,180 | 14,529 | 10 |
| **G1 — Niphad Grape** | 1,908 | 409 | 409 | 2,726 | 4 |
| **G2 — Grape 2024** | 2,433 | 522 | 522 | 3,477 | 4 |
| **C1 — Chilli COLD** | 1,352 | 290 | 290 | 1,932 | 5 |
| **S1 — Sugarcane Mah.** | 1,764 | 378 | 379 | 2,521 | 5 |
| **S2 — Sugarcane Large** | 4,483 | 961 | 961 | 6,405 | 10 |
| **Grand Total** | **22,110** | **4,739** | **4,741** | **31,590** | **38** |

> **Repository Status Note regarding Splits:**  
> All six split sets exist locally. Verify exactly which split CSVs are currently tracked in Git before claiming they are backed up to GitHub (`git ls-files splits/` currently tracks only `splits/tomato/` because relative directory rules in `.gitignore` match directory names like `grape_niphad/` inside `splits/`).

---

## 6. Exact Model Architecture & Training Configuration

All experiments follow the exact baseline protocol validated in T1:

```
Input Image (224 × 224 × 3)
      │
MobileNetV2 Base (ImageNet pretrained weights, frozen: trainable=False)
      │  [Output shape: (None, 7, 7, 1280)]
GlobalAveragePooling2D
      │  [Output shape: (None, 1280)]
BatchNormalization
      │  [Output shape: (None, 1280)]
Dropout (rate = 0.3)
      │  [Output shape: (None, 1280)]
Dense (N classes, activation = 'softmax')
```

### Verified Parameter Counts (Directly from `models/tomato/model_summary.txt`)
- **Total Parameters:** **2,275,914** (8.68 MB)
- **Trainable Parameters:** **15,370** (60.04 KB) — Dense classification head weights ($1,280 \times 10 + 10 = 12,810$) + BatchNormalization trainable parameters ($\gamma, \beta = 2,560$)
- **Non-Trainable / Frozen Parameters:** **2,260,544** (8.62 MB) — MobileNetV2 base weights ($2,257,984$) + BatchNormalization moving statistics ($\mu, \sigma^2 = 2,560$)

### Hyperparameters & Training Pipeline
- **Input Dimensions:** $224 \times 224 \times 3$, normalized using MobileNetV2 `preprocess_input` (scales pixel values to $[-1, 1]$).
- **Optimization:** Adam optimizer ($\text{learning rate} = 1\times 10^{-3}$, $\epsilon = 1\times 10^{-7}$).
- **Loss Function:** `sparse_categorical_crossentropy`.
- **Batch Size:**
  - `32` for local CPU execution (T1).
  - `64` for GPU training (G1–S2).
- **Maximum Epochs:** 30.
- **Random Seed:** `42` across Python, NumPy, and TensorFlow.
- **On-The-Fly Augmentation (Training set only):**
  - Random horizontal flip (`random_flip_left_right`)
  - Random vertical flip (`random_flip_up_down`)
  - Random $90^\circ$ rotation (`rot90` with $k \in \{0, 1, 2, 3\}$)
  - Random crop and resize (scale factor $0.80 - 1.00$)
- **Callbacks:**
  1. `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  2. `ModelCheckpoint(filepath=..., monitor='val_loss', save_best_only=True)`
  3. `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`
  4. `CSVLogger(filename=training_history.csv)`

---

## 7. T1 Tomato Baseline Results (Completed)

- **Execution Environment:** Local Windows CPU
- **Training Time:** 207.5 minutes (30 epochs completed; best validation loss achieved at epoch 30)
- **Overall Test Set Metrics:**
  - **Accuracy:** **90.23%** (1,967 / 2,180 correct)
  - **Weighted Precision:** **90.40%**
  - **Weighted Recall:** **90.23%**
  - **Weighted F1-Score:** **90.18%**
  - **Best Val Loss:** 0.2749 (Val Accuracy: 91.10%)

### Detailed Per-Class Test Performance (T1)

| Class Label | Precision | Recall | F1-Score | Support (Test Images) |
|---|---:|---:|---:|---:|
| Bacterial Spot | 0.9105 | 0.9141 | 0.9123 | 256 |
| Early Blight | 0.8242 | 0.6250 | **0.7109** | 120 |
| Late Blight | 0.8932 | 0.9127 | 0.9028 | 229 |
| Leaf Mold | 0.8962 | 0.8333 | 0.8636 | 114 |
| Septoria Leaf Spot | 0.8591 | 0.8873 | 0.8730 | 213 |
| Spider Mites | 0.8473 | 0.8557 | 0.8515 | 201 |
| Target Spot | 0.7358 | 0.8452 | **0.7867** | 168 |
| Tomato Yellow Leaf Curl Virus | 0.9904 | 0.9658 | **0.9780** | 643 |
| Tomato Mosaic Virus | 0.9767 | 0.9333 | 0.9545 | 45 |
| Healthy | 0.9126 | 0.9843 | 0.9471 | 191 |
| **Macro Average** | **0.8846** | **0.8757** | **0.8780** | **2,180** |
| **Weighted Average** | **0.9040** | **0.9023** | **0.9018** | **2,180** |

### Empirical Class-Level Observations
- **Lowest F1-Scores:** Early Blight and Target Spot had the lowest F1-scores and should be examined further through confusion-matrix and image-level error analysis. (Inspection of `results/tomato/confusion_matrix.csv` shows Early Blight suffered misclassifications primarily into Septoria Leaf Spot [13 samples], Late Blight [11 samples], and Target Spot [8 samples]).
- **Highest F1-Score:** Yellow Leaf Curl Virus achieved the highest F1-score (0.9780) and had 643 test samples. The high performance may relate to distinctive visual characteristics and the relatively large sample size, but this requires further error analysis.

---

## 8. Current Web Application Status

A complete Flask web application foundation is implemented and runnable (`python run.py`).

### Implemented Architecture
- `run.py`: Application entry point (runs on `http://127.0.0.1:5000`).
- `config.py`: `DevelopmentConfig` and `ProductionConfig` classes.
- `app/__init__.py`: Application factory (`create_app()`), initializes database and sets up upload limits (16 MB).
- `app/routes.py`: Endpoints for UI pages and JSON APIs:
  - `GET /` : Main index page (Crop selection grid, Upload tab, Camera tab).
  - `POST /predict` : Handles multi-part file uploads and base64 webcam captures, calls predictor service, records to SQLite, redirects to results.
  - `GET /result/<prediction_id>` : Renders prediction details, confidence bar, per-class breakdown, and Grad-CAM container.
  - `GET /history` : Paginated prediction log with thumbnails and delete actions.
  - `POST /history/delete/<id>` : Deletes record and associated image file from disk.
  - `GET /about` : Research context and live status of the 6 models.
  - `GET /api/status` : JSON status of model availability.
- `app/predictor.py`: Prediction service isolating ML logic from web routes:
  - Automatically loads `.keras` model from `models/<experiment>/` if present on disk.
  - Performs live inference for T1 (`models/tomato/tomato_baseline.keras` present).
  - Cleanly returns placeholder responses (`is_placeholder = True`) for un-trained models (G1–S2) without throwing unhandled exceptions.
- `app/database.py`: Thread-safe SQLite database manager for predictions.
- `app/templates/`: Jinja2 templates (`base.html`, `index.html`, `result.html`, `history.html`, `about.html`, `error.html`).
- `app/static/`: Mobile-first responsive CSS (`main.css`) and JavaScript modules (`upload.js`, `camera.js`, `main.js`).

---

## 9. Current AWS GPU Training Pipeline Status & Known AMI Issue

> **IMPORTANT NOTICE:** AWS account/signup is currently pending. **All AWS activities are stopped.** Do not start EC2 instances, launch training, or incur billing until account status is confirmed.

### Scripted Pipeline Files (Committed to Git)
- `train_experiment.py`: Parameterized training script (`--experiment`, `--dataset-root`, `--batch-size`) mirroring `train_tomato.py`.
- `run_all_experiments.sh`: Sequential execution script for G1 $\to$ G2 $\to$ C1 $\to$ S1 $\to$ S2.
- `aws_setup.sh`: Instance bootstrap script for installing dependencies and preparing directories.
- `fix_split_paths.py`: Rewrites Windows absolute filepaths in split CSVs to Linux paths (`/data/datasets/...`).
- `aws_training_plan.md`: Step-by-step documentation for launching, uploading via SCP, running, downloading results, and terminating instance.

### Critical Known Issue: AMI & Environment Discrepancy
- **Original Plan Assumption:** `aws_setup.sh` and `aws_training_plan.md` were written assuming the **Ubuntu 22.04 Deep Learning AMI** with a pre-configured conda environment named `tensorflow2_p310` and `apt-get` system package manager.
- **Current Target AMI:** The selected AMI is **Deep Learning Base AMI with Single CUDA (Amazon Linux 2023), x86_64**.
- **Impact & Action Required Prior to AWS Launch:**
  1. Amazon Linux 2023 uses `dnf` (not `apt-get`). `sudo apt-get install` commands in `aws_setup.sh` (lines 75–76) will fail on AL2023.
  2. The `tensorflow2_p310` conda environment may not exist on AL2023 Base AMI.
  3. Prior to executing the AWS training run, verify the Python/Conda environment on the launched AL2023 instance using `conda env list` or create a dedicated virtualenv/conda environment, and update `aws_setup.sh` accordingly.

---

## 10. Completed vs. In-Progress vs. Pending Breakdown

```
[Phase 1: Research Foundation]        ✅ COMPLETED
[Phase 2: Dataset Acquisition]        ✅ COMPLETED (6 datasets, 31,590 images)
[Phase 3: Dataset Organization]       ✅ COMPLETED (Cleaned, verified on D:)
[Phase 4: Dataset Analysis]           ✅ COMPLETED (dataset_analysis.py, CSVs)
[Phase 5: Experimental Design]        ✅ COMPLETED (6 experiments, 70/15/15)
[Phase 6: Dataset Splitting]          ✅ COMPLETED (splits/*.csv generated)
[Phase 7: ML Environment Setup]      ✅ COMPLETED (Python 3.11.9, TF 2.18.1)
[Phase 8 - T1: Tomato Baseline]       ✅ COMPLETED (Model & results saved)
─────────────────────────────────────────────────────────────────────────────
[Phase 8 - G1 to S2 Experiments]      ⏳ IN PROGRESS (Pipeline written; AWS paused)
[Phase 11-14: Web App Foundation]     ✅ COMPLETED (Flask, SQLite, UI, Predictor)
─────────────────────────────────────────────────────────────────────────────
[Phase 9: Comparative Analysis]       🔜 PENDING (Awaiting G1–S2 results)
[Phase 10: Error Analysis]            🔜 PENDING (Confusion matrix inspection)
[Phase 15: Grad-CAM Implementation]   🔜 PENDING (app/gradcam.py module)
[Phase 16: End-to-End Integration]    🔜 PENDING (Full multi-model test)
[Phase 17: Production & Deployment]   🔜 PENDING
[Phase 20: Research Thesis Chapters]  🔜 PENDING (Chapters 5, 6, 7, 8, 9)
```

---

## 11. Exact Remaining Tasks in Sequence

When the project resumes, follow this exact step-by-step order:

1. **Verify AWS AMI & Environment:**
   - When AWS account is active, launch `g4dn.xlarge` with the selected AMI.
   - Inspect Python/CUDA setup on the instance.
   - Update `aws_setup.sh` (e.g. adjust package manager commands for AL2023 if needed).
2. **Execute G1–S2 Training on AWS:**
   - Create `/data/datasets/` on the instance.
   - Upload 5 datasets (`grape_niphad`, `grape_2024`, `chilli_cold`, `sugarcane_maharashtra`, `sugarcane_large` — ~1.83 GB total) via SCP.
   - Run `aws_setup.sh` and execute `./run_all_experiments.sh`.
   - Download `results/` and `models/` back to local machine (`D:\CropDiseaseProject`).
   - Terminate the EC2 instance immediately.
3. **Phase 9 — Comparative Research Analysis:**
   - Consolidate test metrics (Accuracy, Precision, Recall, F1) for all 6 experiments into a unified comparative table.
   - Compare intra-crop datasets: G1 (Niphad) vs G2 (Grape 2024), S1 (Maharashtra) vs S2 (Large Sugarcane).
4. **Phase 10 — Error Analysis:**
   - Inspect confusion matrices for each experiment.
   - Analyze top misclassified pairs (e.g. Tomato Early Blight vs Septoria / Late Blight / Target Spot).
5. **Phase 15 — Grad-CAM Explainability:**
   - Implement `app/gradcam.py` using TensorFlow gradient tape targeted at the last convolutional layer of MobileNetV2 (`out_relu`).
   - Wire heatmap generation into `app/predictor.py` and display on `app/templates/result.html`.
6. **Phase 16 — Web Application Multi-Model Integration & Verification:**
   - Drop all downloaded `.keras` model files into their respective subfolders under `models/`.
   - Verify that all 6 experiments yield live predictions and heatmaps through the UI and camera inputs.
7. **Phase 17 — Production Packaging:**
   - Create production configuration with environment variable support (`SECRET_KEY`).
   - Test deployment packaging.
8. **Phase 20 — Research Documentation / Thesis:**
   - Complete Chapter 5 (Model Architecture & Transfer Learning), Chapter 6 (Experimental Results & Comparative Tables), Chapter 7 (Web Application Design & Implementation), Chapter 8 (Discussion & Research Gap Resolution), Chapter 9 (Conclusion & Future Work).

---

## 12. Non-Negotiable Constraints & Architecture Rules

1. **Do Not Re-Download Datasets or Modify Images:**
   The 6 local datasets (31,590 images total) on `D:\CropDiseaseProject` are verified. Never re-download, resize on disk, or duplicate into train/val/test folders.
2. **Do Not Merge Separate Datasets for the Same Crop:**
   G1 and G2 have distinct disease taxonomies (Downy/Powdery Mildew vs Black Rot/Esca). S1 and S2 have distinct disease classes. They must remain separate experiments.
3. **Do Not Change the Base ML Architecture:**
   All 6 experiments must use the uniform MobileNetV2 (ImageNet pretrained, frozen base) configuration to maintain experimental consistency across the research.
4. **No On-Disk Augmentation:**
   Data augmentation must strictly occur on-the-fly during training via `tf.data` pipeline.
5. **Stratified Splitting Integrity:**
   The established 70/15/15 stratified splits generated with seed `42` are the ground truth for all evaluations.
6. **AWS Work on Hold:**
   No cloud resources or instances should be provisioned until the AWS account is formally ready.

---

## 13. File & Repository Structure Reference

```
D:\CropDiseaseProject/
├── .gitignore
├── PROJECT_HANDOFF.md                 <-- Single source of truth handoff document
├── config.py                          <-- Flask configuration (Dev / Prod)
├── run.py                             <-- Web application entry point
├── dataset_analysis.py                <-- Dataset verification script
├── dataset_preparation.py             <-- Dataset validation & splitting script
├── dataset_split.py                   <-- Stratified split generator
├── dataset_summary.csv                <-- Class-by-class image counts
├── dataset_file_list.csv              <-- Master list of all 31,590 files
├── verify_environment.py              <-- ML environment verification script
├── train_tomato.py                    <-- T1 Tomato baseline training script
├── train_experiment.py                <-- Parameterized training script (G1-S2)
├── run_all_experiments.sh             <-- Sequential experiment runner for Linux/AWS
├── aws_setup.sh                       <-- AWS instance bootstrap script
├── aws_training_plan.md               <-- AWS execution instructions & guide
├── fix_split_paths.py                 <-- Path rewriter for Linux training
│
├── app/                               <-- Flask Web Application
│   ├── __init__.py                    <-- App factory
│   ├── database.py                    <-- SQLite database operations
│   ├── predictor.py                   <-- Inference service layer
│   ├── routes.py                      <-- Web routes and endpoints
│   ├── static/
│   │   ├── css/main.css               <-- Mobile-first stylesheet
│   │   ├── js/camera.js               <-- WebRTC Camera capture handler
│   │   ├── js/main.js                 <-- Main interactive UI logic
│   │   ├── js/upload.js               <-- Drag-and-drop & validation logic
│   │   └── img/favicon.svg
│   └── templates/
│       ├── base.html                  <-- Common layout template
│       ├── index.html                 <-- Home page (Crop select, Upload, Camera)
│       ├── result.html                <-- Analysis result & Grad-CAM container
│       ├── history.html               <-- Prediction log with thumbnails
│       ├── about.html                 <-- Project info & model statuses
│       └── error.html                 <-- Error template (404/413/500)
│
├── instance/                          <-- SQLite database storage (gitignored)
│   └── predictions.db
│
├── models/                            <-- Trained .keras models (gitignored)
│   └── tomato/
│       ├── model_summary.txt
│       └── tomato_baseline.keras      <-- Trained T1 model (9.4 MB)
│
├── results/                           <-- Experiment results & metrics
│   └── tomato/
│       ├── confusion_matrix.csv
│       ├── confusion_matrix.png
│       ├── test_metrics.csv
│       ├── test_results.txt
│       ├── training_history.csv
│       └── training_history.png
│
└── splits/                            <-- Stratified train/val/test CSVs
    ├── tomato/                        <-- (Tracked in Git)
    ├── grape_niphad/                  <-- (Local on disk)
    ├── grape_2024/                    <-- (Local on disk)
    ├── chilli_cold/                   <-- (Local on disk)
    ├── sugarcane_maharashtra/         <-- (Local on disk)
    └── sugarcane_large/               <-- (Local on disk)
```
