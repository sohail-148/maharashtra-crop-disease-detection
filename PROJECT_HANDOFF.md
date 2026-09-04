# Crop Disease Detection Project — Master Handoff Package

**Document Version:** 1.4  
**Project Workspace:** `D:\CropDiseaseProject`  
**Git Repository:** `https://github.com/sohail-148/maharashtra-crop-disease-detection` (Branch: `main`)  
**Status as of Handoff:** Completed Phases 1–8 (Model Training & Verification), Phase 9 (Comparative / Cross-Dataset Evaluation), Phase 10 (In-Depth Error Analysis), and Phase 15 (Standalone Grad-CAM Explainability Analysis on 10 Prioritized Cases); 4-Crop Production Architecture Fully Active in Flask Web App; Integration Smoke Tests Verified; Ready for Next Active Phase: Application Grad-CAM Integration into Flask.

---

## 1. Project Title & Objective

- **Full Project Title:**  
  *Image-Based Disease Detection of Maharashtra-Relevant Crops*
- **Subtitle:**  
  *An AI-Based Web Application for Crop Disease Detection Using MobileNetV2 Transfer Learning*
- **Primary Objective:**  
  Build and deploy a complete, accessible web-based application that allows agricultural users to upload or capture a leaf photograph, classifies disease conditions using lightweight MobileNetV2 transfer-learning models across four Maharashtra-relevant crops (Tomato, Grape, Chilli, Sugarcane), provides visual interpretability via Grad-CAM heatmaps, displays prediction confidence, and logs inference history to a local SQLite database.

---

## 2. Research Questions & Methodology

### Central Research Question
> *How effectively can a lightweight MobileNetV2 transfer-learning framework classify diseases across multiple Maharashtra-relevant crop datasets?*

### Supporting Research Questions
1. How accurately does MobileNetV2 classify diseases for each crop?
2. How does performance vary between distinct datasets for the same crop when evaluated on unified models (Niphad Grape vs Grape 2024, Maharashtra Sugarcane vs Large Sugarcane)?
3. Which disease classes are difficult to classify, and what visual or data factors cause the confusion?
4. How does class imbalance affect class-level precision, recall, and F1-scores?
5. Can Grad-CAM provide meaningful, agronomically plausible visual explanations of model focus?
6. Can the trained transfer-learning models be successfully integrated into an efficient, responsive web application?

### Key Contributions & Research Defense
We do **not** claim individual novelty in MobileNetV2, transfer learning, or Grad-CAM. The defensible contribution is the rigorous combination and comparative evaluation:
$$\text{Maharashtra-relevant crop focus} + \text{6 independent public/regional datasets} + \text{Uniform MobileNetV2 pipeline} + \text{Comparative cross-dataset evaluation} + \text{In-depth error analysis} + \text{Grad-CAM explainability} + \text{End-to-end Web Application}$$

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Core Language** | Python 3.11.9 |
| **Deep Learning** | TensorFlow 2.18.1, Keras 3.15.1 (MobileNetV2 pretrained on ImageNet) |
| **Image Processing** | OpenCV 4.10.0, Pillow 12.3.0 |
| **Evaluation Metrics** | scikit-learn 1.5.2, pandas 3.0.5, matplotlib 3.11.1 |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping via `tf.GradientTape`) |
| **Backend Web Server** | Flask 3.1.1, Werkzeug 3.1.8 |
| **Frontend UI** | Semantic HTML5, Vanilla CSS3 (Mobile-First, Responsive), JavaScript (ES6+, Camera `getUserMedia` API) |
| **Database** | SQLite 3 (stored locally in `instance/predictions.db`) |

---

## 4. Datasets, Research Baselines & 4-Crop Production Architecture

### Dual-Layer Strategy
1. **Production Deployment Layer (Exactly 4 Dedicated Crop Models):**  
   For real-world agricultural usability and seamless farmer experience, the web application maps the 4 crops directly to 4 dedicated production models:
   - **🍅 Tomato:** T1 Baseline Model (10 classes | 14,529 images — **Trained & Verified: 90.23% accuracy, 90.18% F1**)
   - **🍇 Grape:** Grape Unified G1+G2 Model (7 canonical classes | 6,203 images — **Trained & Verified: 89.04% accuracy, 88.68% F1**)
   - **🌶️ Chilli:** Chilli C1 Model (5 canonical classes | 1,932 images — **Trained & Verified: 63.79% accuracy, 62.36% F1**)
   - **🌾 Sugarcane:** Sugarcane Unified S1+S2 Model (11 canonical classes | 8,926 images — **Trained & Verified: 83.43% accuracy, 83.29% F1**)
   - *Total Deployment Scope:* **33 canonical classes across 31,590 images**. All unified splits are generated with zero cross-dataset image overlap or data leakage.
   - *Architecture Rule:* G1 and G2 are **not** separate production models; S1 and S2 are **not** separate production models. There are exactly 4 production models.

2. **Research Evaluation Layer (Dataset-Specific Comparative Analysis):**  
   To evaluate dataset-specific performance, regional variance, and generalizability:
   - Grape dataset G1 (Niphad, Nashik) and G2 (Grape 2024) test partitions are evaluated against the unified Grape model to quantify source-partition performance differences.
   - Sugarcane dataset S1 (Maharashtra) and S2 (Large Sugarcane) test partitions are evaluated against the unified Sugarcane model to analyze cross-dataset disease discrimination.
   - The original research baseline splits (T1, G1, G2, C1, S1, S2) are preserved in `splits/` for comparative reporting.

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

### Class Breakdown per Production Crop

#### 🍅 Tomato (10 Classes | 14,529 images) — Model: `models/tomato/tomato_baseline.keras`
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

#### 🍇 Grape Unified (7 Classes | 6,203 images) — Model: `models/grape_unified/grape_unified_baseline.keras`
- Bacterial Leaf Spot (G1): 100
- Black Rot (G2): 808
- Downy Mildew (G1): 965
- Esca / Black Measles (G2): 894
- Healthy Leaves (G1 + G2): 2,362 (G1: 1,254, G2: 1,108)
- Leaf Blight / Isariopsis (G2): 667
- Powdery Mildew (G1): 407

#### 🌶️ Chilli (5 Classes | 1,932 images) — Model: `models/chilli_cold/chilli_cold_baseline.keras`
- Cercospora Leaf Spot: 899
- Healthy: 332
- Murda Complex (Leaf Curl): 275
- Nutritional Deficiency: 266
- Powdery Mildew: 160

#### 🌾 Sugarcane Unified (11 Classes | 8,926 images) — Model: `models/sugarcane_unified/sugarcane_unified_baseline.keras`
- Banded Chlorosis (S2): 466
- Brown Spot (S2): 1,720
- Grassy Shoot (S2): 348
- Healthy Leaves (S1 + S2): 960 (S1: 526, S2: 434)
- Mosaic / Viral Disease (S1 + S2): 1,126 (S1: 460, S2: 666)
- Pokkah Boeng (S2): 300
- Red Rot (S1): 518
- Rust / Brown Rust (S1 + S2): 827 (S1: 513, S2: 314)
- Sett Rot (S2): 655
- Smut (S2): 315
- Yellow Leaf Disease (S1 + S2): 1,691 (S1: 504, S2: 1,187)

---

## 5. Dataset Splitting & Integrity Verification

All datasets were split using stratified sampling (70% Train, 15% Validation, 15% Test) with fixed random seed `42`:

| Dataset Scope | Train Images | Val Images | Test Images | Total Images | Status |
|---|---:|---:|---:|---:|:---:|
| **Tomato (T1)** | 10,167 | 2,182 | 2,180 | 14,529 | ✅ Verified |
| **Grape Unified (G1+G2)** | 4,340 | 932 | 931 | 6,203 | ✅ Verified |
| **Chilli (C1)** | 1,351 | 291 | 290 | 1,932 | ✅ Verified |
| **Sugarcane Unified (S1+S2)** | 6,246 | 1,340 | 1,340 | 8,926 | ✅ Verified |
| **Grape Niphad (G1)** | 1,908 | 409 | 409 | 2,726 | ✅ Verified |
| **Grape 2024 (G2)** | 2,432 | 523 | 522 | 3,477 | ✅ Verified |
| **Sugarcane Maharashtra (S1)** | 1,763 | 379 | 379 | 2,521 | ✅ Verified |
| **Sugarcane Large (S2)** | 4,483 | 961 | 961 | 6,405 | ✅ Verified |

---

## 6. Training Pipeline & Model Architecture

### Baseline Architecture & Hyperparameters
- **Input Size:** $224 \times 224 \times 3$
- **Base Model:** Pretrained MobileNetV2 (ImageNet weights, top excluded, base weights completely **frozen** during baseline training)
- **Top Layers:** GlobalAveragePooling2D $\to$ BatchNormalization $\to$ Dropout(0.3) $\to$ Dense(num_classes, activation="softmax")
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam (learning_rate = 0.001)
- **Batch Size:** 32
- **Epochs:** Up to 30 epochs with EarlyStopping (`patience=5`, `restore_best_weights=True`, monitoring `val_loss`)
- **Callbacks:** ModelCheckpoint (saves best `.keras`), CSVLogger (logs all epoch metrics)
- **Data Augmentation:** Random horizontal/vertical flip, random 90-degree rotation, random crop (0.80–1.0 scale) applied dynamically on-the-fly via `tf.data`.

---

## 7. Model Training & Test Benchmark Results (Phase 8)

All four production models have completed training and test evaluation.

### Master Production Benchmark Table

| Crop | Production Model Name | Scope | Classes | Test Samples | Test Accuracy | Weighted F1 | Training Hardware | Training Time | Model Size | Status |
|---|---|---|---:|---:|---:|---:|---|---:|---:|:---:|
| **🍅 Tomato** | `tomato_baseline.keras` | T1 PlantVillage | 10 | 2,180 | **90.23%** | **90.18%** | Local CPU | ~6.5 hr | 9.37 MB | ✅ COMPLETE |
| **🍇 Grape** | `grape_unified_baseline.keras` | G1 + G2 Unified | 7 | 931 | **89.04%** | **88.68%** | Kaggle GPU (T4) | 6.4 min | 9.32 MB | ✅ COMPLETE |
| **🌶️ Chilli** | `chilli_cold_baseline.keras` | C1 COLD 2024 | 5 | 290 | **63.79%** | **62.36%** | Kaggle GPU (T4) | 2.1 min | 9.29 MB | ✅ COMPLETE |
| **🌾 Sugarcane** | `sugarcane_unified_baseline.keras` | S1 + S2 Unified | 11 | 1,340 | **83.43%** | **83.29%** | Kaggle GPU (T4) | 9.8 min | 9.38 MB | ✅ COMPLETE |

---

## 8. Comparative & Cross-Dataset Evaluation (Phase 9)

**Status:** ✅ **100% COMPLETE**  
**Standalone Utility:** [`d:\CropDiseaseProject\evaluate_comparative.py`](file:///d:/CropDiseaseProject/evaluate_comparative.py)  
**Output Directory:** [`results/comparative_analysis/`](file:///d:/CropDiseaseProject/results/comparative_analysis/) (23 generated artifacts)

### Objective & Execution
The locked unified production models (`grape_unified_baseline.keras` and `sugarcane_unified_baseline.keras`) were evaluated in read-only inference mode against their constituent source-dataset test partitions using identical preprocessing:
- Grape Unified evaluated separately on:
  - **G1 (Niphad, Nashik — Regional):** 409 test samples
  - **G2 (Mendeley 2024 — Benchmark):** 522 test samples
- Sugarcane Unified evaluated separately on:
  - **S1 (Maharashtra — Regional):** 379 test samples
  - **S2 (Large Sugarcane — Benchmark):** 961 test samples

### Comparative Performance Table

| Crop | Dataset Scope / Partition | Samples | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Source |
|---|---|---:|---:|---:|---:|---:|---|
| **🍅 Tomato** | T1 PlantVillage (Full Baseline) | 2,180 | **90.23%** | 90.40% | 90.23% | **90.18%** | Existing Verified Baseline |
| **🍇 Grape** | Unified Full Test Set (G1+G2) | 931 | **89.04%** | 88.63% | 89.04% | **88.68%** | Existing Verified Unified Test |
| **🍇 Grape** | G1 Niphad, Nashik (Regional) | 409 | **95.35%** | 95.83% | 95.35% | **95.56%** | Unified Model on G1 Test Set |
| **🍇 Grape** | G2 Mendeley 2024 (Benchmark) | 522 | **84.10%** | 83.28% | 84.10% | **83.43%** | Unified Model on G2 Test Set |
| **🌶️ Chilli** | C1 COLD 2024 (Full Baseline) | 290 | **63.79%** | 64.74% | 63.79% | **62.36%** | Existing Verified Baseline |
| **🌾 Sugarcane** | Unified Full Test Set (S1+S2) | 1,340 | **83.43%** | 83.47% | 83.43% | **83.29%** | Existing Verified Unified Test |
| **🌾 Sugarcane** | S1 Maharashtra (Regional) | 379 | **80.47%** | 84.59% | 80.47% | **82.16%** | Unified Model on S1 Test Set |
| **🌾 Sugarcane** | S2 Large Sugarcane (Benchmark) | 961 | **84.60%** | 84.67% | 84.60% | **84.44%** | Unified Model on S2 Test Set |

### Key Research Findings
- **Grape Discrepancy:** G1 (Niphad) accuracy (95.35%) was **11.25 percentage points higher** than G2 (84.10%). This is driven by distinct disease pathologies: G1 diseases (Downy Mildew, Powdery Mildew, Healthy) separate cleanly, whereas G2 introduces severe mutual confusion between Leaf Blight and Esca.
- **Sugarcane Discrepancy:** S2 (Large) accuracy (84.60%) was **4.13 percentage points higher** than S1 (80.47%). S2 benefits from easily separable morphological pathologies (Sett Rot, Grassy Shoot), while S1 Red Rot foliar streaks infiltrate Brown Spot and Rust feature distributions.
- **Methodological Rule:** Performance differences represent *observed source-partition performance differences* rather than conclusive proof of domain shift.

---

## 9. In-Depth Error Analysis & Failure Modes (Phase 10)

**Status:** ✅ **100% COMPLETE**  
**Output Directory:** [`results/error_analysis/`](file:///d:/CropDiseaseProject/results/error_analysis/)  
**Primary Report:** [`results/error_analysis/error_analysis_report.txt`](file:///d:/CropDiseaseProject/results/error_analysis/error_analysis_report.txt)  
**Sample Catalog:** [`results/error_analysis/representative_error_samples.csv`](file:///d:/CropDiseaseProject/results/error_analysis/representative_error_samples.csv) (15 verified images)

### Verified Failure Patterns across Models
1. **Grape G2 Necrotic Lesion Confusion:**
   - **Leaf Blight $\to$ Esca (Black Measles):** 32 errors (highest in Grape).
   - **Esca $\to$ Leaf Blight:** 27 errors.
   - **Leaf Blight $\to$ Black Rot:** 13 errors.
   - *Cause:* Both Isariopsis Clavispora (Leaf Blight) and Esca exhibit irregular brown necrotic foliar lesions with chlorotic halos, confounding 2D CNNs at $224 \times 224$ resolution.
2. **Sugarcane Cross-Dataset Infiltration & Apical Convergence:**
   - **S1 Red Rot $\to$ Brown Spot:** 10 errors (S1 Red Rot leaves misclassified into S2-heavy Brown Spot).
   - **S1 Red Rot $\to$ Rust:** 8 errors.
   - **S2 Smut $\to$ Pokkah Boeng:** 14 errors (both deform the terminal spindle whorl silhouette).
   - **S2 Brown Spot $\to$ Yellow Leaf Disease:** 24 errors (generalized senescent chlorosis overrides focal spots).
   - **S1 Mosaic $\to$ Healthy Leaves:** 9 errors (faint viral mottling lacks high-contrast necrosis).
3. **Chilli COLD Majority-Class Attractor & Noise:**
   - **Nutritional Deficiency $\to$ Cercospora:** 19 errors (Nutritional Deficiency error rate: 75.0%).
   - **Healthy $\to$ Cercospora:** 13 errors.
   - *Cause:* Severe class imbalance (Cercospora is 46.5% of test data) creates an attractor effect; in addition, field sunlight reflections and soil dust mimic Cercospora spot features.
4. **Tomato Concentric Lesion Overlap:**
   - **Early Blight $\to$ Septoria Leaf Spot:** 13 errors.
   - **Early Blight $\to$ Late Blight:** 11 errors.
   - **Spider Mites $\to$ Target Spot:** 19 errors (dense feeding stippling mimics punctate target spots).

---

## 10. Standalone Grad-CAM Explainability Analysis (Phase 15)

**Status:** ✅ **100% COMPLETE**  
**Standalone Utility:** [`d:\CropDiseaseProject\gradcam_analysis.py`](file:///d:/CropDiseaseProject/gradcam_analysis.py)  
**Output Directory:** [`results/gradcam/`](file:///d:/CropDiseaseProject/results/gradcam/) (51 generated artifacts across 10 case folders)  
**Consolidated Report:** [`results/gradcam/gradcam_analysis_report.txt`](file:///d:/CropDiseaseProject/results/gradcam/gradcam_analysis_report.txt)

### Architecture & Technical Standard
- **Models Used:** All 4 locked production models (`tomato`, `grape_unified`, `chilli_cold`, `sugarcane_unified`).
- **Terminal Convolutional Layer:** `mobilenetv2_1.00_224.out_relu` (output of Layer 1, spatial tensor: $7 \times 7 \times 1280$).
- **Gradient Target:** Unnormalized pre-softmax class logits $\mathbf{z}_c = \mathbf{x} \mathbf{W}_c + b_c$, preventing saturation across high-confidence predictions.
- **Pipeline:** Channel-pooled gradients $\to$ ReLU rectification $\to$ Max-normalization $\to$ Bilinear upsampling to original photo size $\to$ Jet colormap blend ($\alpha = 0.45$).
- **Generation:** For every case, generated:
  1. `original.jpg` (Full-resolution source image)
  2. `gradcam_predicted_*.jpg` (Predicted class attention overlay)
  3. `gradcam_true_*.jpg` (Ground-truth class attention overlay)
  4. `comparison_panel.png` (Side-by-side: Original | Predicted Heatmap | True Heatmap)
  5. `metadata.txt` (Exact probabilities, layer parameters, and factual observations)

### All 10 Prioritized Diagnostic Cases Analyzed

| Case ID & Crop | Dataset Source | True Class | Predicted Class | Confidence | Output Subdirectory |
|---|---|---|---|---:|---|
| **Case 01** — Grape | Grape G2 | Leaf Blight | Esca (Black Measles) | **99.05%** | `grape_g2_leafblight_to_esca/` |
| **Case 02** — Grape | Grape G2 | Esca (Black Measles) | Leaf Blight | **97.16%** | `grape_g2_esca_to_leafblight/` |
| **Case 03** — Sugarcane | Sugarcane S1 | Red Rot | Brown Spot | **93.19%** | `sugarcane_s1_redrot_to_brownspot/` |
| **Case 04** — Sugarcane | Sugarcane S2 | Smut | Pokkah Boeng | **85.79%** | `sugarcane_s2_smut_to_pokkahboeng/` |
| **Case 05** — Sugarcane | Sugarcane S2 | Brown Spot | Yellow Leaf Disease | **95.89%** | `sugarcane_s2_brownspot_to_yellowleaf/` |
| **Case 06** — Chilli | Chilli C1 | Nutritional Deficiency | Cercospora Leaf Spot | **97.03%** | `chilli_c1_nutritional_deficiency_to_cercospora/` |
| **Case 07** — Tomato | Tomato T1 | Early Blight | Septoria Leaf Spot | **85.27%** | `tomato_t1_earlyblight_to_septoria/` |
| **Case 08** — Tomato | Tomato T1 | Spider Mites | Target Spot | **93.22%** | `tomato_t1_spidermites_to_targetspot/` |
| **Case 09** — Grape | Grape G1 | Bacterial Leaf Spot | Downy Mildew | **98.32%** | `grape_g1_bacterialspot_to_downymildew/` |
| **Case 10** — Sugarcane | Sugarcane S1 | Mosaic / Viral Disease | Healthy Leaves | **97.02%** | `sugarcane_s1_mosaic_to_healthy/` |

### Key Explainability Takeaways
1. **Foliar Grounding:** In 9 out of 10 cases, peak gradients aligned strictly with vegetative blade regions, necrotic lesions, or apical shoot distortions, confirming that models rely on agronomic features rather than background photographic shortcuts.
2. **Local vs Macro Attention:** In Grape G2 (Cases 1 & 2), the model focuses on isolated necrotic patches rather than global interveinal distributions.
3. **Background Chlorosis Dominance:** In Sugarcane S2 (Case 5), widespread chlorosis overpowers focal brown spots, driving prediction toward Yellow Leaf Disease.
4. **Interpretation Boundary:** Grad-CAM heatmaps represent explanatory evidence for selected instances rather than definitive proof of general network attention.

---

## 11. Web Application Architecture & Integration Status

A complete Flask web application is fully operational and actively serving live predictions across all 4 production crops.

### Web Application Architecture
- `run.py`: Application entry point (`http://127.0.0.1:5000`).
- `config.py`: `DevelopmentConfig` and `ProductionConfig` classes.
- `app/__init__.py`: Application factory (`create_app()`), initializes SQLite database and sets 16 MB upload limit.
- `app/routes.py`: Endpoints for UI pages and JSON APIs:
  - `GET /` : Main index page (Crop selection grid, Drag-and-drop Upload tab, WebRTC Camera tab).
  - `POST /predict` : Handles multi-part file uploads and base64 webcam captures, executes `app/predictor.py`, logs to SQLite, and renders results.
  - `GET /result/<prediction_id>` : Renders prediction details, confidence bar, per-class breakdown, and Grad-CAM container.
  - `GET /history` : Paginated prediction history log with thumbnails and delete actions.
  - `POST /predictions/<id>/delete` : Deletes record and associated image file from disk.
  - `GET /about` : Research context, dataset details, and live status of all 4 models.
  - `GET /api/status` : JSON endpoint reporting model readiness.
- `app/predictor.py`: Dedicated inference service layer:
  - Dynamically loads and caches `.keras` models from `models/<crop>/`.
  - Preprocesses input image to $224 \times 224$ with MobileNetV2 `preprocess_input`.
  - Returns structured `PredictionResult` object with predicted class, confidence percentage, and full probability distribution.
- `app/gradcam.py`: Existing application placeholder module. *(Note: The expanded Grad-CAM implementation from `gradcam_analysis.py` has not yet been ported into this file).*
- `app/database.py`: Thread-safe SQLite database manager (`instance/predictions.db`).
- `app/templates/`: Jinja2 templates (`base.html`, `index.html`, `result.html`, `history.html`, `about.html`, `error.html`).
- `app/static/`: Mobile-first responsive CSS (`main.css`) and JavaScript modules (`upload.js`, `camera.js`, `main.js`).

### Live Verification Status
- All 4 production models exist on disk and load cleanly under TensorFlow 2.18.1.
- `GET /api/status` returns `true` for all 4 models:
  ```json
  {"any_ready": true, "models": {"CHILLI": true, "GRAPE": true, "SUGARCANE": true, "TOMATO": true}}
  ```
- All evaluation result folders exist locally:
  - `results/tomato/`
  - `results/grape_unified/`
  - `results/chilli_cold/`
  - `results/sugarcane_unified/`
  - `results/comparative_analysis/`
  - `results/error_analysis/`
  - `results/gradcam/`

---

## 12. Project Lifecycle Phase Breakdown

```
[Phase 1: Research Foundation]            ✅ COMPLETED (Objectives, methodology & scope locked)
[Phase 2: Dataset Acquisition]            ✅ COMPLETED (6 datasets, 31,590 images verified on D:)
[Phase 3: Dataset Organization]           ✅ COMPLETED (Cleaned, verified, zero corruption)
[Phase 4: Dataset Analysis]               ✅ COMPLETED (dataset_analysis.py, CSV catalogs)
[Phase 5: Experimental Design]            ✅ COMPLETED (4 unified production crops + 6 research splits)
[Phase 6: Dataset Splitting]              ✅ COMPLETED (70/15/15 stratified splits generated)
[Phase 7: ML Environment Setup]          ✅ COMPLETED (Python 3.11.9, TF 2.18.1, Keras 3.15.1)
[Phase 8: Model Training (All 4 Crops)]   ✅ COMPLETED (Tomato, Grape, Chilli, Sugarcane)
[Phase 9: Comparative Analysis]           ✅ COMPLETED (G1 vs G2, S1 vs S2, evaluate_comparative.py)
[Phase 10: In-Depth Error Analysis]       ✅ COMPLETED (Confusion patterns, error_analysis_report.txt)
[Phase 11-14: Web App Foundation]         ✅ COMPLETED (Flask, SQLite, UI, Predictor service)
[Phase 15: Grad-CAM Research Utility]     ✅ COMPLETED (gradcam_analysis.py, 10 diagnostic cases)
[Phase 16: Multi-Model Smoke Tests]       ✅ COMPLETED (All 4 models live, smoke tests verified)
─────────────────────────────────────────────────────────────────────────────
[Phase 15B: Application Grad-CAM Port]    🔜 NEXT ACTIVE PHASE (Integrate into app/gradcam.py & UI)
[Phase 16B: End-to-End Validation]        🔜 PENDING (Full UI validation with Grad-CAM overlays)
[Phase 17: Production Packaging]          🔜 PENDING (Prod config, secret key, environment isolation)
[Phase 20: Research Thesis Chapters]      🔜 PENDING (Chapters 5, 6, 7, 8, 9)
```

---

## 13. Post-Training Roadmap & Exact Remaining Phases

The project has completed all core model training, cross-dataset comparative evaluations, error analyses, and standalone Grad-CAM diagnostics. Follow this exact remaining sequence:

1. **Phase 15B — Application Grad-CAM Integration (Next Immediate Step):**
   - Port the validated `GradCAMAnalyzer` logic from `gradcam_analysis.py` into `app/gradcam.py`.
   - Update `app/predictor.py` to optionally generate a Grad-CAM heatmap overlay during live inference.
   - Ensure heatmaps are saved in `app/static/uploads/` and dynamically rendered in `app/templates/result.html`.
   - Maintain sub-second inference responsiveness.

2. **Phase 16B — Final End-to-End Application Validation:**
   - Test full user flows (upload and live camera capture) across all 4 crops on desktop and mobile viewports.
   - Verify that Grad-CAM heatmaps display correctly alongside class probabilities and confidence meters.
   - Validate prediction history logging and deletion workflows in SQLite.

3. **Phase 17 — Production Configuration:**
   - Finalize `config.py` production settings (e.g. `SECRET_KEY` environment variable enforcement, database location).
   - Document application startup procedures for production serving (e.g. Gunicorn / Waitress).

4. **Phase 20 — Research Documentation / Thesis Chapters:**
   - Chapter 5: Model Architecture & Transfer Learning Pipeline.
   - Chapter 6: Experimental Results & Comparative Performance Tables (incorporating Phase 9 metrics).
   - Chapter 7: Web Application Design & Explainable AI Implementation (incorporating Phase 15 Grad-CAM).
   - Chapter 8: Discussion, Error Analysis & Agronomic Interpretability (incorporating Phase 10 findings).
   - Chapter 9: Conclusion, Limitations & Future Work.

---

## 14. Non-Negotiable Constraints & Architecture Rules

1. **Do Not Retrain Completed Production Models:**
   All 4 production models (`tomato`, `grape_unified`, `chilli_cold`, `sugarcane_unified`) are trained, verified, and locked. No retraining or fine-tuning is required.
2. **Exactly 4 Production Models:**
   The production deployment layer maintains exactly 4 dedicated crop models (Tomato, Grape Unified, Chilli, Sugarcane Unified). Do not create separate production models for G1, G2, S1, or S2.
3. **Preserve Raw Dataset & Splitting Integrity:**
   The 6 local datasets (31,590 images total) and established 70/15/15 stratified split CSVs are ground truth. Never modify source images, re-split, or overwrite ground truth CSVs.
4. **Uniform MobileNetV2 ML Pipeline:**
   All research comparisons rely on the frozen MobileNetV2 ImageNet-pretrained backbone with identical classification head hyperparameters.
5. **Defensible Research Language:**
   Document observed cross-dataset performance differences and Grad-CAM visualizations objectively. Do not state that dataset discrepancies prove domain shift, nor claim that Grad-CAM heatmaps prove biological causation.

---

## 15. File & Repository Structure Reference

```
D:\CropDiseaseProject/
├── .gitignore
├── PROJECT_HANDOFF.md                 <-- Master single source of truth document (v1.4)
├── config.py                          <-- Flask configuration (Dev / Prod)
├── run.py                             <-- Web application entry point
├── dataset_analysis.py                <-- Dataset verification script
├── dataset_preparation.py             <-- Dataset validation & splitting script
├── dataset_split.py                   <-- Stratified split generator
├── dataset_summary.csv                <-- Class-by-class image counts
├── dataset_file_list.csv              <-- Master list of all 31,590 files
├── verify_environment.py              <-- ML environment verification script
├── train_tomato.py                    <-- T1 Tomato baseline training script
├── train_experiment.py                <-- Parameterized training script (Grape, Chilli, Sugarcane)
├── evaluate_comparative.py            <-- Phase 9 comparative evaluation utility
├── gradcam_analysis.py                <-- Phase 15 standalone Grad-CAM explainability utility
├── trained_models_results.zip         <-- Kaggle GPU training package archive (25.3 MB)
│
├── notebooks/                         <-- GPU Training Notebooks
│   └── train_kaggle_colab.ipynb       <-- Kaggle/Colab GPU training & evaluation notebook
│
├── app/                               <-- Flask Web Application
│   ├── __init__.py                    <-- App factory
│   ├── database.py                    <-- SQLite database operations
│   ├── gradcam.py                     <-- Application Grad-CAM module (to be ported from Phase 15)
│   ├── predictor.py                   <-- 4-crop inference service layer
│   ├── routes.py                      <-- Web routes and endpoints
│   ├── static/
│   │   ├── css/main.css               <-- Mobile-first stylesheet
│   │   ├── js/camera.js               <-- WebRTC Camera capture handler
│   │   ├── js/main.js                 <-- Main interactive UI logic
│   │   ├── js/upload.js               <-- Drag-and-drop & validation logic
│   │   ├── img/favicon.svg
│   │   └── uploads/                   <-- Uploaded & captured leaf images
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
├── models/                            <-- 4 Trained Production .keras Models
│   ├── tomato/
│   │   ├── model_summary.txt
│   │   └── tomato_baseline.keras      <-- Trained Tomato model (9.37 MB)
│   ├── grape_unified/
│   │   ├── model_summary.txt
│   │   └── grape_unified_baseline.keras <-- Trained Grape model (9.32 MB)
│   ├── chilli_cold/
│   │   ├── model_summary.txt
│   │   └── chilli_cold_baseline.keras   <-- Trained Chilli model (9.29 MB)
│   └── sugarcane_unified/
│       ├── model_summary.txt
│       └── sugarcane_unified_baseline.keras <-- Trained Sugarcane model (9.38 MB)
│
├── results/                           <-- Complete Result Suites & Analysis Outputs
│   ├── tomato/                        <-- Baseline metrics, report, confusion matrix, history
│   ├── grape_unified/                 <-- Baseline metrics, report, confusion matrix, history
│   ├── chilli_cold/                   <-- Baseline metrics, report, confusion matrix, history
│   ├── sugarcane_unified/             <-- Baseline metrics, report, confusion matrix, history
│   ├── comparative_analysis/          <-- Phase 9 comparative metrics, reports, confusion matrices
│   ├── error_analysis/                <-- Phase 10 error summary, class analyses, sample catalog
│   └── gradcam/                       <-- Phase 15 diagnostic case directories & consolidated report
│
├── splits/                            <-- Stratified Train/Val/Test CSVs
│   ├── tomato/                        <-- 10 classes, 14,529 images
│   ├── grape_unified/                 <-- 7 classes, 6,203 images
│   ├── chilli_cold/                   <-- 5 classes, 1,932 images
│   ├── sugarcane_unified/             <-- 11 classes, 8,926 images
│   ├── grape_niphad/                  <-- Research baseline split (4 classes)
│   ├── grape_2024/                    <-- Research baseline split (4 classes)
│   ├── sugarcane_maharashtra/         <-- Research baseline split (5 classes)
│   └── sugarcane_large/               <-- Research baseline split (10 classes)
│
└── [Source Dataset Directories]       <-- 6 Verified Raw Datasets (~2.05 GB total)
    ├── tomato_plantvillage/
    ├── grape_niphad/
    ├── grape_2024/
    ├── chilli_cold/
    ├── sugarcane_maharashtra/
    └── sugarcane_large/
