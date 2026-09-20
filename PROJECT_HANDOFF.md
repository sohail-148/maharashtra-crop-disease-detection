# Crop Disease Detection Project — Master Handoff Package

**Document Version:** 1.8  
**Project Workspace:** `D:\CropDiseaseProject`  
**Git Repository:** `https://github.com/sohail-148/maharashtra-crop-disease-detection` (Branch: `main`)  
**Status as of Handoff:** Completed Phases 1–8 (Model Training & Verification), Phase 9 (Comparative / Cross-Dataset Evaluation), Phase 10 (In-Depth Error Analysis), Phase 15 (Standalone Grad-CAM Explainability Analysis on 10 Prioritized Cases), Phase 16 (Comprehensive 4-Track Model Robustness Audit across 2,556 Inferences), the complete Chilli Research Cycle (Experiments 1–5), Phase 18, and Phase 19A:
- **Chilli Experiments 1–3:** Candidates A, B, and C evaluated on COLD; Candidate B achieved **73.45% accuracy**, **72.31% Weighted F1**, and **68.62% Macro F1** on the official test split as the leading in-domain research candidate.
- **Chilli Experiment 4:** Candidate D trained with mixed-domain data (COLD 2024 + Ulfa 2023 field data, total $n=2,152$) achieved **68.62% official test Accuracy**, **67.90% official test Weighted F1**, and **64.13% official test Macro F1**; on external/field benchmarks Candidate D achieved **92.50% on Track B External** (+40.00% over Candidate B), **81.44% on Track C Real-World** (+26.80% over Candidate B), and **76.62% on Track C High-Quality** (+32.46% over Candidate B).
- **Chilli Experiment 5:** Independent field-source validation using an unseen PlantDoc subset ($n=115$: 62 Cercospora, 53 Healthy; Murda Complex, Nutritional Deficiency, and Powdery Mildew strictly unrepresented). Candidate D achieved **45.22% Accuracy**, **56.14% Weighted F1**, and **56.26% represented-class Macro F1** (+10.12% Weighted F1 over Baseline, +6.39% over Candidate B).
- **Phase 18 (Completed in Commit `23f26ea`):** Clean real-world multi-crop benchmark ($n=1,172$) executed across Tomato, Grape, Sugarcane, and Chilli; Candidate D integrated into Flask for Chilli inference ($71.08\%$ real-world accuracy across $332$ images vs $50.60\%$ Baseline); runtime Grad-CAM visual attention integrated into the Flask web UI; production baselines for Tomato, Grape, and Sugarcane retained.
- **Phase 19A (Completed):** End-to-end demonstration and software verification of the Flask application completed across all 18 dimensions (startup, home page, `/api/status`, all 4 model routes with Chilli -> Candidate D, 4-crop predictions, runtime Grad-CAM, WEBP upload, server-side base64 camera pipeline, invalid/corrupt input error handling, deletion + file cleanup, protected model/dataset integrity).

Next active research phase is Phase 19B (Cross-Crop Generalization, Two-Stage Gatekeeper, and OOD Safeguard Architecture).

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
> *How effectively can a lightweight MobileNetV2 transfer-learning framework classify diseases across multiple Maharashtra-relevant crop datasets, and how robustly do these models generalize across independent external benchmarks, real-world field photography, and out-of-distribution inputs?*

### Supporting Research Questions
1. How accurately does MobileNetV2 classify diseases for each crop under controlled in-domain splits?
2. How does performance vary between distinct datasets for the same crop when evaluated on unified models (Niphad Grape vs Grape 2024, Maharashtra Sugarcane vs Large Sugarcane)?
3. Which disease classes are difficult to classify, and what visual or data factors contribute to the observed confusions?
4. How does class imbalance affect class-level precision, recall, and F1-scores?
5. Can Grad-CAM provide meaningful, agronomically plausible visual explanations of model focus on both correct and incorrect predictions?
6. How robust are the models when evaluated against independent external benchmarks and uncontrolled real-world field photography?
7. What are the observed failure modes when models receive out-of-distribution inputs such as cross-crop leaves or non-leaf objects?
8. Can controlled architectural interventions (such as partial backbone fine-tuning, field-oriented augmentation, and class weighting) improve classification performance on challenging crops like Chilli?

### Key Contributions & Research Defense
We do **not** claim individual novelty in MobileNetV2, transfer learning, or Grad-CAM. The defensible contribution is the rigorous combination, empirical evaluation, and robustness auditing:
$$\text{Maharashtra crop focus} + \text{6 regional datasets} + \text{Uniform MobileNetV2 pipeline} + \text{Comparative evaluation} + \text{Grad-CAM explainability} + \text{4-Track Robustness Audit} + \text{Controlled Model Improvement Experiments}$$

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
1. **Production Deployment Layer (Exactly 4 Dedicated Crop Models — LOCKED):**  
   For real-world agricultural usability, the web application maps the 4 crops directly to 4 locked production models:
   - **Tomato:** T1 Baseline Model (10 classes | 14,529 images — **Verified: 90.23% accuracy, 90.18% F1**)
   - **Grape:** Grape Unified G1+G2 Model (7 canonical classes | 6,203 images — **Verified: 89.04% accuracy, 88.68% F1**)
   - **Chilli:** Chilli C1 Model (5 canonical classes | 1,932 images — **Verified: 63.79% accuracy, 62.36% F1**)
   - **Sugarcane:** Sugarcane Unified S1+S2 Model (11 canonical classes | 8,926 images — **Verified: 83.43% accuracy, 83.29% F1**)
   - *Total Deployment Scope:* **33 canonical classes across 31,590 images**.
   - *Integrity Rule:* The baseline model files in `models/` remain strictly read-only and locked.

2. **Research Evaluation & Experimentation Layer:**  
   - Independent dataset partitions (G1, G2, S1, S2) are maintained for comparative reporting.
   - The 4-Track Model Robustness Audit evaluated external research datasets (PlantDoc, Pakistan Field, GLDD, AI Challenger, Ulfa, Sabidarrow) and OpenCV benchmarks.
   - The Chilli Model Improvement Campaign is maintained separately in `experiments/chilli_improvement/`. Candidate models remain research assets and are **not** promoted to production without explicit verification.

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

---

## 5. Verified Production Baseline Model Performances

| Crop | Architecture & Head | Test Images | Test Accuracy | Weighted F1 | Macro F1 | Model File |
|---|---|---:|---:|---:|---:|---|
| **Tomato** | MobileNetV2 (frozen) + GAP + BN + Dropout(0.3) + Dense(10) | 2,185 | **90.23%** | **90.18%** | **89.54%** | `models/tomato/tomato_baseline.keras` |
| **Grape Unified** | MobileNetV2 (frozen) + GAP + BN + Dropout(0.3) + Dense(7) | 933 | **89.04%** | **88.68%** | **87.21%** | `models/grape_unified/grape_unified_baseline.keras` |
| **Chilli** | MobileNetV2 (frozen) + GAP + BN + Dropout(0.3) + Dense(5) | 290 | **63.79%** | **62.36%** | **57.75%** | `models/chilli_cold/chilli_cold_baseline.keras` |
| **Sugarcane Unified** | MobileNetV2 (frozen) + GAP + BN + Dropout(0.3) + Dense(11) | 1,343 | **83.43%** | **83.29%** | **82.11%** | `models/sugarcane_unified/sugarcane_unified_baseline.keras` |

---

## 6. Phase 9 — Comparative Cross-Dataset Evaluation Summary

- **Grape Unified Model on Regional Partitions:**
  - G1 (Niphad, Nashik, 409 test samples): **95.35% Accuracy**, **95.56% Weighted F1**.
  - G2 (Grape 2024, 524 test samples): **84.12% Accuracy**, **83.31% Weighted F1**.
  - Observed Partition Difference: **11.23% accuracy difference** ($p < 0.001$).
- **Sugarcane Unified Model on Regional Partitions:**
  - S1 (Maharashtra, 381 test samples): **86.61% Accuracy**, **86.43% Weighted F1**.
  - S2 (Large Sugarcane, 962 test samples): **82.17% Accuracy**, **82.05% Weighted F1**.
  - Observed Partition Difference: **4.44% accuracy difference**.

---

## 7. Phase 10 — In-Depth Error Analysis Summary

- Total Test Predictions Evaluated: **4,751 samples**.
- Overall Test Accuracy across all splits: **85.73%** (678 total misclassifications).
- Systematic Confusion Clusters:
  - Grape G2: Leaf Blight $\leftrightarrow$ Esca (Black Measles) mutual confusion (24% of all G2 errors).
  - Chilli C1: Nutritional Deficiency $\to$ Cercospora Leaf Spot (47.5% class error rate).
  - Sugarcane S2: Smut $\to$ Pokkah Boeng (40% class error rate).
  - Sugarcane S1: Red Rot $\to$ Brown Spot (28.9% class error rate).
  - Tomato T1: Spider Mites $\to$ Target Spot (34.8% of all Spider Mite errors).

---

## 8. Phase 15 — Standalone Grad-CAM Explainability Analysis

- Evaluated all 10 prioritized diagnostic error cases from Phase 10 using `gradcam_analysis.py`.
- Generated 51 visual artifacts in `results/gradcam/`.
- Terminal layer: `mobilenetv2_1.00_224.out_relu` ($7 \times 7 \times 1280$). Target: Pre-softmax logits $\mathbf{z}_c$.
- Key Finding: In 9 out of 10 cases, model attention aligned strictly with foliar blade regions, necrotic spots, or apical distortion, confirming that predictions were grounded in foliar features rather than background photographic shortcuts on the in-domain test sets.

---

## 9. Phase 16 — Comprehensive 4-Track Model Robustness Audit

**Objective:** Empirically evaluate the locked production models across four complementary tracks to measure generalization boundaries, domain sensitivity, and out-of-distribution behavior.  
**Total Inferences Evaluated:** **2,556 Empirical Predictions**.  
**Deliverables Directory:** [`results/model_robustness_audit/`](file:///D:/CropDiseaseProject/results/model_robustness_audit/)

### 4-Track Empirical Summary Table

| Audit Track | Scope & Dataset Sources | Samples | Headline Accuracy | Mean Conf (Errors) | High-Conf Errors ($\ge 80\%$) |
|---|---|:---:|:---:|:---:|:---:|
| **Track A: Internal Held-Out Test** | Stratified test splits ($seed=42$, up to 30/class) | **1,119** | **82.75%** | 64.9% | 23.8% |
| **Track B: Independent External Labeled** | Research datasets (PlantDoc, Pakistan, GLDD, Baliwaka, Ulfa, Sabidarrow) | **632** | **24.05%** | **78.6%** | **58.3%** |
| **Track C: Real-World Field User-Style** | Farmer mobile photos (weeds, complex lighting, hands, shadows) | **425** | **29.27%** *(High)* | **80.5%** | **59.8%** |
| **Track D: Out-of-Distribution Stress** | Cross-crop leaves (300) + OpenCV non-leaf benchmark objects (80) | **380** | **0.0%** *(All OOD)* | **76.9%** | **54.2%** |

### Per-Crop Performance Degradation Across Tracks

```
Crop           Track A (Internal)   Track B (External)   Track C (Field High)   Observed Difference (A - B)
Tomato (10 cls)      88.00%               14.34%                30.92%                    -73.66%
Grape  (7 cls)       88.00%               13.33%                10.00%                    -74.67%
Chilli (5 cls)       56.25%               55.00%                50.65%                     -1.25%
Sugarcane (11 cls)   82.00%               38.33%                30.00%                    -43.67%
```

### Empirical Observations & Generalization Analysis

1. **Observed Performance Discrepancies in Tomato and Grape:**
   - On the independent external benchmark (Track B), Tomato achieved 14.34% accuracy and Grape achieved 13.33% accuracy, representing substantial performance reductions relative to their held-out test splits (88.00%).
   - In Tomato, external non-blight samples (such as Healthy, Leaf Mold, Target Spot, Mosaic Virus, and Septoria) were frequently predicted as Early Blight or Late Blight with high confidence (mean confidence on errors: 84.1%). This observed behavior is consistent with sensitivity to background visual context, as the training set (PlantVillage) features artificial laboratory backgrounds while external benchmarks feature natural outdoor field backgrounds.
   - In Grape, external diseased samples frequently were predicted as Healthy Leaves or Powdery Mildew, consistent with sensitivity to camera-specific characteristics and lighting between different collection setups.

2. **Domain Alignment in Chilli:**
   - Chilli exhibited stable performance across domains: 56.25% on Track A, 55.00% on Track B, and 50.65% on Track C (High Quality).
   - Because the Chilli training corpus was collected under field conditions (Krishna River Basin), it showed no drop when evaluated against external field photos. However, baseline diagnostic accuracy remains moderate.

3. **Out-of-Distribution (OOD) Overconfidence Behavior (Track D):**
   - In closed-set softmax classification without an explicit rejection threshold, the models assign out-of-distribution inputs into the trained classes.
   - When presented with unrelated non-leaf benchmark images (e.g. vehicles, animals, furniture), the models produced high-confidence predictions:
     * Tomato model: Mean confidence 90.72% (80.0% of samples at $\ge 80\%$ confidence), most frequently predicting Early Blight.
     * Grape model: Mean confidence 83.25% (65.0% of samples at $\ge 80\%$ confidence), most frequently predicting Healthy Leaves.
     * Chilli model: Mean confidence 75.08% (55.0% of samples at $\ge 80\%$ confidence), most frequently predicting Cercospora Leaf Spot.
     * Sugarcane model: Mean confidence 71.13% (35.0% of samples at $\ge 80\%$ confidence), most frequently predicting Mosaic / Viral Disease.
   - When presented with cross-crop leaves (e.g. Sugarcane leaves input to the Tomato model), the Tomato model predicted Tomato Yellow Leaf Curl Virus with 94.53% mean confidence (max 100.0%).
   - *Takeaway:* Raw softmax confidence is not a reliable indicator of prediction accuracy under out-of-distribution conditions.

---

## 10. Controlled Model Improvement Experiments: Chilli Experiments 1–3 (COLD C1)

**Objective:** Experimentally determine whether controlled training modifications can improve Chilli classification performance on the COLD dataset, particularly on weak classes (Nutritional Deficiency, Murda Complex, Healthy), without compromising baseline integrity.  
**Production Baseline Status:** `models/chilli_cold/chilli_cold_baseline.keras` remains **LOCKED, UNTOUCHED, and SHA-256 VERIFIED**.  
**Experiment Directory:** [`experiments/chilli_improvement/`](file:///D:/CropDiseaseProject/experiments/chilli_improvement/)

### Candidate Configurations (Single-Source COLD Training)
- **Candidate A (Partial Fine-Tuning):**
  - MobileNetV2 base: Lower 125 layers (blocks 1–13) frozen; upper 29 layers (blocks 14–16, `Conv_1`, `out_relu`) unfrozen.
  - Base BatchNormalization layers kept frozen to stabilize running statistics.
  - Optimizer: Adam with $\text{LR} = 1.0 \times 10^{-4}$ (10× smaller than baseline head LR of $10^{-3}$).
  - Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.2, patience=3). Max epochs: 30.
- **Candidate B (Fine-Tuning + Realistic Field Augmentation):**
  - Same partial fine-tuning setup as Candidate A.
  - Added realistic field augmentations: random horizontal flip ($p=0.5$), moderate zoom/crop ($85\%-100\%$), moderate brightness ($\pm 15\%$), moderate contrast ($0.85-1.15$), values clipped to $[-1.0, 1.0]$.
- **Candidate C (Fine-Tuning + Augmentation + Balanced Class Weighting):**
  - Same setup as Candidate B with balanced class weights computed exclusively from `train.csv`:
    `cerocospora`: 0.4299, `healthy`: 1.1757, `murda complex`: 1.4010, `nutritional deficiency`: 1.4460, `powdery mildew`: 2.3929.

### Official Test Split Results ($n = 290$)

| Model | Test Accuracy | Weighted F1 | Macro F1 | Accuracy Delta ($\Delta \text{Acc}$) | Weighted F1 Delta ($\Delta \text{F1}$) | Status |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline (Frozen)** | $63.79\%$ | $62.36\%$ | $57.75\%$ | *Reference* | *Reference* | **Active Production Model** |
| **Candidate A (Fine-Tune)** | $71.38\%$ | $70.33\%$ | $66.47\%$ | **$+7.59\%$** | **$+7.97\%$** | Research Candidate |
| **Candidate B (+Real Aug)** | **$73.45\%$** | **$72.31\%$** | **$68.62\%$** | **$+9.66\%$** | **$+9.95\%$** | **Leading In-Domain Candidate** |
| **Candidate C (+Class Wgt)** | $71.38\%$ | $71.14\%$ | $67.94\%$ | **$+7.59\%$** | **$+8.78\%$** | Research Candidate |

### Class-Level F1 Progression on Official Test Split

| Disease Class | Support | Baseline F1 | Candidate A F1 | Candidate B F1 | Candidate C F1 | Candidate B Delta |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cercospora Leaf Spot** | 135 | $75.17\%$ | $80.81\%$ | **$83.22\%$** | $81.65\%$ | **$+8.05\%$** |
| **Healthy Leaves** | 50 | $52.63\%$ | $55.17\%$ | **$64.65\%$** | $60.55\%$ | **$+12.02\%$** |
| **Murda Complex (Leaf Curl)** | 41 | $50.00\%$ | $64.37\%$ | **$65.06\%$** | $60.47\%$ | **$+15.06\%$** |
| **Nutritional Deficiency** | 40 | $37.04\%$ | **$57.58\%$** | $44.44\%$ | $50.75\%$ | **$+7.40\%$** |
| **Powdery Mildew** | 24 | $73.91\%$ | $74.42\%$ | $85.71\%$ | **$86.27\%$** | **$+11.80\%$** |

### Secondary Generalization Evaluation (Audit Datasets)

*Evaluated against held-out audit samples without retraining:*

| Evaluation Dataset | Samples | Baseline Accuracy | Candidate B Accuracy | Observed Difference |
|---|:---:|:---:|:---:|:---:|
| **Track B: Independent External** (Ulfa benchmark) | 120 | **$55.00\%$** | $52.50\%$ | $-2.50\%$ |
| **Track C: Real-World Field** (All qualities) | 97 | **$57.73\%$** | $54.64\%$ | $-3.09\%$ |
| **Track C: Real-World Field** (High-Quality only) | 77 | **$50.65\%$** | $44.16\%$ | $-6.49\%$ |

### Key Research Findings (Experiments 1–3):
1. **Observed In-Domain Improvement:** Candidate B improved overall test accuracy by $+9.66\%$ ($63.79\% \to 73.45\%$) and Macro F1 by $+10.87\%$ ($57.75\% \to 68.62\%$), with observed gains across all 5 classes.
2. **Cross-Domain Specialization Trade-off:** Candidate B did not improve accuracy on independent external field samples ($52.50\%$ vs baseline $55.00\%$ on Track B, and $54.64\%$ vs baseline $57.73\%$ on Track C). This observed difference is consistent with domain specialization when optimizing strictly on single-source data.
3. **Deployment Status:** **Candidate B is a research asset, NOT a production model.** The production model remains `models/chilli_cold/chilli_cold_baseline.keras`.

---

## 10B. Chilli Experiment 4: Mixed-Domain Field Training (Candidate D)

**Objective:** Test whether combining in-domain training data with curated external field imagery can improve model performance on field benchmarks while maintaining competitive accuracy on the in-domain test split.  
**Training Configuration:** Mixed-domain training using **COLD 2024 + Ulfa 2023 field data** ($n = 2,152$ total: 1,352 COLD training samples + 800 curated field samples).  
*Data Integrity:* Track B ($n=120$) and Track C ($n=97$) evaluation splits were strictly excluded from the training loader/manifest.  
**Experiment Directory:** [`experiments/chilli_field_experiment/`](file:///D:/CropDiseaseProject/experiments/chilli_field_experiment/)  
**Candidate D Deliverables:** `experiments/chilli_field_experiment/candidate_d/model.keras` (21.8 MB), `training_history.csv`, `candidate_d_comparison.csv`.

### Candidate D Performance Comparison

| Evaluation Benchmark | Benchmark Type | Baseline (Frozen) | Candidate B (Fine-Tune) | Candidate D (Mixed-Domain) | Delta (D vs Baseline) | Delta (D vs Cand B) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **COLD Official Test ($n=290$)** | In-Domain Held-Out | $63.79\%$ | **$73.45\%$** | $68.62\%$ | $+4.83\%$ | $-4.83\%$ |
| **COLD Test Weighted F1** | In-Domain Held-Out | $62.36\%$ | **$72.31\%$** | $67.90\%$ | $+5.54\%$ | $-4.41\%$ |
| **COLD Test Macro F1** | In-Domain Held-Out | $57.75\%$ | **$68.62\%$** | $64.13\%$ | $+6.38\%$ | $-4.49\%$ |
| **Track B External ($n=120$)** | External Field (Ulfa) | $55.00\%$ | $52.50\%$ | **$92.50\%$** | **$+37.50\%$** | **$+40.00\%$** |
| **Track C Real-World All ($n=97$)** | Wild Multi-Source Field | $57.73\%$ | $54.64\%$ | **$81.44\%$** | **$+23.71\%$** | **$+26.80\%$** |
| **Track C Real-World High ($n=77$)** | Clean Multi-Source Field | $50.65\%$ | $44.16\%$ | **$76.62\%$** | **$+25.97\%$** | **$+32.46\%$** |

### Key Research Findings (Experiment 4):
1. **Observed Field Improvement:** Candidate D demonstrated substantial observed accuracy improvements on Track B ($92.50\%$, $+40.00\%$ over Candidate B) and Track C ($81.44\%$, $+26.80\%$ over Candidate B).
2. **Balanced Performance:** While in-domain official test accuracy was $4.83\%$ lower than Candidate B ($68.62\%$ vs $73.45\%$), Candidate D maintained a $+4.83\%$ accuracy and $+5.54\%$ Weighted F1 improvement over the Production Baseline.
3. **Evidence Assessment:** This provides evidence consistent with improved field feature representations when field data is included in training. However, because Track B uses the same underlying source family (Ulfa et al.) as the field training component, generalization across all five canonical classes remains incompletely validated (and was further assessed in Experiment 5).

---

## 10C. Chilli Experiment 5: Independent Field-Source Validation (PlantDoc Subset)

**Objective:** Evaluate whether Candidate D's observed field performance advantages replicate when tested against a third, unseen, independent field-data source under strict zero-leakage conditions.  
**Validation Dataset:** PlantDoc Bell Pepper Field Subset (Singh et al., CoDS-COMAD 2020), $n = 115$ images.  
**Experiment Directory:** [`experiments/chilli_independent_validation/`](file:///D:/CropDiseaseProject/experiments/chilli_independent_validation/)

### Mandatory Class Coverage & Scope Disclosure
> [!IMPORTANT]
> **Partial Class Representation in Experiment 5**
> - **Represented Classes ($n=115$):**
>   - Cercospora Leaf Spot: 62 images (53.91%)
>   - Healthy Leaves: 53 images (46.09%)
> - **Unrepresented Classes ($n=0$):**
>   - Murda Complex: **NOT REPRESENTED IN THIS VALIDATION DATASET**
>   - Nutritional Deficiency: **NOT REPRESENTED IN THIS VALIDATION DATASET**
>   - Powdery Mildew: **NOT REPRESENTED IN THIS VALIDATION DATASET**
>
> Experiment 5 covers only two of the five canonical classes and therefore is **NOT a complete five-class field benchmark**. Because three classes are unrepresented, generalization remains incompletely validated.

### Zero-Leakage & Independence Audit:
- Exact SHA-256 collision check against all project splits (Train, Val, Test, Candidate D manifest, Track B, Track C): **0 collisions**.
- Perceptual dHash check (pure NumPy, Hamming distance $\le 5$): **0 near-duplicates**.

### Independent Field Evaluation Results ($n = 115$)

| Metric | Baseline (Frozen) | Candidate B (Fine-Tune+Aug) | Candidate D (Mixed-Domain) | Delta (D vs Baseline) | Delta (D vs Cand B) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Validation Accuracy ($n=115$)** | $43.48\%$ | $40.00\%$ | **$45.22\%$** | **$+1.74\%$** | **$+5.22\%$** |
| **Weighted Precision** | $52.01\%$ | $67.83\%$ | **$74.86\%$** | **$+22.85\%$** | **$+7.03\%$** |
| **Weighted Recall** | $43.48\%$ | $40.00\%$ | **$45.22\%$** | **$+1.74\%$** | **$+5.22\%$** |
| **Weighted F1 Score** | $46.02\%$ | $49.75\%$ | **$56.14\%$** | **$+10.12\%$** | **$+6.39\%$** |
| **Macro F1 (Represented 2 Classes)** | $45.19\%$ | $49.73\%$ | **$56.26\%$** | **$+11.07\%$** | **$+6.53\%$** |
| **Macro F1 (All 5 Canonical Classes)** | $18.08\%$ | $19.89\%$ | **$22.50\%$** | **$+4.43\%$** | **$+2.61\%$** |
| **  * Cercospora Leaf Spot F1 ($n=62$)** | **$55.81\%$** | $50.00\%$ | $54.74\%$ | $-1.08\%$ | **$+4.74\%$** |
| **  * Healthy Leaves F1 ($n=53$)** | $34.57\%$ | $49.46\%$ | **$57.78\%$** | **$+23.21\%$** | **$+8.32\%$** |
|   * Murda Complex ($n=0$) | *Unrepresented* | *Unrepresented* | *Unrepresented* | N/A | N/A |
|   * Nutritional Deficiency ($n=0$) | *Unrepresented* | *Unrepresented* | *Unrepresented* | N/A | N/A |
|   * Powdery Mildew ($n=0$) | *Unrepresented* | *Unrepresented* | *Unrepresented* | N/A | N/A |

### Observed Diagnostic Signatures & Error Analysis:
1. **Baseline Model:** Exhibited high false-positive Cercospora predictions on healthy field leaves (31 out of 53 healthy leaves misdiagnosed as Cercospora, $58.49\%$ false-positive rate), yielding a Healthy F1 of $34.57\%$.
2. **Candidate B:** Reduced false Cercospora predictions (down to 7), but shifted false predictions toward Nutritional Deficiency (20 out of 53 healthy leaves misdiagnosed as Nutritional Deficiency, $37.74\%$), consistent with chromatic sensitivity to outdoor sunlight variation.
3. **Candidate D:** Demonstrated observed improvement on healthy leaf discrimination (Healthy F1: $57.78\%$, $+23.21\%$ over Baseline and $+8.32\%$ over Candidate B; Healthy Precision: $70.27\%$), while reducing false Nutritional Deficiency predictions by $50\%$ relative to Candidate B. Candidate D exhibited some off-target predictions of Murda Complex ($n=26$), consistent with field leaf margin curvature resembling viral curl symptoms.
4. **Overall Assessment:** Candidate D achieved the highest Weighted F1 ($56.14\%$) and 2-class Macro F1 ($56.26\%$) among evaluated models on this independent field source. These results provide evidence consistent with improved field generalization for the represented classes, while overall accuracy ($45.22\%$) highlights that open-field classification on unsegmented images remains challenging.

---

## 10D. Cross-Benchmark Synthesis & Production Trade-Off Analysis

### Complete Evaluation Trajectory Across All Chilli Benchmarks

| Benchmark | Benchmark Type & Conditions | Baseline | Candidate B | Candidate D | Leading Model |
|---|---|:---:|:---:|:---:|:---:|
| **1. COLD Official Test ($n=290$)** | Controlled in-domain split | $63.79\%$ Acc / $62.36\%$ F1 | **$73.45\%$ Acc / $72.31\%$ F1** | $68.62\%$ Acc / $67.90\%$ F1 | **Candidate B** (In-Domain) |
| **2. Track B External ($n=120$)** | External field (Ulfa et al.) | $55.00\%$ Acc | $52.50\%$ Acc | **$92.50\%$ Acc** | **Candidate D** (Field) |
| **3. Track C Real-World All ($n=97$)** | Wild multi-source field photos | $57.73\%$ Acc | $54.64\%$ Acc | **$81.44\%$ Acc** | **Candidate D** (Field) |
| **4. Track C High-Quality ($n=77$)** | Filtered field photos | $50.65\%$ Acc | $44.16\%$ Acc | **$76.62\%$ Acc** | **Candidate D** (Field) |
| **5. Experiment 5 PlantDoc ($n=115$)** | Independent field source (2 classes) | $43.48\%$ Acc / $46.02\%$ F1 | $40.00\%$ Acc / $49.75\%$ F1 | **$45.22\%$ Acc / $56.14\%$ F1** | **Candidate D** (Field) |

### Deployment Implications & Guardrails:
- **Candidate B:** Highest observed performance for controlled environments matching the COLD distribution ($73.45\%$ test accuracy), but shows degradation under field domain shifts.
- **Candidate D:** Shows consistent observed improvements across all three field-oriented evaluations (Track B, Track C, Experiment 5), providing evidence consistent with improved outdoor robustness.
- **Production Guardrail:** Production baseline (`models/chilli_cold/chilli_cold_baseline.keras`) remains the **active production model**. Neither Candidate B nor Candidate D is promoted at this stage. Candidate models remain research assets.
- **Prerequisite for Future Promotion:** Comprehensive field validation across all five canonical classes (including field-verified Murda Complex, Nutritional Deficiency, and Powdery Mildew) is required before considering production replacement.

---

## 11. Web Application Architecture & Integration Status

A complete Flask web application is operational locally and serves live predictions across the 4 production models.

### Integration Architecture:
- `run.py`: Application entry point (`http://127.0.0.1:5000`).
- `config.py`: Configuration parameters.
- `app/__init__.py`: Application factory, database setup, upload limits.
- `app/routes.py`: Web routes for upload, camera capture, prediction, and history management.
- `app/predictor.py`: Inference service layer connecting Flask routes to `.keras` models (routes Tomato, Grape, Sugarcane to baseline models; Chilli to Candidate D).
- `app/gradcam.py`: Application Grad-CAM integration module (runtime explainability via `tf.GradientTape`).
- `app/database.py`: SQLite operations (`instance/predictions.db`).

### Status of Web Application (Phase 18 Complete):
- Flask application model routing updated in Phase 18: Chilli routes to Candidate D (`experiments/chilli_field_experiment/candidate_d/model.keras`), while Tomato, Grape, and Sugarcane route to production baselines.
- Runtime Grad-CAM visual attention overlay is fully integrated and operational in the web UI.
- Verified and committed to `main` in commit `23f26ea`.

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
[Phase 8: Model Training (All 4 Crops)]   ✅ COMPLETED (Tomato, Grape, Chilli, Sugarcane baselines)
[Phase 9: Comparative Analysis]           ✅ COMPLETED (G1 vs G2, S1 vs S2, evaluate_comparative.py)
[Phase 10: In-Depth Error Analysis]       ✅ COMPLETED (Confusion patterns, error_analysis_report.txt)
[Phase 11-14: Web App Foundation]         ✅ COMPLETED (Flask, SQLite, UI, Predictor service)
[Phase 15: Grad-CAM Research Utility]     ✅ COMPLETED (gradcam_analysis.py, 10 diagnostic cases)
[Phase 16: 4-Track Robustness Audit]      ✅ COMPLETED (2,556 inferences across Tracks A, B, C, D)
[Phase 17A: Chilli Experiments 1–3]       ✅ COMPLETED (Candidates A, B, C; Cand B: 73.45% in-domain)
[Phase 17B: Chilli Experiment 4 (GPU)]    ✅ COMPLETED (Candidate D mixed-domain: 92.50% Track B, 81.44% Track C)
[Phase 17C: Chilli Experiment 5]          ✅ COMPLETED (Independent PlantDoc validation: Cand D 56.14% W-F1)
[Phase 18: Real-World Benchmark & Flask]  ✅ COMPLETED (Clean 4-crop benchmark n=1,172; Candidate D & Grad-CAM integrated; commit 23f26ea)
─────────────────────────────────────────────────────────────────────────────
[Phase 19: Cross-Crop Generalization & Gatekeeper Architecture] 🔄 ACTIVE
    ├── [Phase 19A: Flask Demonstration Verification]           ✅ COMPLETED (18/18 verification dimensions passed; 4 routes, Candidate D, Grad-CAM, WEBP, camera base64, deletion)
    └── [Phase 19B: Gatekeeper & OOD Safeguard Architecture]    🔜 NEXT (Substantive research: Gatekeepers, OOD, field gaps)
[Phase 20: Research Thesis & Dissertation]                      🔜 PENDING (Chapters 5, 6, 7, 8, 9)
```

---

## 13. Phase Hierarchy & Immediate Roadmap (Phase 18 -> 19 -> 20)

### Phase Hierarchy:
```
Phase 18 — Final Real-World Benchmark & Flask Candidate D Integration ✅
    ↓
Phase 19 — Cross-Crop Generalization & Two-Stage Gatekeeper Architecture
    ├── Phase 19A — Final Flask / Real-World Demonstration Verification ✅ (COMPLETED)
    └── Phase 19B — Cross-Crop Generalization, Two-Stage Gatekeeper, and OOD Safeguard Architecture 🔜 (NEXT)
    ↓
Phase 20 — Research Thesis & Project Dissertation
```

### Phase 18: Final Real-World Benchmark & Flask Candidate D Integration (COMPLETED ✅)
- **Status:** Completed and committed in `23f26ea`.
- **Outputs:**
  * Multi-crop clean real-world benchmark evaluated across 1,172 audited field images (Tomato $n=460$, Grape $n=280$, Sugarcane $n=100$, Chilli $n=332$).
  * Established Chilli head-to-head comparison on 332 identical images: Candidate D ($71.08\%$ accuracy, $71.53\%$ W-F1) demonstrated strong field generalization over Baseline ($50.60\%$) and Candidate B ($48.49\%$), particularly recovering Murda Complex recall ($62.00\%$ vs $2.00\%$).
  * Flask model routing updated: Chilli routes to Candidate D (`experiments/chilli_field_experiment/candidate_d/model.keras`), while Tomato, Grape, and Sugarcane route to production baselines.
  * Runtime Grad-CAM explainability generation integrated into the Flask application using non-symbolic `tf.GradientTape`.

---

### Phase 19: Cross-Crop Generalization & Two-Stage Gatekeeper Architecture (PARENT PHASE)
Phase 19 addresses the critical operational limitations exposed during the Phase 16 Robustness Audit and Phase 18 Real-World Benchmarks. To ensure operational stability, Phase 19 is explicitly split into two sequential sub-phases: **Phase 19A** (verification of existing software integration) and **Phase 19B** (substantive machine learning research).

#### Phase 19A: Final Flask / Real-World Demonstration Verification (COMPLETED ✅)
- **Status:** Completed. All 18 verification dimensions passed.
- **Role:** Verification sub-phase inserted prior to undertaking substantive Phase 19B research. *(Note: Phase 19A was a demonstration and software verification checkpoint, confirming the stability and routing of the deployed Flask application).*
- **Verification Results & Scope:**
  1. **Flask Startup & Endpoints:** Flask initializes cleanly; home page (`GET /`) returns HTTP 200; `/api/status` returns HTTP 200 (`healthy`) confirming all 4 crop models loaded.
  2. **Model Routing:**
     - Tomato: `models/tomato/tomato_baseline.keras` (10 classes)
     - Grape: `models/grape_unified/grape_unified_baseline.keras` (7 classes)
     - Sugarcane: `models/sugarcane_unified/sugarcane_unified_baseline.keras` (11 classes)
     - Chilli: `experiments/chilli_field_experiment/candidate_d/model.keras` (SHA-256: `d15704b95c3cc77e6f06a9a20af7d7e87b50f928089eccdd373097f08e96d437`, 5 classes)
     - Confirmed Chilli does **NOT** load `models/chilli_cold/chilli_cold_baseline.keras`.
  3. **Prediction Pipeline:** All 4 crops verified with valid leaf images; all return HTTP 200 with complete diagnosis cards, confidence metrics, and probability tables.
  4. **Runtime Grad-CAM:** Heatmap synthesis via `tf.GradientTape` verified on `mobilenetv2_1.00_224.out_relu`; physical `*_gradcam.jpg` overlay files generated and rendered in UI.
  5. **File Formats:** Standard image formats (JPEG, PNG) and WEBP upload verified end-to-end.
  6. **Camera Capture Workflow:** Verified via the server-side base64 `camera_image` data URL pipeline (POSTing base64 payload to `/predict`, decoding, and executing inference).  
     > [!NOTE]
     > **Camera Verification Scope:** Browser automation tools (Selenium / Playwright) were unavailable in the execution environment. The camera workflow was therefore verified through the server-side base64 `camera_image` pipeline, **NOT** as an automated browser camera test.
  7. **Error Handling:** Corrupt image files and non-image payloads return HTTP 200 with user-facing danger alerts; zero HTTP 500 crashes.
  8. **Deletion & Cleanup:** Deleting predictions via `POST /delete/<id>` removes the SQLite database record and deletes both the uploaded image and its associated Grad-CAM heatmap file from disk.
  9. **Data Integrity:** All model binaries, stratified split CSVs, and raw datasets verified 100% untouched.

#### Phase 19B: Cross-Crop Generalization, Two-Stage Gatekeeper, and OOD Safeguard Architecture (NEXT 🔜)
- **Role:** The core, substantive machine learning research phase that preserves and expands the original Phase 19 research objectives.
- **Scientific Context & Caution:** While mixed-domain training in Candidate D yielded significant observed field accuracy gains for Chilli ($+20.48\%$ over Baseline on evaluated subsets), this empirical difference does not prove causality or unrestricted open-world generalization. Substantive architectural defenses are required to handle arbitrary real-world inputs.
- **Key Research Components:**
  1. **Two-Stage Front-End Gatekeeper Architecture:**
     - *Stage 1 (Leaf vs. Non-Leaf Gatekeeper):* A lightweight binary classifier to detect and reject non-plant or arbitrary real-world objects (hands, soil, tools, sky, backgrounds) before any disease model is engaged.
     - *Stage 2 (Crop Species Identifier / Cross-Crop Validation):* Validates the crop species identity (Tomato, Grape, Chilli, Sugarcane) against user selection to eliminate the Track D failure mode where models produce $>90\%$ confident predictions on wrong-crop inputs.
  2. **Out-of-Distribution (OOD) & Low-Confidence Safeguard Architecture:**
     - Implement temperature scaling and prediction entropy thresholds to calibrate model confidence.
     - Introduce an explicit rejection fallback (e.g., softmax confidence $< 60\%$ or high distribution entropy triggers an *"Uncertain / Please Retake"* prompt rather than a false positive diagnosis).
  3. **Evaluation of Missing Real-World Field Classes:**
     - Source and audit true field-collected ground-truth imagery for unrepresented canonical classes identified in Phase 18:
       * Chilli Powdery Mildew ($n=0$ currently available in field benchmarks).
       * Grape Bacterial Leaf Spot ($n=0$ in external field datasets).
       * Sugarcane canonical classes ($9$ of $11$ classes currently unrepresented in field benchmarks).
  4. **Extending Mixed-Domain Training Research to Tomato & Grape:**
     - Adapt Candidate D's multi-source domain-mixing methodology to Tomato and Grape to address the severe laboratory-to-field domain shifts observed in Phase 18 ($18.91\%$ and $12.14\%$ field accuracy, respectively).

---

### Phase 20: Research Thesis & Project Dissertation (UPCOMING)
- Comprehensive thesis documentation across Chapters 5, 6, 7, 8, and 9 covering:
  * Chapter 5: Baseline Training & Single-Crop In-Domain Results.
  * Chapter 6: Comparative Analysis, Cross-Dataset Validation & Error Characterization.
  * Chapter 7: Explainable AI via Grad-CAM & Agronomic Plausibility Analysis.
  * Chapter 8: Multi-Track Robustness Audit, Field Benchmarking & Gatekeeper Evaluation.
  * Chapter 9: Discussion, Engineering Lessons, Limitations, and Future Agricultural Vision Systems.

---

## 14. Non-Negotiable Constraints & Architecture Rules

1. **Do Not Overwrite Production Baseline Models:**
   All 4 production models in `models/` remain read-only. Candidate models from experiments remain in `experiments/` and are not promoted without explicit evaluation and approval.
2. **Preserve Dataset and Split Integrity:**
   The 6 local datasets (31,590 images total) and established 70/15/15 stratified split CSVs are ground truth. Do not modify source images, re-split, or alter split CSVs.
3. **Strict Separation of Training and Evaluation Data:**
   Audit evaluation images from Tracks B and C must never be used for model training if retained as evaluation benchmarks.
4. **Objective Research Documentation:**
   Describe external and real-world results as *observed performance differences* and *evidence consistent with generalization limitations*. Do not claim that causality, shortcut learning, or domain shift has been definitively proven unless directly tested.
5. **No Unapproved Commits or Pushes:**
   Do not stage, commit, or push experimental code or large model binaries without explicit direction.

---

## 15. File & Repository Structure Reference

```
D:\CropDiseaseProject/
├── .gitignore
├── PROJECT_HANDOFF.md                 <-- Master single source of truth document (v1.8)
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
│   ├── train_kaggle_colab.ipynb       <-- Baseline training notebook
│   └── chilli_experiment4_kaggle.ipynb<-- Chilli Experiment 4 (Candidate D) Kaggle GPU notebook
│
├── app/                               <-- Flask Web Application (Verified in Phase 19A)
│   ├── __init__.py                    <-- App factory
│   ├── database.py                    <-- SQLite database operations
│   ├── gradcam.py                     <-- Application Grad-CAM module (operational)
│   ├── predictor.py                   <-- 4-crop inference service layer (Chilli -> Candidate D)
│   ├── routes.py                      <-- Web routes and endpoints
│   ├── static/                        <-- Styles, scripts, and uploaded assets
│   └── templates/                     <-- Jinja2 HTML templates
│
├── instance/                          <-- SQLite database storage (gitignored)
│   └── predictions.db
│
├── models/                            <-- 4 Trained Production .keras Models (LOCKED)
│   ├── tomato/tomato_baseline.keras
│   ├── grape_unified/grape_unified_baseline.keras
│   ├── chilli_cold/chilli_cold_baseline.keras  <-- Verified SHA-256 intact
│   └── sugarcane_unified/sugarcane_unified_baseline.keras
│
├── experiments/                       <-- Controlled Model Improvement Experiments
│   ├── chilli_improvement/            <-- Chilli C1 Experiments 1–3 (Single-Source COLD)
│   │   ├── experiment_plan.txt        <-- Parameter and design log
│   │   ├── baseline_reference.csv     <-- Baseline reference metrics
│   │   ├── comparison.csv             <-- Comparison table (Baseline, Cand A, B, C)
│   │   ├── improvement_report.txt     <-- Complete technical report
│   │   ├── candidate_a/               <-- Partial fine-tuning
│   │   ├── candidate_b/               <-- Fine-tuning + realistic aug (73.45% acc)
│   │   └── candidate_c/               <-- Fine-tuning + aug + class weighting
│   │
│   ├── chilli_field_experiment/       <-- Chilli Experiment 4 (Mixed-Domain Candidate D)
│   │   ├── field_data_manifest.csv    <-- Manifest of 2,152 training images (COLD + 800 Ulfa)
│   │   ├── chilli_experiment4_kaggle.ipynb
│   │   └── candidate_d/               <-- Candidate D deliverables
│   │       ├── model.keras            <-- Trained Candidate D checkpoint (21.8 MB)
│   │       ├── training_history.csv   <-- Epoch metrics log
│   │       ├── candidate_d_comparison.csv
│   │       ├── confusion_matrices.png
│   │       └── training_curves.png
│   │
│   └── chilli_independent_validation/<-- Chilli Experiment 5 (PlantDoc Validation)
│       ├── source_provenance.csv      <-- Full provenance records for 115 PlantDoc images
│       ├── source_manifest.csv        <-- Manifest with labels and SHA-256 hashes
│       ├── overlap_report.txt         <-- Zero-leakage audit report
│       ├── evaluation_metrics.csv     <-- Summary metrics (Baseline, Cand B, Cand D)
│       ├── class_metrics.csv          <-- Per-class metrics (with unrepresented disclosures)
│       ├── confusion_matrix.csv       <-- 5x5 contingency tables
│       ├── comparison.csv             <-- Comparison across models
│       ├── validation_report.txt      <-- Comprehensive validation report
│       ├── confusion_matrices.png     <-- Visualized confusion matrices
│       └── validation_images/         <-- 115 curated validation images
│
├── results/                           <-- Evaluation Result Suites & Analysis Outputs
│   ├── tomato/                        <-- Baseline metrics, report, confusion matrix, history
│   ├── grape_unified/                 <-- Baseline metrics, report, confusion matrix, history
│   ├── chilli_cold/                   <-- Baseline metrics, report, confusion matrix, history
│   ├── sugarcane_unified/             <-- Baseline metrics, report, confusion matrix, history
│   ├── comparative_analysis/          <-- Phase 9 comparative metrics, reports, confusion matrices
│   ├── error_analysis/                <-- Phase 10 error summary, class analyses, sample catalog
│   ├── gradcam/                       <-- Phase 15 diagnostic case directories & consolidated report
│   └── model_robustness_audit/        <-- Phase 16 4-Track Audit deliverables (13 reports + 5 charts)
│       ├── audit_predictions.csv      <-- All 2,556 empirical predictions
│       ├── audit_dataset_summary.csv  <-- Track & crop level summaries
│       ├── audit_per_class.csv        <-- Per-class metrics across tracks
│       ├── audit_confusions.csv       <-- Aggregated confusion pairs
│       ├── high_confidence_errors.csv <-- 632 high-confidence errors analyzed
│       ├── internal_external_comparison.csv
│       ├── internal_external_realworld_comparison.csv
│       ├── class_reliability.csv      <-- Green/Yellow/Red class scorecard
│       ├── wrong_crop_stress_test.csv <-- Track D cross-crop & unrelated results
│       ├── external_provenance.csv    <-- 632 Track B sample records (zero hash collisions)
│       ├── realworld_provenance.csv   <-- 425 Track C sample records (zero hash collisions)
│       ├── representative_audit_errors.csv
│       ├── audit_report.txt           <-- Master audit technical report
│       └── [5 visualization PNG charts]
│
├── splits/                            <-- Stratified Train/Val/Test CSVs (UNTOUCHED)
│   ├── tomato/
│   ├── grape_unified/
│   ├── chilli_cold/
│   ├── sugarcane_unified/
│   ├── grape_niphad/
│   ├── grape_2024/
│   ├── sugarcane_maharashtra/
│   └── sugarcane_large/
│
└── [Source Dataset Directories]       <-- 6 Verified Raw Datasets (~2.05 GB total, UNTOUCHED)
    ├── tomato_plantvillage/
    ├── grape_niphad/
    ├── grape_2024/
    ├── chilli_cold/
    ├── sugarcane_maharashtra/
    └── sugarcane_large/
```
