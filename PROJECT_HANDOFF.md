# Crop Disease Detection Project — Master Handoff Package

**Document Version:** 2.1  
**Project Workspace:** `D:\CropDiseaseProject`  
**Git Repository:** `https://github.com/sohail-148/maharashtra-crop-disease-detection` (Branch: `main`)  
**Status as of Handoff:** Completed Phases 1–8 (Model Training & Verification), Phase 9 (Comparative / Cross-Dataset Evaluation), Phase 10 (In-Depth Error Analysis), Phase 15 (Standalone Grad-CAM Explainability Analysis on 10 Prioritized Cases), Phase 16 (Comprehensive 4-Track Model Robustness Audit across 2,556 Inferences), the complete Chilli Research Cycle (Experiments 1–5), Phase 18, Phase 19A, Phase 19B-1, Phase 19B-2, Phase 19B-2A, Phase 19B-3 / 19B-3A, and Phase 19B-4:
- **Chilli Experiments 1–3:** Candidates A, B, and C evaluated on COLD; Candidate B achieved **73.45% accuracy**, **72.31% Weighted F1**, and **68.62% Macro F1** on the official test split as the leading in-domain research candidate.
- **Chilli Experiment 4:** Candidate D trained with mixed-domain data (COLD 2024 + Ulfa 2023 field data, total $n=2,152$) achieved **68.62% official test Accuracy**, **67.90% official test Weighted F1**, and **64.13% official test Macro F1**; on external/field benchmarks Candidate D achieved **92.50% on Track B External** (+40.00% over Candidate B), **81.44% on Track C Real-World** (+26.80% over Candidate B), and **76.62% on Track C High-Quality** (+32.46% over Candidate B).
- **Chilli Experiment 5:** Independent field-source validation using an unseen PlantDoc subset ($n=115$: 62 Cercospora, 53 Healthy; Murda Complex, Nutritional Deficiency, and Powdery Mildew strictly unrepresented). Candidate D achieved **45.22% Accuracy**, **56.14% Weighted F1**, and **56.26% represented-class Macro F1** (+10.12% Weighted F1 over Baseline, +6.39% over Candidate B).
- **Phase 18 (Completed in Commit `23f26ea`):** Clean real-world multi-crop benchmark ($n=1,172$) executed across Tomato, Grape, Sugarcane, and Chilli; Candidate D integrated into Flask for Chilli inference ($71.08\%$ real-world accuracy across $332$ images vs $50.60\%$ Baseline); runtime Grad-CAM visual attention integrated into the Flask web UI; production baselines for Tomato, Grape, and Sugarcane retained.
- **Phase 19A (Completed in Commit `bb3df48`):** End-to-end demonstration and software verification of the Flask application completed across all 18 dimensions (startup, home page, `/api/status`, all 4 model routes with Chilli -> Candidate D, 4-crop predictions, runtime Grad-CAM, WEBP upload, server-side base64 camera pipeline, invalid/corrupt input error handling, deletion + file cleanup, protected model/dataset integrity).
- **Phase 19B-1 (Completed in Commit `1660011`):** Offline confidence calibration (Temperature Scaling fit strictly on validation splits) and validation-selected rejection analysis completed across all 4 production baselines and Candidate D. Proved that temperature scaling improves in-domain calibration (reducing ECE and NLL), but does not separate correct vs incorrect real-world predictions; validation-selected confidence thresholds fail to adequately reject wrong-crop leaves (up to 73.3% false pass) or non-leaf objects (up to 60% false pass), proving that raw or calibrated softmax confidence alone is insufficient as an OOD safeguard.
- **Phase 19B-2 (Completed):** Dedicated Stage 2 Crop Species Identifier (MobileNetV2 4-class: Tomato, Grape, Chilli, Sugarcane) trained strictly on `splits/*/train.csv` ($n=22,110$) without modifying production disease models. Evaluated across validation ($n=4,739$), internal test ($n=4,741$), Track D ($n=100$), Track B ($n=632$), Track C ($n=425$), and Phase 18 Benchmark ($n=1,172$). Achieved **99.87% internal test accuracy**, **100.00% Track D cross-crop accuracy**, and detected **100.00% (300/300)** of simulated user-crop mismatches.
- **Phase 19B-2A (Completed):** Crop Identifier Field Error Analysis across all 1,172 Phase 18 benchmark images and 4,741 internal test images:
  1. Crop identifier internal test accuracy: **99.87%**.
  2. Track D cross-crop accuracy: **100.00%**.
  3. Simulated user-crop mismatch detection: **300/300 = 100.00%**.
  4. Track B field/external performance: **approximately 36%** (36.23% Model A / 35.60% Model B).
  5. Track C real-world performance: **approximately 40%** (40.71% Model A / 39.29% Model B).
  6. Phase 18 field benchmark: **approximately 39%** (39.68% Model A / 39.16% Model B).
  7. The crop identifier is therefore demonstrated effective for the evaluated controlled cross-crop mismatch task, but is **NOT validated as a mandatory unrestricted field gatekeeper**.
  8. Tomato field images were frequently predicted as Chilli/Grape (recall collapsed to 12.2%–13.5% in field data), and Grape field images frequently as Tomato (57.5%–63.9% confusion with Tomato); Sugarcane achieved **100.00% recall with 0 errors** across all field datasets.
  9. Model A vs Model B augmentation produced only marginal field changes (39.68% vs 39.16%, with 57.0% error agreement).
  10. The observed field errors are consistent with sensitivity to domain-specific visual characteristics; we do NOT state that background shortcut learning or any other single mechanism is proven causal.
  11. New independent field training data would be required to improve field crop-identification generalization without contaminating the existing held-out field benchmarks.
- **Phase 19B-3 / 19B-3A (Completed):** Non-Leaf Negative Dataset Sourcing & Curation completed with comprehensive legal/licensing audit across 10 candidate sources. Approved with restrictions: Google Open Images V7 (SRC-02, CC BY 2.0 with mandatory per-image attribution metadata), Wikimedia Commons (SRC-03, strictly image-specific CC0/CC BY/CC BY-SA with mandatory 7-field provenance registry), Kaggle Soil Types (SRC-04, CC0/CC BY-SA), and Kaggle Fruits-360 (SRC-05, CC BY-SA 4.0, training-only non-leaf fruits). Rejected: MS COCO 2017 (SRC-01, unverified Flickr terms), Kaggle Flowers (SRC-06, Unknown license / web scrape), PlantVillage Fruit (SRC-07, high contamination risk with internal training/test sets), Track D Unrelated (SRC-08, active held-out benchmark asset), ImageNet (SRC-09, redistribution prohibited), and Agriculture-Vision (SRC-10, aerial perspective mismatch). Designed a quality-driven 1:1 balanced binary dataset ($n=3,000$ to $6,000$: positive leaves stratified across internal training splits, matched 1:1 to curated negatives across 5 subcategories), documented operational leaf-area curation rules (>=25% positive, <5% negative), noted dataset-source domain shortcut limitations, and established a multi-stage contamination screening protocol.
- **Phase 19B-4 (Completed):** Stage 1 Leaf vs. Non-Leaf Binary Gatekeeper trained and evaluated on 1,606 curated images (803 Leaf positives sampled strictly from internal training splits, 803 Non-Leaf negatives from conditionally approved sources: Open Images V7, Fruits-360, Kaggle Soil Types, Wikimedia Commons; 70/15/15 stratified split; 0 hash duplicates or collisions against 9,921 protected benchmark images):
  1. **Architecture:** MobileNetV2 (frozen ImageNet backbone) + Batch Normalization + Dropout (0.3) + Dense(128, ReLU) + Dropout (0.2) + Dense(1, Sigmoid).
  2. **Internal Test Performance ($\theta=0.50$):** **99.19% Accuracy**, **100.00% Leaf Retention** (123/123), **98.40% Non-Leaf Rejection** (123/125), Precision 98.40%, F1 99.19%, **ROC-AUC 0.9974**.
  3. **Source-Wise Rejection:** Google Open Images V7 100.00% (56/56), Fruits-360 100.00% (29/29), Kaggle Soil Types 96.15% (25/26), Wikimedia Commons 92.86% (13/14).
  4. **Subcategory-Wise Rejection:** Tools & Equipment 100.00% (13/13), Hands & Apparel 100.00% (32/32), Non-Leaf Plant Parts 100.00% (35/35), Soil & Ground 96.43% (27/28), Buildings & Structures 94.12% (16/17).
  5. **Crop-Wise Leaf Retention:** Tomato 100.00% (31/31), Grape 100.00% (31/31), Chilli 100.00% (31/31), Sugarcane 100.00% (30/30).
  6. **Track D Stress Evaluation:** **19/19 actual non-leaf distractors rejected = 100.00%**. (`aloeL.jpg` was excluded from the non-leaf denominator because it is itself an Aloe Vera plant leaf and was correctly classified as Leaf with $p=0.9986$).
  7. **Held-Out Real-World Field Leaf Retention:** Evaluated across 2,229 uncurated field images: **2,150 / 2,229 = 96.55% retained** (Track B External: 95.41% [603/632], Track C Real-World: 97.65% [415/425], Phase 18 Benchmark: 96.59% [1,132/1,172]).
  8. **Threshold Sweep & Operational Trade-Offs:**
     * $\theta = 0.20$: Leaf-retention-oriented operating point (97.73% field leaf retention, 97.60% internal non-leaf rejection).
     * $\theta = 0.50$: Default balanced operating point (96.55% field leaf retention, 98.40% internal non-leaf rejection, 99.19% test accuracy).
- **Phase 19B-5 (Completed):** Field Data Gap Audit, Sourcing Strategy, and Finite Clean Acquisition completed for Tomato and Grape:
  1. **Tomato Field Audit ($n=460$ across Phase 18):** Overall field accuracy was only **18.91%** ($87/460$). Attractor sink collapse was severe: **87.6% (403/460)** of all field predictions collapsed into either Late Blight (217) or Early Blight (186). Healthy field leaves suffered a **100% false-positive disease rate** (0/58 correct).
  2. **Grape Field Audit ($n=280$ across Phase 18):** Overall field accuracy was only **11.43%** ($32/280$). Attractor collapse: **74.3%** collapsed into Healthy Leaves or Powdery Mildew.
  3. **FieldPlant Tomato Taxonomy Correction:** Review of Moupojou et al. (IEEE Access 2023) confirmed FieldPlant legitimately supplies 3 canonical classes: Healthy, Tomato Mosaic Virus, and Tomato Yellow Leaf Curl Virus. Invalid mappings (Bacterial Wilt != Bacterial Spot, Brown Spots != Target Spot) were strictly rejected.
  4. **GVLiD Internal Identity Verification:** Confirmed that internal G2 dataset (`grape_2024`, $n=3,477$) is 100% identical to GVLiD (Shikalgar et al., 2024); GVLiD was marked INTERNAL / ALREADY USED.
  5. **Authentic Field Acquisition:** Acquired, screened, and verified **247 authentic in-situ field images** with 0 benchmark collisions against 31,805 unique SHA-256 hashes:
     * Tomato (FieldPlant, Cameroon): Healthy (13), Mosaic Virus (13), Yellow Leaf Curl Virus (45) = 71 images.
     * Grape (HERMOS, Turkey): Powdery Mildew (45), Downy Mildew (44), Healthy Leaves (43) = 132 images.
     * Grape (Standalone ESCA, Italy): Esca (44) = 44 images.
     * Grape Total = 176 images.
  6. **Benchmark Record Distinction:** Clarified that master registry entries (38,376 records) represent sample prediction records across multi-candidate evaluations; physical held-out field benchmark count is exactly 1,172 images (Tomato: 460, Grape: 280, Chilli: 332, Sugarcane: 100).
- **Phase 19B-6 (Completed):** Field-Data-Assisted Retraining Experiment for Tomato and Grape executed on Kaggle GPU (Tesla T4) and evaluated across all benchmark tracks:
  1. **Augmented Training Datasets:**
     * Tomato: 10,170 base + 71 authentic FieldPlant field images ($n=10,241$).
     * Grape: 4,341 base + 176 authentic field images ($n=4,517$).
     * Strict baseline architecture preserved: Frozen MobileNetV2 (ImageNet) + GAP + BN + Dropout(0.3) + Dense(N, Softmax), Adam(1e-3), seed 42.
  2. **Model Artifact Verification:**
     * Tomato adapted model: `experiments/field_data_19b5/models/tomato_field_adapted.keras` (9,821,400 bytes, SHA-256: `211cdb63aa6aef2f56601c6cb1cdfa784c344b806bf97550d1f8ce25b94f27ff`).
     * Grape adapted model: `experiments/field_data_19b5/models/grape_field_adapted.keras` (9,775,317 bytes, SHA-256: `f3e2367e32c9b15951c4b42522540dff2e9869f6aee6a8613e4e4b7882b0b867`).
     * Both models are confirmed present and verified as the exact artifacts used for final comparative evaluation.
  3. **Resolution of Tomato Intermediate-Report Discrepancy (90.14% vs. 89.72%):**
     * During Kaggle execution, Version 2 completed Tomato training and reported 90.14% test accuracy, but errored out at the cell transition to Grape due to a newline string syntax error before Grape began.
     * Version 3 fixed the syntax error and trained both Tomato and Grape to complete termination.
     * In this clean, completed Version 3 run, Tomato converged with test loss 0.3002 and test accuracy **89.72%** (recorded in `training_summary.txt`).
     * The model artifact downloaded from the completed Version 3 run (`tomato_field_adapted.keras`) is the authoritative artifact evaluated locally on `splits/tomato/test.csv` ($n=2,180$), yielding exactly **89.72% Accuracy** and **87.16% Macro F1**.
     * The 90.14% figure was an intermediate metric from the incomplete Version 2 run; 89.72% is the authoritative final comparative metric.
  4. **Multi-Track Comparative Benchmark Results:**
     * **Tomato Internal Test ($n=2,180$):** Baseline 90.23%, Adapted 89.72% ($\Delta -0.50$ percentage points); Baseline Macro F1 87.80%, Adapted Macro F1 87.16% ($\Delta -0.64$ percentage points).
     * **Tomato Track B External ($n=272$):** Baseline 15.07%, Adapted 15.07% ($\Delta +0.00$ percentage points); Baseline Macro F1 7.78%, Adapted Macro F1 12.08% ($\Delta +4.29$ percentage points).
     * **Tomato Track C Real-World ($n=188$):** Baseline 24.47%, Adapted 27.66% ($\Delta +3.19$ percentage points); Baseline Macro F1 12.99%, Adapted Macro F1 17.72% ($\Delta +4.73$ percentage points).
     * **Tomato Phase 18 Combined Field ($n=460$):** Baseline 18.91%, Adapted 20.22% ($\Delta +1.30$ percentage points); Baseline Macro F1 9.43%, Adapted Macro F1 14.68% ($\Delta +5.25$ percentage points).
     * **Grape Internal Test ($n=931$):** Baseline 89.04%, Adapted 87.86% ($\Delta -1.18$ percentage points); Baseline Macro F1 86.31%, Adapted Macro F1 83.28% ($\Delta -3.03$ percentage points).
     * **Grape Track B External ($n=180$):** Baseline 13.89%, Adapted 15.56% ($\Delta +1.67$ percentage points); Baseline Macro F1 7.98%, Adapted Macro F1 10.32% ($\Delta +2.34$ percentage points).
     * **Grape Track C Real-World ($n=100$):** Baseline 7.00%, Adapted 9.00% ($\Delta +2.00$ percentage points); Baseline Macro F1 5.26%, Adapted Macro F1 6.70% ($\Delta +1.44$ percentage points).
     * **Grape Phase 18 Combined Field ($n=280$):** Baseline 11.43%, Adapted 13.21% ($\Delta +1.79$ percentage points); Baseline Macro F1 7.34%, Adapted Macro F1 9.33% ($\Delta +2.00$ percentage points).
  5. **Target Augmented Classes Analysis (Phase 18 Field Benchmark):**
     * `Tomato Yellow Leaf Curl Virus`: Recall 5.36% -> 42.86% ($\Delta +37.50$ percentage points), F1 8.33% -> 24.24%.
     * `Tomato Mosaic Virus`: Recall 0.00% -> 7.50% ($\Delta +7.50$ percentage points), F1 0.00% -> 7.79%.
     * `Tomato Healthy`: Recall 0.00% -> 27.59% ($\Delta +27.59$ percentage points), F1 0.00% -> 25.00%.
     * `Grape Downy Mildew`: Recall 4.00% -> 26.00% ($\Delta +22.00$ percentage points), F1 6.67% -> 18.18%.
     * `Grape Powdery Mildew`: Recall 0.00% -> 3.33% ($\Delta +3.33$ percentage points), F1 0.00% -> 2.25%.
     * `Grape Healthy Leaves`: Recall 59.52% -> 52.38% ($\Delta -7.14$ percentage points), F1 30.12% -> 42.31% ($\Delta +12.19$ percentage points).
     * `Grape Esca`: Recall 2.00% -> 0.00% ($\Delta -2.00$ percentage points), F1 2.86% -> 0.00%.
  6. **Scientific Interpretation:**
     * *Core finding:* Adding the selected independently collected field images was associated with modest overall field-benchmark gains and larger class-specific changes, while internal test performance decreased modestly.
     * The field-data intervention produced modest overall improvements on the evaluated external/field benchmarks.
     * Several targeted classes showed substantially larger recall changes.
     * Internal performance declined modestly for both adapted models (-0.50 pp for Tomato, -1.18 pp for Grape).
     * The intervention did NOT eliminate the large real-world generalization gap (field accuracy remains ~13-20% vs ~88-90% internal test accuracy).
     * The results demonstrate an observed effect of adding the selected independent field data under this experimental setup.
     * We do NOT claim universal causality, do NOT claim that field adaptation solved domain shift, do NOT claim universal field generalization, and do NOT justify replacing the production models on the basis of these results alone.
  7. **Experimental Limitations:**
     * Tomato received only 71 additional field images.
     * Tomato additions covered only Healthy, Mosaic Virus, and Yellow Leaf Curl Virus.
     * Grape received 176 additional field images.
     * Several canonical classes remained unaugmented (e.g., Tomato Spider Mites, Target Spot, Leaf Mold; Grape Bacterial Leaf Spot, Black Rot).
     * Added field datasets originated from different geographic and acquisition domains (Cameroon, Turkey, Italy).
     * The experiment tests this specific field-data intervention and does not establish universal improvement.
     * External field benchmarks remain finite samples and may not represent all future field conditions.
     * The experiment does not establish that field-data augmentation will improve every disease class.
     * Small class-level changes should be interpreted together with the corresponding sample counts.
  8. **Preservation of Production vs. Experimental Separation:**
     * Production models in `models/` remain 100% UNCHANGED, LOCKED, and READ-ONLY.
     * Web application (`app/`) remains 100% UNCHANGED.
     * Dataset splits (`splits/`) and evaluation benchmarks (Track B, Track C, Phase 18) remain 100% UNCHANGED.
     * Experimental models remain strictly under `experiments/field_data_19b5/models/` as research artifacts.

Active research phase: Phase 19B-6 Completed. Next phase is Phase 19B-7 (Final Error Analysis & Academic Research Synthesis).

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
    ├── [Phase 19A: Flask Demonstration Verification]               ✅ COMPLETED (18/18 verification dimensions passed; commit bb3df48)
    ├── [Phase 19B-1: Confidence Calibration & Rejection Analysis]  ✅ COMPLETED (T fit on val; ECE/Brier/NLL evaluated; confidence alone insufficient for OOD)
    ├── [Phase 19B-2: Stage 2 Crop Species Identifier]              ✅ COMPLETED (99.87% internal, 100% Track D mismatch detection)
    ├── [Phase 19B-2A: Crop Identifier Field Error Analysis]        ✅ COMPLETED (~39% field accuracy analyzed; domain-specific errors)
    ├── [Phase 19B-3 / 19B-3A: Non-Leaf Negative Sourcing & Audit]  ✅ COMPLETED (Strict licensing audit, 4 sources conditionally approved)
    ├── [Phase 19B-4: Stage 1 Binary Leaf/Non-Leaf Gatekeeper]      ✅ COMPLETED (99.19% test acc, 100% leaf recall, 98.4% non-leaf spec, 96.55% field retention)
    └── [Phase 19B-5: Missing Field Classes / Tomato-Grape Field]   🔄 ACTIVE (Field audit & sourcing strategy completed; acquisition pending approval)
[Phase 20: Research Thesis & Dissertation]                          🔜 PENDING (Chapters 5, 6, 7, 8, 9)
```

---

## 13. Phase Hierarchy & Immediate Roadmap (Phase 18 -> 19 -> 20)

### Phase Hierarchy:
```
Phase 18 — Final Real-World Benchmark & Flask Candidate D Integration ✅
    ↓
Phase 19 — Cross-Crop Generalization & Two-Stage Gatekeeper Architecture
    ├── Phase 19A — Final Flask / Real-World Demonstration Verification ✅ (COMPLETED)
    └── Phase 19B — Cross-Crop Generalization, Two-Stage Gatekeeper, and OOD Safeguard Architecture
        ├── Phase 19B-1 — Offline Calibration & Rejection Curve Analysis ✅ (COMPLETED)
        ├── Phase 19B-2 — Dedicated Stage 2 Crop Species Identifier ✅ (COMPLETED)
        ├── Phase 19B-2A — Crop Identifier Field Error Analysis ✅ (COMPLETED)
        ├── Phase 19B-3 / 19B-3A — Non-Leaf Negative Sourcing & Audit ✅ (COMPLETED)
        ├── Phase 19B-4 — Stage 1 Leaf vs. Non-Leaf Binary Gatekeeper ✅ (COMPLETED)
        └── Phase 19B-5 — Missing Field Classes / Tomato-Grape Field Data 🔄 (ACTIVE)
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
Phase 19 addresses the critical operational limitations exposed during the Phase 16 Robustness Audit and Phase 18 Real-World Benchmarks.

#### Phase 19A: Final Flask / Real-World Demonstration Verification (COMPLETED ✅)
- **Status:** Completed and committed in `bb3df48`. All 18 verification dimensions passed.
- **Verification Results & Scope:**
  1. **Flask Startup & Endpoints:** Flask initializes cleanly; home page (`GET /`) returns HTTP 200; `/api/status` returns HTTP 200 (`healthy`) confirming all 4 crop models loaded.
  2. **Model Routing:** Tomato Baseline (10 classes), Grape Unified Baseline (7 classes), Sugarcane Unified Baseline (11 classes), Chilli Candidate D (5 classes; SHA-256: `d15704b95c3cc77e6f06a9a20af7d7e87b50f928089eccdd373097f08e96d437`). Confirmed Chilli does **NOT** load baseline.
  3. **Prediction Pipeline:** All 4 crops verified with valid leaf images; all return HTTP 200 with complete diagnosis cards, confidence metrics, and probability tables.
  4. **Runtime Grad-CAM:** Heatmap synthesis via `tf.GradientTape` verified on `mobilenetv2_1.00_224.out_relu`; physical `*_gradcam.jpg` overlay files generated and rendered in UI.
  5. **File Formats:** Standard image formats (JPEG, PNG) and WEBP upload verified end-to-end.
  6. **Camera Capture Workflow:** Verified via server-side base64 `camera_image` data URL pipeline (POSTing base64 payload to `/predict`, decoding, and executing inference).  
     > [!NOTE]
     > **Camera Verification Scope:** Browser automation tools (Selenium / Playwright) were unavailable in the execution environment. The camera workflow was therefore verified through the server-side base64 `camera_image` pipeline, **NOT** as an automated browser camera test.
  7. **Error Handling:** Corrupt image files and non-image payloads return HTTP 200 with user-facing danger alerts; zero HTTP 500 crashes.
  8. **Deletion & Cleanup:** Deleting predictions via `POST /delete/<id>` removes SQLite record, uploaded image, and Grad-CAM heatmap file from disk.
  9. **Data Integrity:** All model binaries, stratified split CSVs, and raw datasets verified 100% untouched.

#### Phase 19B: Cross-Crop Generalization, Two-Stage Gatekeeper, and OOD Safeguard Architecture

##### Phase 19B-1: Offline Confidence Calibration & Rejection Analysis (COMPLETED ✅)
- **Status:** Completed. All 9 analysis deliverables generated under `results/phase19b_1_calibration/`.
- **Methodology:** Post-hoc Temperature Scaling ($T$) fit **strictly on validation splits** (`splits/*/val.csv`) by minimizing NLL via scalar optimization. Internal held-out test splits, Track B (external), Track C (real-world), and Track D (cross-crop & unrelated) were quarantined strictly as held-out evaluation sets.
- **Optimal Temperatures ($T$):**
  * Tomato Baseline: $T = 0.932$ (in-domain laboratory data is already sharply calibrated).
  * Grape Unified Baseline: $T = 1.148$ (softens moderate in-domain overconfidence).
  * Chilli Baseline: $T = 1.263$ (substantially softens in-domain overconfidence).
  * Sugarcane Unified Baseline: $T = 0.961$ (well-calibrated in-domain).
  * Chilli Candidate D (Supplementary): $T = 1.498$ (softens sharp COLD validation distribution).
- **Calibration Findings (ECE, Brier, NLL):**
  * *In-Domain:* Temperature scaling consistently improved in-domain probability reliability across all models, halving ECE on held-out internal test splits (Tomato test ECE: $0.0115 \to 0.0052$; Grape test ECE: $0.0329 \to 0.0196$; Chilli test ECE: $0.0801 \to 0.0676$).
  * *Out-of-Domain:* On external (Track B) and real-world (Track C) benchmarks, calibration error remained high (Tomato Track B ECE: $0.6900$, Track C ECE: $0.6061$; Grape Track B ECE: $0.6179$, Track C ECE: $0.6552$; Sugarcane Track B ECE: $0.3311$, Track C ECE: $0.4610$). Because temperature scaling is strictly monotonic, it preserves logit rankings and cannot resolve domain shift or separate overlapping error distributions.
- **Rejection & Threshold Findings:**
  * Validation-selected thresholds ($\theta_{\text{acc} \ge 90\%}$ and $\theta_{\text{acc} \ge 95\%}$) allowed **44.0% to 98.7% of wrong-crop leaves (Track D Cross-Crop)** and **up to 60% of non-leaf objects (Track D Unrelated)** to bypass rejection and receive false positive diagnoses.
  * Raising thresholds to aggressively reject OOD inputs severely degraded real-world operational coverage, discarding **40% to 97% of legitimate real-world field predictions**.
- **Core Conclusion & Next Step:**
  * Raw or calibrated softmax confidence alone is **fundamentally insufficient** as the sole OOD/wrong-crop safeguard.
  * The empirical findings directly justify proceeding to **Phase 19B-2: Dedicated Stage 2 Crop Species Identifier**.
  * *Scientific Guardrail:* Findings reflect observed empirical differences; we do not claim calibration proves causality or solves OOD, nor that gatekeepers are the only possible solution.

##### Phase 19B-2: Dedicated Stage 2 Crop Species Identifier (COMPLETED ✅)
- **Role:** Build and evaluate a separate lightweight MobileNetV2 4-class crop species identifier (Tomato, Grape, Chilli, Sugarcane) as a front-end validation gatekeeper.
- **Data Strategy:** Trained strictly on `splits/*/train.csv` ($n=22,110$: Tomato $10,170$, Grape $4,341$, Chilli $1,352$, Sugarcane $6,247$). Validation on `splits/*/val.csv` ($n=4,739$). All test splits, Track D, Track B, Track C, and Phase 18 Benchmark preserved strictly for evaluation.
- **Models Trained:** Model A (Baseline, standard preprocessing) and Model B (Augmented with geometric and photometric perturbations).
- **Core Results:**
  * Internal Test Accuracy: **99.87%** (Macro F1: $0.9978$).
  * Track D Cross-Crop Accuracy: **100.00%** (Macro F1: $1.0000$).
  * Simulated User-Crop Mismatch Detection: **300 / 300 (100.00%)** of simulated user-crop mismatches detected and prevented.
  * Track B External Accuracy: **36.23%** (Model A) / **35.60%** (Model B).
  * Track C Real-World Accuracy: **40.71%** (Model A) / **39.29%** (Model B).
  * Phase 18 Benchmark Accuracy: **39.68%** (Model A) / **39.16%** (Model B).
- **Artifacts:** Deliverables preserved in `results/phase19b_2_crop_identifier/` and `experiments/phase19b_2_crop_identifier/`.

##### Phase 19B-2A: Crop Identifier Field Error Analysis (COMPLETED ✅)
- **Role:** Comprehensive empirical error analysis across all 1,172 Phase 18 benchmark images and 4,741 internal test images to analyze the domain shift between controlled and field evaluations.
- **Key Findings:**
  1. Internal test accuracy: **99.87%**.
  2. Track D cross-crop accuracy: **100.00%**.
  3. Simulated user-crop mismatch detection: **300/300 = 100.00%**.
  4. Track B field/external performance: **approximately 36%** (36.23% Model A / 35.60% Model B).
  5. Track C real-world performance: **approximately 40%** (40.71% Model A / 39.29% Model B).
  6. Phase 18 field benchmark: **approximately 39%** (39.68% Model A / 39.16% Model B).
  7. The crop identifier is therefore demonstrated effective for the evaluated controlled cross-crop mismatch task, but is **NOT validated as a mandatory unrestricted field gatekeeper**.
  8. Tomato field images were frequently predicted as Chilli/Grape (recall collapsed to 12.2%–13.5% in field data; 0.0% on three independent field sources), and Grape field images frequently as Tomato (57.5%–63.9% confusion with Tomato); Sugarcane achieved **100.00% recall with 0 errors** across all field datasets.
  9. Model A vs Model B augmentation produced only marginal field changes (39.68% vs 39.16%, with 57.0% identical error agreement).
  10. The observed field errors are consistent with sensitivity to domain-specific visual characteristics; we do NOT state that background shortcut learning or any other single mechanism is proven causal.
  11. New independent field training data would be required to improve field crop-identification generalization without contaminating the existing held-out field benchmarks.
- **Artifacts:** Deliverables preserved in `results/phase19b_2a_error_analysis/` (`field_confusion_matrix.csv`, `source_confusion_analysis.csv`, `confidence_analysis.csv`, `model_a_vs_b_errors.csv`, `representative_errors.csv`, `phase19b_2a_report.txt`).

##### Phase 19B-3 / 19B-3A: Non-Leaf Negative Dataset Sourcing & Curation (COMPLETED ✅)
- **Role:** Comprehensive data sourcing, legal/licensing audit, and dataset design for the Stage 1 binary Leaf vs. Non-Leaf Gatekeeper.
- **Audit Findings across 10 Candidate Sources (Phase 19B-3A Reassessment):**
  * *Approved with Restrictions:*
    1. Google Open Images V7 (SRC-02): Approved for farm machinery, tools, boots, and gloves. Licensing: Annotations CC BY 4.0; images listed as CC BY 2.0. Mandatory requirement: Record original Flickr URL, author/creator credit, and CC BY 2.0 deed link in a provenance registry.
    2. Wikimedia Commons (SRC-03): Approved for authentic rural Indian agricultural contexts (irrigation pipes, stakes, empty furrows, soil, farmers working). Licensing is strictly image-specific. Mandatory requirement: Log 7-field provenance metadata (source URL, author, specific license, license URL, date, SHA-256, subcategory); licenses restricted to CC0, Public Domain, CC BY, and CC BY-SA.
    3. Kaggle Soil Types (SRC-04): Approved for bare agricultural soil textures. Licensing: CC0 (Satpathy) / CC BY-SA 4.0 (Salader). Prefer CC0 where possible; log attribution for Salader.
    4. Kaggle Fruits-360 (SRC-05): Approved strictly for training-only non-leaf fruit negatives (harvested tomato, grape, pepper fruits without leaves) to teach the model that fruit != leaf. Official license: CC BY-SA 4.0 (ShareAlike obligations noted for any derived dataset distribution).
  * *Rejected Sources (Not Approved):*
    1. MS COCO 2017 (SRC-01): REJECTED. Annotations are CC BY 4.0, but underlying images are governed by individual Flickr terms without verified open licenses; per-image verification across thousands of images is impractical and legally uncertain.
    2. Kaggle Flowers Recognition (SRC-06): REJECTED. License is listed as Unknown on Kaggle; scraped from Flickr, Google, and Yandex without creator attribution or verified open licenses.
    3. PlantVillage Fruit Subsets (SRC-07): REJECTED due to critical contamination risk with internal Tomato training/test splits.
    4. Track D Unrelated Dataset (SRC-08): REJECTED FOR TRAINING because it is an active, published held-out benchmark asset from Phase 16.
    5. ImageNet ILSVRC (SRC-09): REJECTED due to strict redistribution prohibitions in terms of use.
    6. Agriculture-Vision Dataset (SRC-10): REJECTED due to aerial perspective mismatch and restricted academic license.
- **Dataset Design Specifications:**
  * Formulation: Generic Leaf vs. Non-Leaf (Formulation A recommended over 4-crop-leaf or 5-class joint formulations to decouple leaf detection from crop species recognition).
  * Flexible Quality-Driven Sizing: Target range $n=3,000$ to $6,000$ images, strictly 1:1 balanced. Dataset size is quality-driven and provenance-constrained rather than artificially forced to 6,000. Positives will be subsampled 1:1 to match whatever number of pristine negatives pass screening.
  * Operational Area Rules: Positive leaf $\ge 25\%$ frame area; Negative non-leaf $< 5\%$ frame area; Ambiguous cases ($5\% \le \text{area} < 25\%$) strictly excluded. Documented as operational curation rules rather than universal definitions.
  * Positive Sourcing: Stratified subsample from internal training splits (`splits/*/train.csv`: Tomato 25%, Grape 25%, Chilli 25%, Sugarcane 25%). Validation, test, and all external benchmark images strictly excluded.
  * Scientific Limitation: Acknowledged risk of dataset-source domain shortcut when pairing in-domain leaves with public negatives; Phase 19B-4 must evaluate source-specific performance and not assume internal accuracy implies robust field rejection.
  * Contamination Controls: Multi-stage contamination screening (SHA-256 collision check against all 31,590 repository images and benchmarks, perceptual dHash/pHash Hamming $\le 5$ check, cross-source deduplication, and manual visual boundary inspection) intended to minimize detectable leakage risk.
- **Artifacts:** Deliverables preserved in `results/phase19b_3_nonleaf_sourcing/` (`source_audit.csv`, `source_audit.txt`, `dataset_design.txt`).

##### Phase 19B-4: Stage 1 Leaf vs. Non-Leaf Binary Gatekeeper Training & Evaluation (COMPLETED ✅)
- **Role & Architectural Implementation:**
  * Trained a lightweight binary MobileNetV2 Leaf vs. Non-Leaf Gatekeeper to serve as a pre-filtering stage before crop identification and disease diagnosis.
  * Architecture: Pretrained ImageNet MobileNetV2 backbone (frozen, 1,280-d GAP) + Batch Normalization + Dropout (0.30) + Dense (128, ReLU) + Dropout (0.20) + Dense (1, Sigmoid).
  * Checkpoint & Weights: Preserved in `results/phase19b_4_gatekeeper/gatekeeper_head.keras` and `gatekeeper_mobilenetv2.weights.h5`.
- **Dataset Construction & Provenance:**
  * Total Size: 1,606 verified images, strictly 1:1 balanced (803 Positive Leaves, 803 Negative Non-Leaves).
  * Positive Pool (803 leaves): Sampled exclusively from internal training splits (`splits/*/train.csv`: 201 Tomato, 201 Grape, 201 Chilli, 200 Sugarcane).
  * Negative Pool (803 non-leaves): Sourced exclusively from conditionally approved Phase 19B-3A sources: Google Open Images V7 (383 images, 47.7%), Fruits-360 non-leaf isolated fruits (180 images, 22.4%), Kaggle Soil Types (179 images, 22.3%), and Wikimedia Commons 640px thumbnails (61 images, 7.6%).
  * Negative Subcategories: Non-leaf plant parts/fruits (230, 28.6%), human hands & apparel (205, 25.5%), soil & ground (181, 22.5%), buildings & structures (106, 13.2%), agricultural tools & equipment (81, 10.1%).
  * Stratified Split: 70% Train (1,120 images: 560 Leaf, 560 Non-Leaf) / 15% Validation (238 images: 120 Leaf, 118 Non-Leaf) / 15% Test (248 images: 123 Leaf, 125 Non-Leaf).
  * Contamination & Integrity: 1,606 strictly unique SHA-256 hashes (0 cross-split duplicates, 0 collisions against 9,921 protected benchmark and test files across Track B, Track C, Track D, Phase 18, Exp 5, and Candidate D). 100% complete 7-field provenance metadata logged in `provenance_registry.csv`.
- **Internal Evaluation Performance ($\theta = 0.50$):**
  * Validation Split ($n=238$): 100.00% Accuracy, 100.00% Leaf Recall, 100.00% Specificity, ROC-AUC 1.0000.
  * Held-Out Test Split ($n=248$): **99.19% Accuracy**, **100.00% Leaf Retention** (123/123), **98.40% Non-Leaf Rejection** (123/125), Precision 98.40%, F1 99.19%, **ROC-AUC 0.9974**.
  * Source-Wise Test Rejection: Open Images V7 100.0% (56/56), Fruits-360 100.0% (29/29), Kaggle Soil Types 96.15% (25/26), Wikimedia Commons 92.86% (13/14).
  * Subcategory-Wise Test Rejection: Tools & Equipment 100.0% (13/13), Hands & Apparel 100.0% (32/32), Non-Leaf Plant Parts 100.0% (35/35), Soil & Ground 96.43% (27/28), Buildings & Structures 94.12% (16/17).
  * Crop-Wise Test Leaf Retention: Tomato 100.0% (31/31), Grape 100.0% (31/31), Chilli 100.0% (31/31), Sugarcane 100.0% (30/30).
- **Track D Unrelated Stress Evaluation ($n=20$ Distractor Images):**
  * **19/19 actual non-leaf distractors rejected = 100.00%** (mean predicted leaf prob = 0.0009 across actual non-leaf distractors).
  * `aloeL.jpg` was excluded from the non-leaf denominator because it is biologically an Aloe Vera leaf plant photograph; the model predicted $p=0.9986$ (Leaf), which is physically and semantically correct for a generic leaf detector.
- **Held-Out Real-World Field Leaf Retention (Zero-Shot Generalization):**
  * Evaluated across 2,229 uncurated field images: **2,150 / 2,229 = 96.55% overall field leaf retention** at default $\theta=0.50$.
  * Track B External Benchmark: 95.41% retained (603 / 632 field leaves passed; mean prob = 0.9575).
  * Track C Real-World Benchmark: 97.65% retained (415 / 425 field leaves passed; mean prob = 0.9795).
  * Phase 18 Final Field Benchmark: 96.59% retained (1,132 / 1,172 field leaves passed; mean prob = 0.9673).
- **Decision Threshold Sweep & Operating Trade-Offs ($\theta \in [0.05, 0.95]$):**
  * $\theta = 0.20$ (**Leaf-retention-oriented operating point**): 97.73% field leaf retention across all 2,229 field leaves, 97.60% internal non-leaf rejection, 95.0% Track D rejection (100% of non-leaf distractors). Ideal for user-facing applications where rejecting a farmer's genuine leaf photo is far more harmful than admitting a rare borderline non-leaf distractor.
  * $\theta = 0.50$ (**Default balanced operating point**): 96.55% field leaf retention, 98.40% internal non-leaf rejection, 99.19% test accuracy.
  * $\theta = 0.90$ (**Stricter non-leaf filtering / higher specificity**): 99.20% internal non-leaf rejection, 99.60% test accuracy, at the cost of lower field-leaf retention (95.65% field retention; 4.35% false rejection rate on real-world field foliage).
- **Scientific Qualification:**
  * These empirical results provide strong evidence of useful leaf vs. non-leaf discrimination and demonstrate that generic foliage features generalize across real-world field conditions (96.55% field retention without field training).
  * However, these findings do **NOT** constitute proof of complete, universal, or solved out-of-distribution (OOD) detection. Performance remains bounded by the diversity of evaluated distractor classes.
- **Preserved Deliverables in `results/phase19b_4_gatekeeper/`:**
  * `dataset_manifest.csv` (1,606 sample records with local paths, splits, labels, SHA-256)
  * `split_summary.csv` (stratified split counts and percentages)
  * `provenance_registry.csv` (100% complete 7-field provenance metadata for all 803 negatives)
  * `acquisition_manifest.csv` (833 candidate query records)
  * `rejection_log.csv` (33 disqualified samples with specific reasons)
  * `data_audit_report.txt` (full pre-training integrity audit)
  * `eval_internal_metrics.csv` (validation and test performance metrics)
  * `eval_sourcewise_negatives.csv` (rejection rates across 4 sources)
  * `eval_subcategory_negatives.csv` (rejection rates across 5 subcategories)
  * `eval_cropwise_retention.csv` (retention rates across 4 crops)
  * `eval_track_d_stress.csv` (per-sample predictions on 20 Track D stress images)
  * `eval_field_retention.csv` (field leaf retention across Tracks B, C, and Phase 18)
  * `threshold_sweep.csv` (metrics across all 19 thresholds from 0.05 to 0.95)
  * `confusion_matrix_test.png` & `threshold_curves.png` (visual trade-off curves)
  * `gatekeeper_head.keras` & `gatekeeper_mobilenetv2.weights.h5` (trained model assets)
  * `phase19b_4_evaluation_report.txt` (consolidated empirical report)

##### Phase 19B-5: Missing Field Classes / Tomato-Grape Field Data (ACQUISITION COMPLETED ✅)
- **Role:** Source missing field disease classes and independent field data for Tomato and Grape to address domain sensitivity without contaminating held-out benchmarks.
- **Acquisition & Screening Summary:**
  * **Total Candidates Processed:** 248 images across 3 independent, authoritative open repositories.
  * **Total Images Accepted:** **247 authentic in-situ field images** (99.6% acceptance rate).
  * **Total Images Rejected:** 1 image (0.4%; `IMG_8834.JPG` rejected as within-batch exact duplicate).
  * **Accepted Breakdown by Crop:**
    - **Tomato ($n=71$ images):** Sourced from FieldPlant (Moupojou et al., 2023, CC BY 4.0; Cameroon):
      * `tomato_healthy`: **13 images** (100% of available source images)
      * `tomato_mosaic_virus`: **13 images** (100% of available source images)
      * `tomato_yellow_leaf_curl_virus`: **45 images** (selected from 65 available)
      * *(Taxonomic note: "Tomato Brown Spots" [952 images] and "Tomato blight leaf" [359 images] were strictly excluded to prevent invalid cross-pathogen remapping).*
    - **Grape ($n=176$ images):** Sourced from HERMOS (Turkey) and Standalone ESCA (Marche, Italy):
      * `grape_powdery_mildew`: **45 images** (HERMOS; pure single-disease vineyard foliage)
      * `grape_downy_mildew`: **44 images** (HERMOS; 35 pure single-disease + 9 with background canopy foliage)
      * `grape_healthy`: **43 images** (HERMOS; pure healthy vineyard foliage)
      * `grape_esca`: **44 images** (Standalone ESCA dataset; Marche, Italy)
      * *(Pathology note: Dead Arm / Phomopsis viticola and non-plant annotations were strictly excluded).*
  * **Resolution Tiers:** Ultra-High ($\ge 2500\text{px}$): 180 images (72.9%), High ($1200\text{--}2499\text{px}$): 56 images (22.7%), Medium ($500\text{--}1199\text{px}$): 11 images (4.5%).
- **Contamination Screening & Deduplication:**
  * **Exhaustive Screening Registry:** 38,376 records compiled with 31,805 unique SHA-256 hashes across Track B (632), Track C (425), Track D (20), Phase 18 (1,836), Exp 5 (115), Candidate D (2,152), Gatekeeper 19B-4 (1,606), and internal splits for all 4 crops (31,590).
  * **Zero Benchmark Collision:** Exactly **0** SHA-256 collisions against any benchmark asset.
  * **Zero Internal Leakage:** Exactly **0** SHA-256 collisions against any internal training/val/test split.
  * **Perceptual Near-Duplicate Screening:** Exactly **0** perceptual near-duplicates (64-bit dHash Hamming distance $\le 5$ against 5,160 held-out evaluation images).
  * **Disqualified Aggregators Preserved:** FieldVitis (aggregates PlantDoc), PlantDoc (active benchmark), GLDD (140 benchmark images), NGLD (internal G1), and GVLiD (internal G2) were strictly excluded from acquisition.
- **Preserved Deliverables in `results/phase19b_5_field_data_gap/` and `experiments/field_data_19b5/manifests/`:**
  * `acquisition_manifest.csv` (248 candidate records with dimensions, file sizes, hashes, and acquisition status)
  * `accepted_field_data.csv` (247 accepted records with relative paths, canonical classes, resolution tiers, DOI, and license)
  * `rejection_log.csv` (1 rejected record documenting within-batch duplicate hash)
  * `provenance_registry.csv` (complete institutional provenance for FieldPlant, HERMOS, and ESCA Standalone)
  * `protected_benchmark_hashes.csv` (38,376 entries across all project splits and benchmarks)
  * `final_acquisition_summary.txt` (full closeout audit report)
  * `current_field_coverage.csv`, `class_gap_analysis.csv`, `candidate_source_audit.csv`, `sourcing_recommendation.txt`
- **Registry Terminology Note:** The protected registry encompasses 38,376 records (31,805 unique SHA-256 hashes); evaluation benchmarks within this registry consist of 632 unique images (Track B), 425 unique images (Track C), 20 unique images (Track D), 1,172 unique physical field images (Phase 18 benchmark, recorded across 1,836 evaluation prediction entries), 115 unique images (Chilli Exp 5 PlantDoc), 2,152 unique field images (Chilli Candidate D), 1,606 unique images (Gatekeeper 19B-4), and 31,590 split-file records across the 4 crop datasets.
- **Status:** Phase 19B-5 COMPLETED ✅. All 247 images curated, screened, verified collision-free, and staged.

##### Phase 19B-6: Field-Data-Assisted Retraining Experiment (COMPLETED ✅)
- **Role:** Empirical retraining experiment on Kaggle GPU to determine whether adding genuinely independent field imagery to training splits improves real-world Tomato and Grape disease classification without degrading in-domain performance.
- **Authoritative Findings:**
  * **Tomato In-Domain Preservation:** Internal test accuracy remained solidly preserved: 90.23% Baseline vs **89.72% Field-Adapted** ($\Delta -0.50$ pp, Macro F1 87.80% $\rightarrow$ 87.16%). (Intermediate 90.14% figure was resolved as from an aborted Version 2 run; 89.72% from completed Version 3 is authoritative).
  * **Tomato Field Gains:** Track C Real-World accuracy improved from 24.47% $\rightarrow$ **27.66%** (+3.19 pp, Macro F1 +4.73 pp); Phase 18 Field accuracy improved from 18.91% $\rightarrow$ **20.22%** (+1.30 pp, Macro F1 +5.25 pp).
  * **Tomato Target Class Recall (Phase 18):** Yellow Leaf Curl Virus recall surged from 5.36% $\rightarrow$ **42.86%** (+37.50 pp); Healthy recall surged from 0.00% $\rightarrow$ **27.59%** (+27.59 pp); Mosaic recall improved from 0.00% $\rightarrow$ **7.50%** (+7.50 pp).
  * **Grape In-Domain Preservation:** Internal test accuracy was preserved at **87.86%** vs 89.04% Baseline ($\Delta -1.18$ pp, Macro F1 86.31% $\rightarrow$ 83.28%).
  * **Grape Field Gains:** Track C Real-World accuracy improved from 7.00% $\rightarrow$ **9.00%** (+2.00 pp); Phase 18 Field accuracy improved from 11.43% $\rightarrow$ **13.21%** (+1.79 pp, Macro F1 +2.00 pp).
  * **Grape Target Class Recall (Phase 18):** Downy Mildew recall surged from 4.00% $\rightarrow$ **26.00%** (+22.00 pp); Healthy Leaves F1 improved from 30.12% $\rightarrow$ **42.31%** (+12.19 pp).
  * **Attractor Sink Breakup:** Probability mass was redistributed away from dominant attractor sinks (Early/Late Blight in Tomato; Healthy in Grape) toward legitimate field representations without harming unaffected classes.
- **Scientific Guardrails & Limitations:** Adding the selected independently collected field images was associated with modest overall field-benchmark gains and larger class-specific changes, while internal test performance decreased modestly. The intervention did NOT eliminate the broad generalization gap. Production models in `models/` remain locked and unchanged; adapted models remain offline research assets in `experiments/field_data_19b5/models/`.
- **Status:** Phase 19B-6 COMPLETED ✅.

##### Phase 19B-7: Final Research Synthesis & Academic Conclusion (NEXT / UPCOMING ⏳)
- **Role:** Master academic synthesis consolidating all quantitative empirical findings across Phases 1–19B-6 into unified dissertation tables, cross-phase error characterization, discussion of limitations, and architectural recommendations for future agricultural vision systems.

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
│   ├── chilli_independent_validation/<-- Chilli Experiment 5 (PlantDoc Validation)
│   │   ├── source_provenance.csv      <-- Full provenance records for 115 PlantDoc images
│   │   ├── source_manifest.csv        <-- Manifest with labels and SHA-256 hashes
│   │   ├── overlap_report.txt         <-- Zero-leakage audit report
│   │   ├── evaluation_metrics.csv     <-- Summary metrics (Baseline, Cand B, Cand D)
│   │   ├── class_metrics.csv          <-- Per-class metrics (with unrepresented disclosures)
│   │   ├── confusion_matrix.csv       <-- 5x5 contingency tables
│   │   ├── comparison.csv             <-- Comparison across models
│   │   ├── validation_report.txt      <-- Comprehensive validation report
│   │   ├── confusion_matrices.png     <-- Visualized confusion matrices
│   │   └── validation_images/         <-- 115 curated validation images
│   │
│   └── phase19b_2_crop_identifier/    <-- Phase 19B-2 Crop Identifier Models (Offline Research)
│       ├── model_a_baseline/          <-- Model A (unaugmented, 22,110 samples)
│       └── model_b_augmented/         <-- Model B (augmented, 44,220 samples)
│
├── results/                           <-- Evaluation Result Suites & Analysis Outputs
│   ├── tomato/                        <-- Baseline metrics, report, confusion matrix, history
│   ├── grape_unified/                 <-- Baseline metrics, report, confusion matrix, history
│   ├── chilli_cold/                   <-- Baseline metrics, report, confusion matrix, history
│   ├── sugarcane_unified/             <-- Baseline metrics, report, confusion matrix, history
│   ├── comparative_analysis/          <-- Phase 9 comparative metrics, reports, confusion matrices
│   ├── error_analysis/                <-- Phase 10 error summary, class analyses, sample catalog
│   ├── gradcam/                       <-- Phase 15 diagnostic case directories & consolidated report
│   ├── model_robustness_audit/        <-- Phase 16 4-Track Audit deliverables (13 reports + 5 charts)
│   ├── final_realworld_benchmark/     <-- Phase 18 clean multi-crop benchmark (1,172 samples)
│   ├── phase19b_1_calibration/        <-- Phase 19B-1 temperature scaling & rejection deliverables
│   │   ├── calibration_summary.csv    <-- Master calibration summary
│   │   ├── temperature_scaling_results.csv
│   │   ├── threshold_sweep_validation.csv
│   │   ├── threshold_sweep_heldout.csv
│   │   ├── ece_results.csv
│   │   ├── brier_results.csv
│   │   ├── reliability_data.csv
│   │   ├── rejection_analysis.csv
│   │   └── phase19b_1_report.txt      <-- Consolidated technical report
│   │
│   ├── phase19b_2_crop_identifier/    <-- Phase 19B-2 Crop Species Identifier evaluation suite
│   │   ├── training_summary.csv
│   │   ├── crop_identifier_comparison.csv
│   │   ├── validation_metrics.csv
│   │   ├── internal_test_metrics.csv
│   │   ├── cross_crop_metrics.csv
│   │   ├── field_benchmark_metrics.csv
│   │   ├── per_class_metrics.csv
│   │   ├── confusion_matrix.csv
│   │   ├── per_sample_predictions.csv
│   │   ├── mismatch_detection_results.csv
│   │   └── phase19b_2_report.txt
│   │
│   ├── phase19b_2a_error_analysis/    <-- Phase 19B-2A Crop Identifier Field Error Analysis suite
│   │   ├── field_confusion_matrix.csv
│   │   ├── source_confusion_analysis.csv
│   │   ├── confidence_analysis.csv
│   │   ├── model_a_vs_b_errors.csv
│   │   ├── representative_errors.csv
│   │   └── phase19b_2a_report.txt
│   │
│   ├── phase19b_3_nonleaf_sourcing/   <-- Phase 19B-3 Non-Leaf Sourcing & Dataset Design
│   │   ├── source_audit.csv           <-- Legal, licensing, and suitability audit of 10 candidate sources
│   │   ├── source_audit.txt           <-- Detailed sourcing narrative and selection rationale
│   │   └── dataset_design.txt         <-- Full dataset architecture, boundary rules, and 4-tier protocol
│   │
│   └── phase19b_4_gatekeeper/         <-- Phase 19B-4 Binary Leaf/Non-Leaf Gatekeeper Evaluation Suite
│       ├── dataset_manifest.csv       <-- Complete 1,606-sample dataset manifest (splits, labels, SHA-256)
│       ├── split_summary.csv          <-- Stratified split distribution (70/15/15)
│       ├── provenance_registry.csv    <-- 100% complete 7-field provenance metadata for 803 negatives
│       ├── acquisition_manifest.csv   <-- 833 candidate query records
│       ├── rejection_log.csv          <-- Rejection logs for 33 disqualified samples
│       ├── data_audit_report.txt      <-- Pre-training dataset integrity audit report
│       ├── eval_internal_metrics.csv  <-- Validation and test performance metrics
│       ├── eval_sourcewise_negatives.csv <-- Rejection rates across 4 sources
│       ├── eval_subcategory_negatives.csv <-- Rejection rates across 5 subcategories
│       ├── eval_cropwise_retention.csv <-- Retention rates across 4 crop foliages
│       ├── eval_track_d_stress.csv    <-- Predictions on 20 Track D stress distractor images
│       ├── eval_field_retention.csv   <-- Field leaf retention across Tracks B, C, and Phase 18
│       ├── threshold_sweep.csv        <-- Multi-track metrics across 19 decision thresholds
│       ├── confusion_matrix_test.png  <-- Visual test split confusion matrix
│       ├── threshold_curves.png       <-- Sensitivity vs. specificity vs. field retention curve
│       ├── gatekeeper_head.keras      <-- Trained binary classifier head
│       ├── gatekeeper_mobilenetv2.weights.h5 <-- Model weights checkpoint
│       └── phase19b_4_evaluation_report.txt <-- Consolidated technical evaluation report
│
│   ├── phase19b_5_field_data_gap/     <-- Phase 19B-5 Tomato & Grape Field Data Gap Audit
│   │   ├── current_field_coverage.csv <-- Class-by-class coverage & failure mode audit
│   │   ├── class_gap_analysis.csv     <-- Prioritized target sample counts and visual requirements
│   │   ├── candidate_source_audit.csv <-- 9 candidate repositories audited for licensing & overlap
│   │   ├── sourcing_recommendation.txt <-- Complete sourcing strategy & contamination protocol
│   │   └── final_acquisition_summary.txt <-- Manifest & clean hash screening report (247 images)
│   │
│   └── phase19b_6_field_adaptation/   <-- Phase 19B-6 Field Data Retraining & Evaluation Suite
│       ├── baseline_metrics.csv       <-- Production baseline benchmark metrics across all tracks
│       ├── baseline_classwise_metrics.csv <-- Production baseline classwise metrics
│       ├── field_adapted_metrics.csv  <-- Adapted model metrics across all tracks
│       ├── track_comparison.csv       <-- Delta comparison across Internal Test, Track B, Track C, Phase 18
│       ├── classwise_comparison.csv   <-- Per-class precision, recall, and F1 deltas
│       ├── confusion_matrix_tomato.png<-- 2x2 comparison (Baseline vs Adapted on Internal & Phase 18)
│       ├── confusion_matrix_grape.png <-- 2x2 comparison (Baseline vs Adapted on Internal & Phase 18)
│       ├── tomato_training_history.csv<-- Kaggle GPU training metrics by epoch
│       ├── grape_training_history.csv <-- Kaggle GPU training metrics by epoch
│       ├── training_summary.txt       <-- GPU training execution times and convergence log
│       └── phase19b_6_report.txt      <-- Consolidated empirical evaluation report
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
