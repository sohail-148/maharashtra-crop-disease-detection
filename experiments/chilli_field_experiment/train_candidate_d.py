"""
train_candidate_d.py — Chilli Experiment 4: Multi-Source Field-Data Generalization

Training Script for Candidate D (GPU-Accelerated):
- Architecture: MobileNetV2 ImageNet + GAP + BN + Dropout(0.3) + Dense(5)
- Partial Fine-Tuning: Upper 29 layers unfrozen (base BN layers frozen)
- Augmentation: Candidate B realistic field transformations (flip, zoom, brightness, contrast)
- Optimizer: Adam (LR = 1e-4)
- Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau(factor=0.2, patience=3)
- Training Data: Multi-source field_data_manifest.csv (2,152 images: 1,352 original + 800 clean field)
- Validation Data: Official held-out val.csv (290 images)
- Evaluation Sets:
  1. Official Test Split (290 images)
  2. Track B Held-Out External Set (120 images)
  3. Track C Real-World Field Set (97 images)
  4. Track C High-Quality Subset (77 images)
- Compares: Baseline vs Candidate B vs Candidate D
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Reproducibility & Device Setup
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def detect_hardware():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[Hardware] GPU Detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"  Note on memory growth: {e}")
    else:
        print("[Hardware] No GPU detected. Running on CPU.")
    return len(gpus) > 0

# ---------------------------------------------------------------------------
# 2. Canonical Class Mapping
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "cerocospora",
    "healthy",
    "murda complex",
    "nutritional deficiency",
    "powdery mildew"
]
NUM_CLASSES = 5
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# 3. Data Loading & Candidate B Realistic Field Augmentation
# ---------------------------------------------------------------------------
def parse_image_train(filename, label):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    # Candidate B realistic field augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    
    # Moderate random crop / zoom (85% to 100%)
    crop_scale = tf.random.uniform([], 0.85, 1.0)
    crop_h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * crop_scale, tf.int32)
    crop_w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * crop_scale, tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMG_SIZE)
    
    # Clip values to valid MobileNetV2 range [-1.0, 1.0]
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, label

def parse_image_eval(filename, label):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

def create_dataset(filenames, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(filenames), seed=SEED)
        ds = ds.map(parse_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(parse_image_eval, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------------
# 4. Model Architecture Construction (Candidate B Configuration)
# ---------------------------------------------------------------------------
def build_candidate_d_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze upper 29 layers; freeze lower 125 layers
    base_model.trainable = True
    for layer in base_model.layers[:-29]:
        layer.trainable = False
        
    # Crucial: keep all BatchNormalization layers inside base frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    # Classification head
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_tensor")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout", seed=SEED)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Chilli_Candidate_D_MultiSource")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------------------------------------------------------
# 5. Evaluation Helper
# ---------------------------------------------------------------------------
def evaluate_dataset(model, filenames, true_labels, dataset_name="Evaluation Set"):
    ds = create_dataset(filenames, true_labels, is_training=False)
    preds = model.predict(ds, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    
    per_class_f1 = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = (np.array(true_labels) == i)
        if np.sum(mask) > 0:
            c_f1 = f1_score(true_labels == i, pred_labels == i, zero_division=0)
            per_class_f1[cls] = c_f1
        else:
            per_class_f1[cls] = None
            
    return {
        "dataset_name": dataset_name,
        "sample_count": len(filenames),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(np.mean(confidences)),
        "per_class_f1": per_class_f1,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "confidences": confidences
    }

# ---------------------------------------------------------------------------
# 6. Main Execution Pipeline
# ---------------------------------------------------------------------------
def run_experiment(project_root, output_dir=None, epochs=30):
    detect_hardware()
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "experiments", "chilli_field_experiment", "candidate_d")
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_path = os.path.join(project_root, "experiments", "chilli_field_experiment", "field_data_manifest.csv")
    val_csv_path = os.path.join(project_root, "splits", "chilli_cold", "val.csv")
    test_csv_path = os.path.join(project_root, "splits", "chilli_cold", "test.csv")
    ext_csv_path = os.path.join(project_root, "results", "model_robustness_audit", "external_provenance.csv")
    rw_csv_path = os.path.join(project_root, "results", "model_robustness_audit", "realworld_provenance.csv")
    
    for p, name in [(manifest_path, "Manifest"), (val_csv_path, "Validation CSV"), (test_csv_path, "Test CSV")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {name} at {p}")
            
    print(f"\n[Data] Loading Multi-Source Training Manifest: {manifest_path}")
    train_df = pd.read_csv(manifest_path)
    train_paths = train_df["image_path"].values
    train_labels = train_df["class_index"].values
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Class breakdown:\n{train_df['class_label'].value_counts().to_string()}")
    
    print(f"\n[Data] Loading Official Validation Split: {val_csv_path}")
    val_df = pd.read_csv(val_csv_path)
    val_paths = val_df["file_path"].values
    val_labels = val_df["class_index"].values
    print(f"  Validation samples: {len(val_paths)}")
    
    print(f"\n[Data] Loading Official Test Split: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    test_paths = test_df["file_path"].values
    test_labels = test_df["class_index"].values
    print(f"  Test split samples: {len(test_paths)}")
    
    ext_df = pd.read_csv(ext_csv_path)
    chilli_ext = ext_df[ext_df["crop"].str.lower() == "chilli"].copy()
    class_map = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    chilli_ext["class_index"] = chilli_ext["class_label"].map(class_map)
    chilli_ext = chilli_ext.dropna(subset=["class_index"])
    chilli_ext["class_index"] = chilli_ext["class_index"].astype(int)
    print(f"\n[Data] Loaded Track B External Set: {len(chilli_ext)} samples")
    
    rw_df = pd.read_csv(rw_csv_path)
    chilli_rw = rw_df[rw_df["crop"].str.lower() == "chilli"].copy()
    chilli_rw["class_index"] = chilli_rw["class_label"].map(class_map)
    chilli_rw = chilli_rw.dropna(subset=["class_index"])
    chilli_rw["class_index"] = chilli_rw["class_index"].astype(int)
    chilli_rw_high = chilli_rw[chilli_rw["ground_truth_quality"] == "HIGH"].copy()
    print(f"[Data] Loaded Track C Real-World Set: {len(chilli_rw)} samples (High Quality: {len(chilli_rw_high)})")
    
    train_ds = create_dataset(train_paths, train_labels, is_training=True)
    val_ds = create_dataset(val_paths, val_labels, is_training=False)
    
    print("\n[Model] Building Candidate D Model (Partial Fine-Tuning + Candidate B Augmentation)...")
    model = build_candidate_d_model()
    
    best_model_path = os.path.join(output_dir, "model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\n[Training] Starting Candidate D Training (Max Epochs: {epochs}, Batch Size: {BATCH_SIZE})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    print("\n================================================================================")
    print("INDEPENDENT 4-WAY EVALUATION SUITE FOR CANDIDATE D")
    print("================================================================================")
    
    res_test = evaluate_dataset(model, test_paths, test_labels, "Official In-Domain Test Split")
    res_track_b = evaluate_dataset(model, chilli_ext["image_path"].values, chilli_ext["class_index"].values, "Track B: Independent External (Ulfa)")
    res_track_c_all = evaluate_dataset(model, chilli_rw["image_path"].values, chilli_rw["class_index"].values, "Track C: Real-World Field (All Qualities)")
    res_track_c_high = evaluate_dataset(model, chilli_rw_high["image_path"].values, chilli_rw_high["class_index"].values, "Track C: Real-World Field (High Quality Only)")
    
    eval_results = [res_test, res_track_b, res_track_c_all, res_track_c_high]
    for res in eval_results:
        print(f"\n--- {res['dataset_name']} (N = {res['sample_count']}) ---")
        print(f"  Accuracy:         {res['accuracy']*100:.2f}%")
        print(f"  Weighted F1:      {res['weighted_f1']*100:.2f}%")
        print(f"  Macro F1:         {res['macro_f1']*100:.2f}%")
        print(f"  Mean Confidence:  {res['mean_confidence']*100:.2f}%")
        print("  Per-Class F1:")
        for cls, f1_val in res['per_class_f1'].items():
            f1_str = f"{f1_val*100:.2f}%" if f1_val is not None else "N/A (0 support)"
            print(f"    * {cls:<25}: {f1_str}")
            
    # Reference metrics
    ref_baseline = {
        "Test Accuracy": 63.79, "Test Weighted F1": 62.36, "Test Macro F1": 57.75,
        "Track B Acc": 55.00, "Track C All Acc": 57.73, "Track C High Acc": 50.65,
        "cercospora F1": 75.17, "healthy F1": 52.63, "murda F1": 50.00, "deficiency F1": 37.04, "mildew F1": 73.91
    }
    ref_cand_b = {
        "Test Accuracy": 73.45, "Test Weighted F1": 72.31, "Test Macro F1": 68.62,
        "Track B Acc": 52.50, "Track C All Acc": 54.64, "Track C High Acc": 44.16,
        "cercospora F1": 83.22, "healthy F1": 64.65, "murda F1": 65.06, "deficiency F1": 44.44, "mildew F1": 85.71
    }
    cand_d_metrics = {
        "Test Accuracy": res_test["accuracy"] * 100,
        "Test Weighted F1": res_test["weighted_f1"] * 100,
        "Test Macro F1": res_test["macro_f1"] * 100,
        "Track B Acc": res_track_b["accuracy"] * 100,
        "Track C All Acc": res_track_c_all["accuracy"] * 100,
        "Track C High Acc": res_track_c_high["accuracy"] * 100,
        "cercospora F1": (res_test["per_class_f1"]["cerocospora"] or 0) * 100,
        "healthy F1": (res_test["per_class_f1"]["healthy"] or 0) * 100,
        "murda F1": (res_test["per_class_f1"]["murda complex"] or 0) * 100,
        "deficiency F1": (res_test["per_class_f1"]["nutritional deficiency"] or 0) * 100,
        "mildew F1": (res_test["per_class_f1"]["powdery mildew"] or 0) * 100,
    }
    
    comp_df = pd.DataFrame([
        {"Metric": "Official Test Accuracy", "Baseline (Frozen)": f"{ref_baseline['Test Accuracy']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Test Accuracy']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Test Accuracy']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Test Accuracy'] - ref_baseline['Test Accuracy']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Test Accuracy'] - ref_cand_b['Test Accuracy']:+.2f}%"},
        {"Metric": "Official Test Weighted F1", "Baseline (Frozen)": f"{ref_baseline['Test Weighted F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Test Weighted F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Test Weighted F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Test Weighted F1'] - ref_baseline['Test Weighted F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Test Weighted F1'] - ref_cand_b['Test Weighted F1']:+.2f}%"},
        {"Metric": "Official Test Macro F1", "Baseline (Frozen)": f"{ref_baseline['Test Macro F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Test Macro F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Test Macro F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Test Macro F1'] - ref_baseline['Test Macro F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Test Macro F1'] - ref_cand_b['Test Macro F1']:+.2f}%"},
        {"Metric": "Track B External Benchmark", "Baseline (Frozen)": f"{ref_baseline['Track B Acc']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Track B Acc']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Track B Acc']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Track B Acc'] - ref_baseline['Track B Acc']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Track B Acc'] - ref_cand_b['Track B Acc']:+.2f}%"},
        {"Metric": "Track C Real-World (All)", "Baseline (Frozen)": f"{ref_baseline['Track C All Acc']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Track C All Acc']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Track C All Acc']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Track C All Acc'] - ref_baseline['Track C All Acc']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Track C All Acc'] - ref_cand_b['Track C All Acc']:+.2f}%"},
        {"Metric": "Track C Real-World (High)", "Baseline (Frozen)": f"{ref_baseline['Track C High Acc']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['Track C High Acc']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['Track C High Acc']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['Track C High Acc'] - ref_baseline['Track C High Acc']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['Track C High Acc'] - ref_cand_b['Track C High Acc']:+.2f}%"},
        {"Metric": "  * Cercospora F1", "Baseline (Frozen)": f"{ref_baseline['cercospora F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['cercospora F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['cercospora F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['cercospora F1'] - ref_baseline['cercospora F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['cercospora F1'] - ref_cand_b['cercospora F1']:+.2f}%"},
        {"Metric": "  * Healthy Leaves F1", "Baseline (Frozen)": f"{ref_baseline['healthy F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['healthy F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['healthy F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['healthy F1'] - ref_baseline['healthy F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['healthy F1'] - ref_cand_b['healthy F1']:+.2f}%"},
        {"Metric": "  * Murda Complex F1", "Baseline (Frozen)": f"{ref_baseline['murda F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['murda F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['murda F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['murda F1'] - ref_baseline['murda F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['murda F1'] - ref_cand_b['murda F1']:+.2f}%"},
        {"Metric": "  * Nutritional Def F1", "Baseline (Frozen)": f"{ref_baseline['deficiency F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['deficiency F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['deficiency F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['deficiency F1'] - ref_baseline['deficiency F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['deficiency F1'] - ref_cand_b['deficiency F1']:+.2f}%"},
        {"Metric": "  * Powdery Mildew F1", "Baseline (Frozen)": f"{ref_baseline['mildew F1']:.2f}%", "Candidate B (Fine-Tune+Aug)": f"{ref_cand_b['mildew F1']:.2f}%", "Candidate D (Multi-Source Field)": f"{cand_d_metrics['mildew F1']:.2f}%", "Delta (D - Baseline)": f"{cand_d_metrics['mildew F1'] - ref_baseline['mildew F1']:+.2f}%", "Delta (D - Cand B)": f"{cand_d_metrics['mildew F1'] - ref_cand_b['mildew F1']:+.2f}%"},
    ])
    
    print(comp_df.to_string(index=False))
    comp_csv_path = os.path.join(output_dir, "candidate_d_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n[Artifact] Saved comparison report to: {comp_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chilli Candidate D (Multi-Source Field-Data)")
    parser.add_argument("--project_root", type=str, default=r"D:\CropDiseaseProject", help="Project root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for Candidate D")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    args = parser.parse_args()
    
    run_experiment(project_root=args.project_root, output_dir=args.output_dir, epochs=args.epochs)
