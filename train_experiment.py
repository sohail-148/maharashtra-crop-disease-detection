"""
train_experiment.py — Parameterized MobileNetV2 Baseline Pipeline
Phase 8: Experiments G1, G2, C1, S1, S2

Usage (local Windows):
    python train_experiment.py --experiment G1

Usage (AWS, with dataset root override):
    python train_experiment.py --experiment G1 --dataset-root /data/datasets

The --dataset-root flag rewrites the file_path column in the split CSVs at
runtime (in memory only — original CSV files are never modified).

Environment variable alternative (takes lower priority than --dataset-root):
    export DATASET_ROOT=/data/datasets
    python train_experiment.py --experiment G1

Baseline rules (identical to T1 Tomato):
  - Pretrained MobileNetV2 (ImageNet), base layers FROZEN
  - Lightweight classification head: GAP → BN → Dropout(0.3) → Dense(N)
  - On-the-fly augmentation on training images only
  - EarlyStopping patience=5 on val_loss, save best model
  - No fine-tuning, no class weighting, no oversampling
  - Test set touched only at final evaluation

Outputs saved under:
    models/<experiment_name>/   — .keras model + summary
    results/<experiment_name>/  — CSVs, PNGs, text report
"""

import os
import sys
import time
import random
import argparse
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

# ---------------------------------------------------------------------------
# Experiment registry
# Approved 4-Crop Architecture Remaining Training Jobs:
#   GRAPE     -> grape_unified (G1 + G2, 7 classes)
#   CHILLI    -> chilli_cold (C1, 5 classes)
#   SUGARCANE -> sugarcane_unified (S1 + S2, 11 classes)
# (Legacy G1, G2, C1, S1, S2 retained for standalone experiment training)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # Approved 4-Crop Architecture Jobs:
    "GRAPE": {
        "splits_name":    "grape_unified",
        "dataset_folders": ["grape_niphad", "grape_2024"],
        "display":        "Grape Unified — G1 Niphad + G2 2024 (7 Classes)",
    },
    "CHILLI": {
        "splits_name":    "chilli_cold",
        "dataset_folders": ["chilli_cold"],
        "display":        "Chilli — C1 COLD 2024 (5 Classes)",
    },
    "SUGARCANE": {
        "splits_name":    "sugarcane_unified",
        "dataset_folders": ["sugarcane_maharashtra", "sugarcane_large"],
        "display":        "Sugarcane Unified — S1 Maharashtra + S2 Large (11 Classes)",
    },
    # Legacy individual dataset jobs:
    "G1": {
        "splits_name":    "grape_niphad",
        "dataset_folders": ["grape_niphad"],
        "display":        "G1 Grape — Niphad",
    },
    "G2": {
        "splits_name":    "grape_2024",
        "dataset_folders": ["grape_2024"],
        "display":        "G2 Grape — 2024",
    },
    "C1": {
        "splits_name":    "chilli_cold",
        "dataset_folders": ["chilli_cold"],
        "display":        "C1 Chilli — COLD",
    },
    "S1": {
        "splits_name":    "sugarcane_maharashtra",
        "dataset_folders": ["sugarcane_maharashtra"],
        "display":        "S1 Sugarcane — Maharashtra",
    },
    "S2": {
        "splits_name":    "sugarcane_large",
        "dataset_folders": ["sugarcane_large"],
        "display":        "S2 Sugarcane — Large",
    },
}

ALL_DATASET_FOLDERS = [
    "tomato_plantvillage", "grape_niphad", "grape_2024",
    "chilli_cold", "sugarcane_maharashtra", "sugarcane_large"
]

# ---------------------------------------------------------------------------
# Fixed hyperparameters — identical to T1 baseline
# ---------------------------------------------------------------------------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
DROPOUT    = 0.3
SEED       = 42


# ---------------------------------------------------------------------------
# Path rewriting
# ---------------------------------------------------------------------------

def rewrite_paths(df: pd.DataFrame, new_root: str) -> pd.DataFrame:
    """
    Replace the Windows absolute prefix in file_path column with new_root.

    The split CSVs store paths like:
        D:\\CropDiseaseProject\\grape_niphad\\Healthy Leaves\\img.jpg

    On Linux/Cloud (Kaggle/Colab/AWS) the equivalent is:
        /content/datasets/grape_niphad/Healthy Leaves/img.jpg

    Strategy: strip everything up to and including the dataset folder name,
    then prepend new_root.  This is robust regardless of what the original
    Windows drive / project directory was.
    """
    def _fix(path: str) -> str:
        # Normalise backslashes to forward slashes
        p = path.replace("\\", "/")
        # Find the dataset folder component
        for folder_name in ALL_DATASET_FOLDERS:
            marker = "/" + folder_name + "/"
            idx = p.find(marker)
            if idx != -1:
                # Keep from the dataset folder name onwards
                relative = p[idx + 1:]          # e.g. grape_niphad/Healthy Leaves/img.jpg
                return os.path.join(new_root, relative).replace("\\", "/")
        # Fallback: just normalise slashes
        return p

    df = df.copy()
    df["file_path"] = df["file_path"].apply(_fix)
    return df


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(file_path, label):
    raw   = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(
        image,
        k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
    )
    crop_size = tf.random.uniform([], 0.80, 1.0)
    h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * crop_size, tf.int32)
    w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * crop_size, tf.int32)
    image = tf.image.random_crop(image, size=[h, w, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def make_dataset(df, training=False, batch_size=BATCH_SIZE):
    file_paths = df["file_path"].values
    labels     = df["class_index"].values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes: int):
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model   = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp_id, dataset_root=None, batch_size=BATCH_SIZE):
    # type: (str, Optional[str], int) -> dict
    # --- Reproducibility ---
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    cfg = EXPERIMENTS[exp_id]
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # --- Directories ---
    splits_dir  = os.path.join(PROJECT_ROOT, "splits", cfg["splits_name"])
    models_dir  = os.path.join(PROJECT_ROOT, "models",  cfg["splits_name"])
    results_dir = os.path.join(PROJECT_ROOT, "results", cfg["splits_name"])
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path   = os.path.join(models_dir,  f"{cfg['splits_name']}_baseline.keras")
    history_csv  = os.path.join(results_dir, "training_history.csv")
    history_png  = os.path.join(results_dir, "training_history.png")
    report_txt   = os.path.join(results_dir, "test_results.txt")
    metrics_csv  = os.path.join(results_dir, "test_metrics.csv")
    cm_png       = os.path.join(results_dir, "confusion_matrix.png")
    cm_csv       = os.path.join(results_dir, "confusion_matrix.csv")
    summary_txt  = os.path.join(models_dir,  "model_summary.txt")

    # --- Header ---
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Experiment {exp_id}: {cfg['display']}")
    print(sep)

    # --- Load splits ---
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df  = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    idx_df   = pd.read_csv(os.path.join(splits_dir, "class_index.csv"))

    # --- Rewrite paths if running on AWS ---
    effective_root = dataset_root or os.environ.get("DATASET_ROOT")
    if effective_root:
        print(f"  Rewriting file paths -> root: {effective_root}")
        train_df = rewrite_paths(train_df, effective_root)
        val_df   = rewrite_paths(val_df,   effective_root)
        test_df  = rewrite_paths(test_df,  effective_root)

    num_classes  = len(idx_df)
    class_names  = idx_df.sort_values("class_index")["class_label"].tolist()

    print(f"\n  Train : {len(train_df):,} images")
    print(f"  Val   : {len(val_df):,} images")
    print(f"  Test  : {len(test_df):,} images")
    print(f"  Classes: {num_classes}  ->  {class_names}")
    print(f"  Batch size: {batch_size}")

    # --- Verify first path is readable (fast sanity check) ---
    sample_path = train_df["file_path"].iloc[0]
    if not os.path.exists(sample_path):
        print(f"\n  WARNING: Sample path does not exist: {sample_path}")
        print("  If running on AWS, make sure --dataset-root or DATASET_ROOT is set correctly.")

    # --- tf.data ---
    print("\n  Building tf.data pipelines ...")
    train_ds = make_dataset(train_df, training=True,  batch_size=batch_size)
    val_ds   = make_dataset(val_df,   training=False, batch_size=batch_size)
    test_ds  = make_dataset(test_df,  training=False, batch_size=batch_size)

    # --- Model ---
    print("  Building model ...")
    model, base = build_model(num_classes)

    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable    = total_params - trainable_params

    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write(f"\n\nTotal params     : {total_params:,}")
        f.write(f"\nTrainable params : {trainable_params:,}")
        f.write(f"\nFrozen params    : {non_trainable:,}")

    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}  (head only)")
    print(f"  Frozen params    : {non_trainable:,}  (MobileNetV2 base)")

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1,
        ),
        CSVLogger(history_csv),
    ]

    # --- Train ---
    print(f"\n  Training  (max {EPOCHS} epochs, early stopping patience=5)")
    print("-" * 68)

    t_start  = time.time()
    history  = model.fit(
        train_ds, epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks, verbose=1,
    )
    elapsed    = time.time() - t_start
    epochs_run = len(history.history["loss"])

    print(f"\n  Training finished in {elapsed/60:.1f} min ({epochs_run} epochs)")

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{cfg['display']} — MobileNetV2 Baseline Training History", fontsize=13)
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(history_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluate ---
    print("\n  Evaluating on test set ...")
    y_true, y_pred = [], []
    for images, labels_batch in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels_batch.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_acc  = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    test_rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    test_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Test Precision : {test_prec*100:.2f}%  (weighted)")
    print(f"  Test Recall    : {test_rec*100:.2f}%  (weighted)")
    print(f"  Test F1-score  : {test_f1*100:.2f}%  (weighted)")

    # Clean class name display (strip dataset prefixes if present)
    short_names = [c.replace("_", " ") for c in class_names]

    report = classification_report(
        y_true, y_pred, target_names=short_names, digits=4, zero_division=0,
    )
    print("\n" + report)

    # Save text report
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"{cfg['display']} — MobileNetV2 Baseline: Test Set Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy  : {test_acc*100:.2f}%\n")
        f.write(f"Precision : {test_prec*100:.2f}%  (weighted)\n")
        f.write(f"Recall    : {test_rec*100:.2f}%  (weighted)\n")
        f.write(f"F1-score  : {test_f1*100:.2f}%  (weighted)\n\n")
        f.write(f"Epochs trained : {epochs_run}\n")
        f.write(f"Training time  : {elapsed/60:.1f} min\n")
        f.write(f"Total params   : {total_params:,}\n")
        f.write(f"Trainable      : {trainable_params:,}\n\n")
        f.write("Per-class report:\n")
        f.write(report)

    # Per-class metrics CSV
    report_dict = classification_report(
        y_true, y_pred, target_names=short_names,
        output_dict=True, zero_division=0,
    )
    metrics_rows = [
        {
            "class":     cls,
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in short_names
    ]
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    # Confusion matrix
    cm     = confusion_matrix(y_true, y_pred)
    cm_df  = pd.DataFrame(cm, index=short_names, columns=short_names)
    cm_df.to_csv(cm_csv)

    fig_h  = max(8, num_classes)
    fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{cfg['display']} — Confusion Matrix (Test Set)", fontsize=12)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Final summary ---
    best_val_acc  = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    print(f"\n{sep}")
    print(f"  {exp_id} COMPLETE")
    print(sep)
    print(f"  Epochs        : {epochs_run}  |  Time : {elapsed/60:.1f} min")
    print(f"  Best val acc  : {best_val_acc*100:.2f}%  |  Best val loss : {best_val_loss:.4f}")
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print(f"  Test F1       : {test_f1*100:.2f}%  (weighted)")
    print(f"  Model saved   : {os.path.relpath(model_path)}")
    print(f"  Results dir   : {os.path.relpath(results_dir)}")
    print(sep)

    return {
        "experiment":      exp_id,
        "display":         cfg["display"],
        "epochs":          epochs_run,
        "time_min":        round(elapsed / 60, 1),
        "best_val_acc":    round(best_val_acc * 100, 2),
        "test_accuracy":   round(test_acc * 100, 2),
        "test_f1":         round(test_f1 * 100, 2),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNetV2 baseline training — single experiment"
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment ID: GRAPE, CHILLI, SUGARCANE (or legacy G1, G2, C1, S1, S2)",
    )
    parser.add_argument(
        "--dataset-root", "-d",
        default=None,
        help=(
            "Absolute path to the directory that CONTAINS the dataset folders "
            "(e.g. /data/datasets on AWS). Overrides DATASET_ROOT env var. "
            "Leave unset to use the original Windows paths from the split CSVs "
            "(i.e. running locally)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}; increase to 64 on GPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        args.experiment,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
    )
