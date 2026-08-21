"""
Phase 8 — T1 Tomato: MobileNetV2 Baseline

Baseline rules:
  - Pretrained MobileNetV2 (ImageNet), base layers FROZEN
  - Lightweight classification head only
  - On-the-fly augmentation on training images only
  - Early stopping on val_loss, save best model
  - No fine-tuning, no class weighting, no oversampling
  - Test set touched only at final evaluation

Outputs (under models/tomato/ and results/tomato/):
  - tomato_baseline.keras          best saved model
  - training_history.csv           epoch-by-epoch metrics
  - training_history.png           accuracy + loss curves
  - test_results.txt               classification report
  - test_metrics.csv               per-class precision/recall/F1
  - confusion_matrix.png           visualised confusion matrix
  - confusion_matrix.csv           raw counts
  - model_summary.txt              parameter count / architecture
"""

import os
import time
import random

# Suppress TF info/warnings before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR  = os.path.join(ROOT, "splits", "tomato")
MODELS_DIR  = os.path.join(ROOT, "models",  "tomato")
RESULTS_DIR = os.path.join(ROOT, "results", "tomato")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR,  "tomato_baseline.keras")
HISTORY_CSV  = os.path.join(RESULTS_DIR, "training_history.csv")
HISTORY_PNG  = os.path.join(RESULTS_DIR, "training_history.png")
REPORT_TXT   = os.path.join(RESULTS_DIR, "test_results.txt")
METRICS_CSV  = os.path.join(RESULTS_DIR, "test_metrics.csv")
CM_PNG       = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CM_CSV       = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
SUMMARY_TXT  = os.path.join(MODELS_DIR,  "model_summary.txt")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 30        # upper bound; early stopping usually triggers earlier
LR          = 1e-3      # head-only learning rate
DROPOUT     = 0.3
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# 1. Load split CSVs
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Phase 8 — T1 Tomato: MobileNetV2 Baseline")
print("=" * 65)

train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
idx_df   = pd.read_csv(os.path.join(SPLITS_DIR, "class_index.csv"))

class_names = idx_df.sort_values("class_index")["class_label"].tolist()
label2idx   = dict(zip(idx_df["class_label"], idx_df["class_index"]))

print(f"\n  Train : {len(train_df):,} images")
print(f"  Val   : {len(val_df):,} images")
print(f"  Test  : {len(test_df):,} images")
print(f"  Classes: {NUM_CLASSES}")

# ---------------------------------------------------------------------------
# 2. tf.data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(file_path, label):
    """Read image → decode → resize → MobileNetV2 preprocess."""
    raw   = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)   # scales to [-1, 1]
    return image, label


def augment(image, label):
    """On-the-fly augmentation for training images only."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(
        image,
        k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    # Random zoom via crop-and-resize
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

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


print("\n  Building tf.data pipelines ...")
train_ds = make_dataset(train_df, training=True)
val_ds   = make_dataset(val_df,   training=False)
test_ds  = make_dataset(test_df,  training=False)

# ---------------------------------------------------------------------------
# 3. Build model
# ---------------------------------------------------------------------------
print("  Building model ...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False    # freeze all base layers for baseline

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Save model summary
total_params     = model.count_params()
trainable_params = sum(
    tf.size(w).numpy() for w in model.trainable_weights
)
non_trainable    = total_params - trainable_params

summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))
summary_text = "\n".join(summary_lines)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary_text)
    f.write(f"\n\nTotal params     : {total_params:,}")
    f.write(f"\nTrainable params : {trainable_params:,}")
    f.write(f"\nFrozen params    : {non_trainable:,}")

print(f"\n  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}  (head only)")
print(f"  Frozen params    : {non_trainable:,}  (MobileNetV2 base)")

# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    CSVLogger(HISTORY_CSV),
]

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print(f"  Training  (max {EPOCHS} epochs, early stopping patience=5)")
print("-" * 65)

t_start = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)
t_end   = time.time()
elapsed = t_end - t_start
epochs_run = len(history.history["loss"])

print(f"\n  Training finished in {elapsed/60:.1f} min ({epochs_run} epochs)")

# ---------------------------------------------------------------------------
# 6. Plot training history
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("T1 Tomato — MobileNetV2 Baseline Training History", fontsize=13)

ax = axes[0]
ax.plot(history.history["accuracy"],     label="Train")
ax.plot(history.history["val_accuracy"], label="Val")
ax.set_title("Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history.history["loss"],     label="Train")
ax.plot(history.history["val_loss"], label="Val")
ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(HISTORY_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved training curves → {os.path.relpath(HISTORY_PNG)}")

# ---------------------------------------------------------------------------
# 7. Evaluate on test set
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("  Evaluating on test set ...")
print("-" * 65)

# Collect predictions
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Overall metrics
test_acc  = accuracy_score(y_true, y_pred)
test_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
test_rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
test_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\n  Test Accuracy  : {test_acc*100:.2f}%")
print(f"  Test Precision : {test_prec*100:.2f}%  (weighted)")
print(f"  Test Recall    : {test_rec*100:.2f}%  (weighted)")
print(f"  Test F1-score  : {test_f1*100:.2f}%  (weighted)")

# Classification report
short_names = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]
report = classification_report(
    y_true, y_pred,
    target_names=short_names,
    digits=4,
    zero_division=0,
)
print("\n" + report)

# Save report
with open(REPORT_TXT, "w") as f:
    f.write("T1 Tomato — MobileNetV2 Baseline: Test Set Results\n")
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
    y_true, y_pred,
    target_names=short_names,
    output_dict=True,
    zero_division=0,
)
metrics_rows = []
for cls in short_names:
    row = report_dict[cls]
    metrics_rows.append({
        "class":     cls,
        "precision": round(row["precision"], 4),
        "recall":    round(row["recall"],    4),
        "f1_score":  round(row["f1-score"],  4),
        "support":   int(row["support"]),
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"  Saved per-class metrics → {os.path.relpath(METRICS_CSV)}")

# ---------------------------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=short_names, columns=short_names)
cm_df.to_csv(CM_CSV)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title("T1 Tomato — Confusion Matrix (Test Set)", fontsize=12)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=7,
        )

plt.tight_layout()
plt.savefig(CM_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved confusion matrix  → {os.path.relpath(CM_PNG)}")

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
best_val_acc  = max(history.history["val_accuracy"])
best_val_loss = min(history.history["val_loss"])

print("\n" + "=" * 65)
print("PHASE 8 — BASELINE COMPLETE")
print("=" * 65)
print(f"  Model             : MobileNetV2 (frozen) + head")
print(f"  Total params      : {total_params:,}")
print(f"  Trainable params  : {trainable_params:,}")
print(f"  Epochs trained    : {epochs_run}")
print(f"  Training time     : {elapsed/60:.1f} min")
print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
print(f"  Best val loss     : {best_val_loss:.4f}")
print(f"  Test accuracy     : {test_acc*100:.2f}%")
print(f"  Test precision    : {test_prec*100:.2f}%  (weighted)")
print(f"  Test recall       : {test_rec*100:.2f}%  (weighted)")
print(f"  Test F1-score     : {test_f1*100:.2f}%  (weighted)")
print(f"\n  Saved model       : {os.path.relpath(MODEL_PATH)}")
print(f"  Results dir       : {os.path.relpath(RESULTS_DIR)}")
print("=" * 65)
