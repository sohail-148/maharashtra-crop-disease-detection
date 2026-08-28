"""
app/predictor.py — Prediction service layer

This module is the ONLY place in the application that knows about ML models.
Routes never import TensorFlow directly; they call predict_image() here.

Current state
-------------
No trained models are available yet (AWS training in progress).
predict_image() returns a clearly-labelled placeholder result so the
entire web application can be developed, tested, and demonstrated
without a real model.

When models become available
-----------------------------
1. Drop each <experiment>_baseline.keras file into models/<name>/
2. Update the MODELS dict below with the correct path.
3. Uncomment the TensorFlow block and remove the placeholder block.
4. No other file in the application needs to change.

Placeholder contract
--------------------
- is_placeholder = True is ALWAYS set when no real model is used.
- The UI displays a clear "Model not yet available" banner when this is True.
- Confidence is returned as 0.0 — no fabricated probability is shown.
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Crop / model registry
# 4-Crop Production Architecture:
#   TOMATO    -> Tomato (T1 baseline model, 10 classes)
#   GRAPE     -> Grape Unified (G1+G2 model, 7 classes)
#   CHILLI    -> Chilli (C1 model, 5 classes)
#   SUGARCANE -> Sugarcane Unified (S1+S2 model, 11 classes)
# ---------------------------------------------------------------------------

CROPS = {
    "TOMATO": {
        "crop":         "Tomato",
        "display_code": "10 Classes",
        "classes": [
            "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Spider Mites", "Target Spot",
            "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy",
        ],
        "model_file":   "tomato/tomato_baseline.keras",
        "alt_model_files": [],
    },
    "GRAPE": {
        "crop":         "Grape",
        "display_code": "7 Classes",
        "classes": [
            "Bacterial Leaf Spot", "Black Rot", "Downy Mildew",
            "Esca (Black Measles)", "Healthy Leaves", "Leaf Blight",
            "Powdery Mildew",
        ],
        "model_file":   "grape_unified/grape_unified_baseline.keras",
        "alt_model_files": ["grape/grape_unified.keras"],
    },
    "CHILLI": {
        "crop":         "Chilli",
        "display_code": "5 Classes",
        "classes": [
            "Cercospora Leaf Spot", "Healthy", "Murda Complex (Leaf Curl)",
            "Nutritional Deficiency", "Powdery Mildew",
        ],
        "model_file":   "chilli_cold/chilli_cold_baseline.keras",
        "alt_model_files": ["chilli/chilli_cold.keras"],
    },
    "SUGARCANE": {
        "crop":         "Sugarcane",
        "display_code": "11 Classes",
        "classes": [
            "Banded Chlorosis", "Brown Spot", "Grassy Shoot", "Healthy Leaves",
            "Mosaic / Viral Disease", "Pokkah Boeng", "Red Rot",
            "Rust (Brown Rust)", "Sett Rot", "Smut", "Yellow Leaf Disease",
        ],
        "model_file":   "sugarcane_unified/sugarcane_unified_baseline.keras",
        "alt_model_files": ["sugarcane/sugarcane_unified.keras"],
    },
}

# Aliases for backwards compatibility with legacy experiment IDs
CROP_ALIASES = {
    "T1": "TOMATO",
    "G1": "GRAPE",
    "G2": "GRAPE",
    "C1": "CHILLI",
    "S1": "SUGARCANE",
    "S2": "SUGARCANE",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class PredictionResult:
    """Plain data object returned by predict_image()."""

    def __init__(self, crop: str, experiment: str, disease: str,
                 confidence: float, all_probs: Optional[list],
                 is_placeholder: bool, message: str = ""):
        self.crop           = crop
        self.experiment     = experiment
        self.disease        = disease
        self.confidence     = confidence        # float 0–1
        self.all_probs      = all_probs         # list of (class, prob) or None
        self.is_placeholder = is_placeholder
        self.message        = message           # human-readable status

    def confidence_pct(self) -> str:
        """Return confidence as a percentage string, e.g. '93.4%'."""
        if self.is_placeholder:
            return "N/A"
        return f"{self.confidence * 100:.1f}%"

    def to_dict(self) -> dict:
        return {
            "crop":           self.crop,
            "experiment":     self.experiment,
            "disease":        self.disease,
            "confidence":     self.confidence,
            "confidence_pct": self.confidence_pct(),
            "all_probs":      self.all_probs,
            "is_placeholder": self.is_placeholder,
            "message":        self.message,
        }


def predict_image(image_path: str, experiment: str,
                  models_dir: str) -> PredictionResult:
    """
    Run inference on a single image file.

    Parameters
    ----------
    image_path   : absolute path to the uploaded image
    experiment   : experiment ID — one of T1, G1, G2, C1, S1, S2
    models_dir   : absolute path to the models/ directory

    Returns
    -------
    PredictionResult
    """
    if experiment is None:
        experiment = ""
    exp_key = CROP_ALIASES.get(experiment.upper(), experiment.upper())

    if exp_key not in CROPS:
        return PredictionResult(
            crop="Unknown", experiment=experiment,
            disease="Unknown", confidence=0.0,
            all_probs=None, is_placeholder=True,
            message=f"Unknown crop/experiment ID: {experiment}",
        )

    cfg        = CROPS[exp_key]
    crop_name  = cfg["crop"]

    # ------------------------------------------------------------------
    # Check whether a trained model file exists (primary or alt paths)
    # ------------------------------------------------------------------
    model_path = os.path.join(models_dir, cfg["model_file"])
    if not os.path.exists(model_path):
        for alt in cfg.get("alt_model_files", []):
            alt_path = os.path.join(models_dir, alt)
            if os.path.exists(alt_path):
                model_path = alt_path
                break

    if not os.path.exists(model_path):
        return PredictionResult(
            crop=crop_name, experiment=exp_key,
            disease="Model not yet available",
            confidence=0.0, all_probs=None,
            is_placeholder=True,
            message=(
                f"Model file not found: {cfg['model_file']}. "
                "GPU training is pending. "
                "Train on Kaggle/Colab GPU or place the trained .keras file in models/ to activate predictions."
            ),
        )

    # ------------------------------------------------------------------
    # Real inference — only reached when the model file exists.
    # ------------------------------------------------------------------
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        IMG_SIZE = (224, 224)

        # Load model (cached by TF's internal mechanism after first load)
        model = tf.keras.models.load_model(model_path)

        # Preprocess
        raw   = tf.io.read_file(image_path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image = preprocess_input(image)
        batch = tf.expand_dims(image, axis=0)

        # Predict
        probs      = model.predict(batch, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        disease    = cfg["classes"][pred_idx]

        all_probs = sorted(
            [(cls, float(p)) for cls, p in zip(cfg["classes"], probs)],
            key=lambda x: x[1], reverse=True,
        )

        return PredictionResult(
            crop=crop_name, experiment=exp_key,
            disease=disease, confidence=confidence,
            all_probs=all_probs, is_placeholder=False,
            message="",
        )

    except Exception as exc:  # noqa: BLE001
        return PredictionResult(
            crop=crop_name, experiment=exp_key,
            disease="Inference error",
            confidence=0.0, all_probs=None,
            is_placeholder=True,
            message=f"Inference failed: {exc}",
        )


def list_available_models(models_dir: str) -> dict:
    """Return a dict of crop_id -> bool (model file present)."""
    status = {}
    for crop_id, cfg in CROPS.items():
        present = os.path.exists(os.path.join(models_dir, cfg["model_file"]))
        if not present:
            for alt in cfg.get("alt_model_files", []):
                if os.path.exists(os.path.join(models_dir, alt)):
                    present = True
                    break
        status[crop_id] = present
    return status
