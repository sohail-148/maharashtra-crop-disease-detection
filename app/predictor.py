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
# Crop / experiment registry
# Matches the 6 experiments from the ML training pipeline.
# ---------------------------------------------------------------------------

CROPS = {
    "T1": {
        "crop":    "Tomato",
        "classes": [
            "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Spider Mites", "Target Spot",
            "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy",
        ],
        "model_file": "tomato/tomato_baseline.keras",
    },
    "G1": {
        "crop":    "Grape (Niphad)",
        "classes": ["Bacterial Leaf Spot", "Downy Mildew",
                    "Healthy Leaves", "Powdery Mildew"],
        "model_file": "grape_niphad/grape_niphad_baseline.keras",
    },
    "G2": {
        "crop":    "Grape (2024)",
        "classes": ["Black Rot", "Esca", "Healthy", "Leaf Blight"],
        "model_file": "grape_2024/grape_2024_baseline.keras",
    },
    "C1": {
        "crop":    "Chilli",
        "classes": ["Cerocospora", "Healthy", "Murda Complex",
                    "Nutritional Deficiency", "Powdery Mildew"],
        "model_file": "chilli_cold/chilli_cold_baseline.keras",
    },
    "S1": {
        "crop":    "Sugarcane (Maharashtra)",
        "classes": ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"],
        "model_file": "sugarcane_maharashtra/sugarcane_maharashtra_baseline.keras",
    },
    "S2": {
        "crop":    "Sugarcane (Large)",
        "classes": [
            "Banded Chlorosis", "Brown Spot", "Brown Rust", "Grassy Shoot",
            "Healthy Leaves", "Pokkah Boeng", "Sett Rot", "Smut",
            "Viral Disease", "Yellow Leaf",
        ],
        "model_file": "sugarcane_large/sugarcane_large_baseline.keras",
    },
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
    if experiment not in CROPS:
        return PredictionResult(
            crop="Unknown", experiment=experiment,
            disease="Unknown", confidence=0.0,
            all_probs=None, is_placeholder=True,
            message=f"Unknown experiment ID: {experiment}",
        )

    cfg        = CROPS[experiment]
    crop_name  = cfg["crop"]
    model_path = os.path.join(models_dir, cfg["model_file"])

    # ------------------------------------------------------------------
    # Check whether a trained model file exists
    # ------------------------------------------------------------------
    if not os.path.exists(model_path):
        return PredictionResult(
            crop=crop_name, experiment=experiment,
            disease="Model not yet available",
            confidence=0.0, all_probs=None,
            is_placeholder=True,
            message=(
                f"Model file not found: {cfg['model_file']}. "
                "Training on AWS GPU is pending. "
                "Upload the trained .keras file to activate predictions."
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
            crop=crop_name, experiment=experiment,
            disease=disease, confidence=confidence,
            all_probs=all_probs, is_placeholder=False,
            message="",
        )

    except Exception as exc:  # noqa: BLE001
        return PredictionResult(
            crop=crop_name, experiment=experiment,
            disease="Inference error",
            confidence=0.0, all_probs=None,
            is_placeholder=True,
            message=f"Inference failed: {exc}",
        )


def list_available_models(models_dir: str) -> dict:
    """Return a dict of experiment_id -> bool (model file present)."""
    return {
        exp_id: os.path.exists(os.path.join(models_dir, cfg["model_file"]))
        for exp_id, cfg in CROPS.items()
    }
