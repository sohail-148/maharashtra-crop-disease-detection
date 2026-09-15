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
        "model_file":   "experiments/chilli_field_experiment/candidate_d/model.keras",
        "alt_model_files": [
            "chilli_cold/chilli_cold_baseline.keras",
            "chilli/chilli_cold.keras",
        ],
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

_MODEL_CACHE: dict = {}


def resolve_model_path(models_dir: str, model_file: str) -> Optional[str]:
    """
    Resolve a model path from models_dir, project root, or absolute path.
    """
    if os.path.isabs(model_file) and os.path.exists(model_file):
        return os.path.normpath(model_file)
    # Check relative to models_dir (e.g. 'tomato/tomato_baseline.keras')
    p1 = os.path.normpath(os.path.join(models_dir, model_file))
    if os.path.exists(p1):
        return p1
    # Check relative to project root / parent of models_dir (e.g. 'experiments/...')
    project_root = os.path.dirname(os.path.abspath(models_dir))
    p2 = os.path.normpath(os.path.join(project_root, model_file))
    if os.path.exists(p2):
        return p2
    return None


def get_cached_model(model_path: str):
    """
    Retrieve or load a Keras model instance.
    Cached strictly by canonical absolute file path to avoid redundant disk I/O.
    """
    import tensorflow as tf

    canon_path = os.path.normcase(os.path.abspath(model_path))
    if canon_path not in _MODEL_CACHE:
        _MODEL_CACHE[canon_path] = tf.keras.models.load_model(canon_path)
    return _MODEL_CACHE[canon_path]


def clear_model_cache():
    """Clear all in-memory cached models."""
    _MODEL_CACHE.clear()


def load_and_preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Load an image from disk, orient via EXIF, convert to RGB, resize,
    apply MobileNetV2 preprocess_input, and return preprocessed batch tensor.

    Supports JPEG, PNG, BMP, WEBP, TIFF, etc.
    """
    from PIL import Image, ImageOps
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        batch = np.expand_dims(arr, axis=0)
        return tf.convert_to_tensor(batch, dtype=tf.float32)


class PredictionResult:
    """Plain data object returned by predict_image()."""

    def __init__(self, crop: str, experiment: str, disease: str,
                 confidence: float, all_probs: Optional[list],
                 is_placeholder: bool, message: str = "",
                 gradcam_path: Optional[str] = None,
                 is_error: bool = False,
                 model_path: Optional[str] = None):
        self.crop           = crop
        self.experiment     = experiment
        self.disease        = disease
        self.confidence     = confidence        # float 0–1
        self.all_probs      = all_probs         # list of (class, prob) or None
        self.is_placeholder = is_placeholder
        self.message        = message           # human-readable status
        self.gradcam_path   = gradcam_path      # relative path to static, e.g. "uploads/<uuid>_gradcam.jpg"
        self.is_error       = is_error
        self.model_path     = model_path

    def confidence_pct(self) -> str:
        """Return confidence as a percentage string, e.g. '93.4%'."""
        if self.is_placeholder or self.is_error:
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
            "is_error":       self.is_error,
            "message":        self.message,
            "gradcam_path":   self.gradcam_path,
            "model_path":     self.model_path,
        }


def predict_image(image_path: str, experiment: str,
                  models_dir: str, generate_gradcam: bool = True) -> PredictionResult:
    """
    Run inference on a single image file.

    Parameters
    ----------
    image_path       : absolute path to the uploaded image
    experiment       : experiment ID — one of T1, G1, G2, C1, S1, S2 (or crop names)
    models_dir       : absolute path to the models/ directory
    generate_gradcam : whether to generate a Grad-CAM explanation overlay (default True)

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
    model_path = resolve_model_path(models_dir, cfg["model_file"])
    if not model_path:
        for alt in cfg.get("alt_model_files", []):
            alt_path = resolve_model_path(models_dir, alt)
            if alt_path:
                model_path = alt_path
                break

    if not model_path or not os.path.exists(model_path):
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

        # Retrieve model from cache (keyed by canonical absolute path)
        model = get_cached_model(model_path)

        # Preprocess with robust format and EXIF handling (JPEG, PNG, BMP, WEBP)
        batch = load_and_preprocess_image(image_path, target_size=IMG_SIZE)

        # Predict
        probs      = model.predict(batch, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        disease    = cfg["classes"][pred_idx]

        all_probs = sorted(
            [(cls, float(p)) for cls, p in zip(cfg["classes"], probs)],
            key=lambda x: x[1], reverse=True,
        )

        # --------------------------------------------------------------
        # Grad-CAM Visual Explanation (Additive & Exception-Isolated)
        # --------------------------------------------------------------
        gradcam_path = None
        if generate_gradcam:
            try:
                from app.gradcam import generate_gradcam_overlay

                # Derive output path next to image_path: <stem>_gradcam.jpg
                img_dir = os.path.dirname(image_path)
                stem = os.path.splitext(os.path.basename(image_path))[0]
                gradcam_filename = f"{stem}_gradcam.jpg"
                gradcam_abs = os.path.join(img_dir, gradcam_filename)

                generate_gradcam_overlay(
                    model=model,
                    image_path=image_path,
                    target_class_idx=pred_idx,
                    output_path=gradcam_abs,
                    tensor=batch,
                )

                # Relative path from static/ (e.g. "uploads/<uuid>_gradcam.jpg")
                if "static" in img_dir:
                    rel_prefix = img_dir.split("static" + os.sep)[-1].replace(os.sep, "/")
                    gradcam_path = f"{rel_prefix}/{gradcam_filename}"
                else:
                    gradcam_path = f"uploads/{gradcam_filename}"

            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    f"Grad-CAM explanation generation failed for {image_path}: {exc}",
                    exc_info=True,
                )
                gradcam_path = None

        return PredictionResult(
            crop=crop_name, experiment=exp_key,
            disease=disease, confidence=confidence,
            all_probs=all_probs, is_placeholder=False,
            is_error=False,
            message="",
            gradcam_path=gradcam_path,
            model_path=model_path,
        )

    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error(
            f"Image analysis failed for {image_path}: {exc}", exc_info=True
        )
        return PredictionResult(
            crop=crop_name, experiment=exp_key,
            disease="Unable to analyze image",
            confidence=0.0, all_probs=None,
            is_placeholder=False,
            is_error=True,
            message=(
                "The provided file could not be decoded or processed as a valid image. "
                "Please ensure the file is an uncorrupted JPG, PNG, BMP, or WEBP image."
            ),
            gradcam_path=None,
        )


def list_available_models(models_dir: str) -> dict:
    """Return a dict of crop_id -> bool (model file present)."""
    status = {}
    for crop_id, cfg in CROPS.items():
        path = resolve_model_path(models_dir, cfg["model_file"])
        if not path:
            for alt in cfg.get("alt_model_files", []):
                path = resolve_model_path(models_dir, alt)
                if path:
                    break
        status[crop_id] = (path is not None)
    return status
