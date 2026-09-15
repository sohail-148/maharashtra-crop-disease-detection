"""
app/gradcam.py — Runtime Grad-CAM Explainability Generation for Flask Application
Maharashtra Crop Disease Detection Project

Generates visual model attention heatmaps for user-uploaded leaf images
using the locked production MobileNetV2 models.

Implementation follows the validated non-symbolic GradientTape methodology:
- Passes input through model.layers[1] (MobileNetV2 base) to get 7x7x1280 features
- Watches features with tf.GradientTape
- Passes through GAP -> BatchNorm -> Dropout -> Dense
- Computes gradients of the pre-softmax predicted class logit w.r.t. spatial features
- Global average pools gradients to compute channel importance weights
- ReLU activates the weighted combination
- Normalizes and bilinearly upsamples to original image dimensions
- Synthesizes overlay with original PIL image using matplotlib 'jet' colormap
"""

import os
import logging
from typing import Optional
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

logger = logging.getLogger(__name__)


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    tensor: tf.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    """
    Compute a normalized 7x7 Grad-CAM activation heatmap for a target class.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded production model with 6 layers:
        [0] InputLayer
        [1] MobileNetV2 base (Functional, (None, 7, 7, 1280))
        [2] GlobalAveragePooling2D
        [3] BatchNormalization
        [4] Dropout
        [5] Dense
    tensor : tf.Tensor
        Preprocessed input tensor of shape (1, 224, 224, 3).
    target_class_idx : int
        Index of the target class (typically predicted class).

    Returns
    -------
    np.ndarray
        2D float32 array of shape (7, 7) with values in [0, 1].
    """
    if len(model.layers) < 6:
        raise ValueError(f"Expected model with at least 6 layers, got {len(model.layers)}")

    base_model = model.layers[1]
    gap_layer = model.layers[2]
    bn_layer = model.layers[3]
    dropout_layer = model.layers[4]
    dense_layer = model.layers[5]

    if not (0 <= target_class_idx < dense_layer.units):
        raise ValueError(
            f"target_class_idx {target_class_idx} out of range for model with {dense_layer.units} classes"
        )

    with tf.GradientTape() as tape:
        conv_features = base_model(tensor, training=False)
        tape.watch(conv_features)

        x = gap_layer(conv_features)
        x = bn_layer(x, training=False)
        x = dropout_layer(x, training=False)

        # Pre-softmax class logits to avoid vanishing gradients on confident predictions
        logits = tf.matmul(x, dense_layer.kernel) + dense_layer.bias
        loss = logits[:, target_class_idx]

    grads = tape.gradient(loss, conv_features)
    if grads is None:
        raise RuntimeError("Gradient computation returned None")

    # Global average pooling of gradients along spatial dimensions (H, W) -> shape (1, 1280)
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_features * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
    cam = tf.nn.relu(cam)

    # Normalize to [0, 1]
    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    return cam[0].numpy()


def generate_gradcam_overlay(
    model: tf.keras.Model,
    image_path: str,
    target_class_idx: int,
    output_path: str,
    tensor: Optional[tf.Tensor] = None,
    alpha: float = 0.45,
) -> str:
    """
    Generate and save a Grad-CAM heatmap overlay for the target class.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded production Keras model.
    image_path : str
        Absolute path to the input image file.
    target_class_idx : int
        Target class index (predicted class).
    output_path : str
        Absolute path to save the resulting JPEG overlay.
    tensor : Optional[tf.Tensor]
        Optional precomputed preprocessed tensor (1, 224, 224, 3).
        If None, will be preprocessed from image_path.
    alpha : float, default=0.45
        Colormap blending ratio (0.0 = original, 1.0 = heatmap only).

    Returns
    -------
    str : output_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 1. Obtain preprocessed tensor
    if tensor is None:
        from PIL import ImageOps
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im_resized = im.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(im_resized, dtype=np.float32)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            tensor = tf.convert_to_tensor(np.expand_dims(arr, axis=0))

    # 2. Compute 7x7 Grad-CAM heatmap
    heatmap_7x7 = compute_gradcam_heatmap(model, tensor, target_class_idx)

    # 3. Load original image to match exact original dimensions (respecting EXIF orientation)
    from PIL import ImageOps
    with Image.open(image_path) as raw_pil:
        pil_img = ImageOps.exif_transpose(raw_pil).convert("RGB")
        orig_w, orig_h = pil_img.size

        # 4. Bilinear upsampling of heatmap to original image dimensions
        heatmap_img = Image.fromarray(heatmap_7x7)
        heatmap_resized = np.array(heatmap_img.resize((orig_w, orig_h), resample=Image.BILINEAR))

        # 5. Apply colormap ('jet')
        cmap = matplotlib.colormaps["jet"]
        colored_cam = cmap(heatmap_resized)[:, :, :3]

        # 6. Alpha blend with original image
        orig_arr = np.array(pil_img, dtype=np.float32) / 255.0
        overlay = alpha * colored_cam + (1.0 - alpha) * orig_arr
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

        # 7. Save JPEG overlay
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(output_path, format="JPEG", quality=92)

    return output_path

