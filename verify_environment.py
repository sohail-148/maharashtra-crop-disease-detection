"""
Phase 7 — Environment Verification

Confirms all required ML libraries are importable and reports versions.
Also verifies TensorFlow can load MobileNetV2 with ImageNet weights.

Does NOT train a model.
"""

import sys

print("=" * 60)
print("PHASE 7 — ENVIRONMENT VERIFICATION")
print("=" * 60)

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
print(f"\n{'Python':<25} {sys.version.split()[0]}")

# ---------------------------------------------------------------------------
# Core libraries
# ---------------------------------------------------------------------------
results = []

def check(label, import_fn):
    try:
        version = import_fn()
        results.append((label, version, "OK"))
        print(f"  {label:<23} {version:<15} OK")
    except Exception as e:
        results.append((label, str(e), "FAIL"))
        print(f"  {label:<23} FAIL — {e}")

print(f"\n{'Library':<25} {'Version':<15} {'Status'}")
print("-" * 55)

def tf_version():
    import tensorflow as tf
    return tf.__version__

def keras_version():
    import keras
    return keras.__version__

def cv_version():
    import cv2
    return cv2.__version__

def sklearn_version():
    import sklearn
    return sklearn.__version__

def pandas_version():
    import pandas as pd
    return pd.__version__

def matplotlib_version():
    import matplotlib
    return matplotlib.__version__

def pillow_version():
    from PIL import Image
    import PIL
    return PIL.__version__

def numpy_version():
    import numpy as np
    return np.__version__

check("TensorFlow",   tf_version)
check("Keras",        keras_version)
check("OpenCV",       cv_version)
check("scikit-learn", sklearn_version)
check("pandas",       pandas_version)
check("matplotlib",   matplotlib_version)
check("Pillow",       pillow_version)
check("NumPy",        numpy_version)

# ---------------------------------------------------------------------------
# MobileNetV2 load test
# ---------------------------------------------------------------------------
print("\n" + "-" * 55)
print("MobileNetV2 LOAD TEST")
print("-" * 55)

try:
    import tensorflow as tf
    # Suppress TF info/warning logs for clean output
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("  Loading MobileNetV2 (ImageNet weights, include_top=False) ...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    params = base_model.count_params()
    print(f"  Status        : OK")
    print(f"  Input shape   : {base_model.input_shape}")
    print(f"  Output shape  : {base_model.output_shape}")
    print(f"  Parameters    : {params:,}")
    print(f"  Layers        : {len(base_model.layers)}")
    mobilenet_ok = True
except Exception as e:
    print(f"  FAIL — {e}")
    mobilenet_ok = False

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_ok = all(r[2] == "OK" for r in results) and mobilenet_ok

for label, version, status in results:
    mark = "✓" if status == "OK" else "✗"
    print(f"  {mark}  {label:<23} {version}")

print(f"  {'✓' if mobilenet_ok else '✗'}  MobileNetV2 load")

print()
if all_ok:
    print("  ALL CHECKS PASSED — environment is ready for Phase 8.")
else:
    print("  SOME CHECKS FAILED — review the output above.")

print("=" * 60)
