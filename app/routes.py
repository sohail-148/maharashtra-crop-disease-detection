"""
app/routes.py — All Flask route handlers

Blueprint: main

Routes
------
GET  /                      index  — crop selection + image input
POST /predict               predict — process image, run predictor, save to DB
GET  /history               history — paginated prediction history
GET  /about                 about
POST /predictions/<id>/delete  delete a single prediction record
GET  /api/status            JSON — model availability status
"""

import os
import uuid
import math
from datetime import datetime, timezone

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, jsonify, abort,
)
from werkzeug.utils import secure_filename

from app.predictor import CROPS, CROP_ALIASES, predict_image, list_available_models
from app.database  import (
    save_prediction, get_recent_predictions,
    get_prediction_by_id, delete_prediction,
)

main = Blueprint("main", __name__)

# ---------------------------------------------------------------------------
# Icons for each crop (used in templates)
# ---------------------------------------------------------------------------
CROP_ICONS = {
    "TOMATO":    "🍅",
    "GRAPE":     "🍇",
    "CHILLI":    "🌶️",
    "SUGARCANE": "🌾",
    # Legacy aliases
    "T1": "🍅", "G1": "🍇", "G2": "🍇", "C1": "🌶️", "S1": "🌾", "S2": "🌾",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def save_upload(file_storage) -> str:
    """
    Save an uploaded FileStorage object to UPLOAD_FOLDER.
    Returns the relative path from static/ for use in templates.
    """
    original   = secure_filename(file_storage.filename)
    ext        = original.rsplit(".", 1)[-1].lower() if "." in original else "jpg"
    unique_name= f"{uuid.uuid4().hex}.{ext}"
    abs_path   = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file_storage.save(abs_path)
    return f"uploads/{unique_name}"


def save_camera_image(data_url: str) -> str:
    """
    Decode a base64 data URL (from camera capture) and save as JPEG.
    Returns the relative path from static/.
    """
    import base64
    # Strip the data:image/...;base64, header
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes  = base64.b64decode(data_url)
    unique_name= f"{uuid.uuid4().hex}.jpg"
    abs_path   = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    with open(abs_path, "wb") as f:
        f.write(img_bytes)
    return f"uploads/{unique_name}"


def _template_context() -> dict:
    """Common context injected into every render_template call."""
    return {
        "crops":        CROPS,
        "crop_icons":   CROP_ICONS,
        "model_status": list_available_models(current_app.config["MODELS_DIR"]),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@main.route("/")
def index():
    return render_template("index.html", **_template_context())


@main.route("/predict", methods=["POST"])
def predict():
    raw_exp    = request.form.get("experiment", "").strip().upper()
    experiment = CROP_ALIASES.get(raw_exp, raw_exp)
    if experiment not in CROPS:
        flash("Please select a valid crop.", "error")
        return redirect(url_for("main.index"))

    # ----------------------------------------------------------------
    # Determine image source: file upload or camera capture
    # ----------------------------------------------------------------
    rel_path   = None   # relative path from static/
    abs_path   = None   # absolute path for predictor

    uploaded_file = request.files.get("image")
    camera_data   = request.form.get("camera_image", "").strip()

    if uploaded_file and uploaded_file.filename:
        if not allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload JPG, PNG, BMP, or WEBP.", "error")
            return redirect(url_for("main.index"))
        # Size check (Werkzeug enforces MAX_CONTENT_LENGTH globally,
        # but we do an explicit check here for a cleaner message)
        uploaded_file.seek(0, 2)   # seek to end
        size = uploaded_file.tell()
        uploaded_file.seek(0)      # reset
        if size > current_app.config["MAX_CONTENT_LENGTH"]:
            flash("File is too large (maximum 10 MB).", "error")
            return redirect(url_for("main.index"))

        rel_path = save_upload(uploaded_file)

    elif camera_data:
        try:
            rel_path = save_camera_image(camera_data)
        except Exception:
            flash("Could not process camera image. Please try again.", "error")
            return redirect(url_for("main.index"))

    else:
        flash("No image provided. Please upload or capture a leaf image.", "error")
        return redirect(url_for("main.index"))

    abs_path = os.path.join(
        current_app.root_path, "static", rel_path.replace("/", os.sep)
    )

    # ----------------------------------------------------------------
    # Run prediction
    # ----------------------------------------------------------------
    result = predict_image(
        image_path   = abs_path,
        experiment   = experiment,
        models_dir   = current_app.config["MODELS_DIR"],
    )

    # ----------------------------------------------------------------
    # Persist to SQLite
    # ----------------------------------------------------------------
    pred_id = save_prediction(
        db_path        = current_app.config["DATABASE_PATH"],
        crop           = result.crop,
        experiment     = result.experiment,
        disease        = result.disease,
        confidence     = result.confidence,
        image_path     = rel_path,
        is_placeholder = result.is_placeholder,
    )

    return render_template(
        "result.html",
        result        = result,
        image_url     = rel_path,
        prediction_id = pred_id,
        **_template_context(),
    )


@main.route("/history")
def history():
    per_page = current_app.config["HISTORY_PER_PAGE"]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    rows, total = get_recent_predictions(
        db_path  = current_app.config["DATABASE_PATH"],
        page     = page,
        per_page = per_page,
    )
    total_pages = math.ceil(total / per_page) if total else 1

    return render_template(
        "history.html",
        rows        = rows,
        total       = total,
        page        = page,
        total_pages = total_pages,
        **_template_context(),
    )


@main.route("/about")
def about():
    return render_template("about.html", **_template_context())


@main.route("/predictions/<int:pred_id>/delete", methods=["POST"], endpoint="delete_prediction")
def delete_prediction_route(pred_id):
    record = get_prediction_by_id(current_app.config["DATABASE_PATH"], pred_id)
    if record and record["image_path"]:
        img_abs = os.path.join(
            current_app.root_path, "static", record["image_path"].replace("/", os.sep)
        )
        if os.path.exists(img_abs):
            try:
                os.remove(img_abs)
            except OSError:
                pass

    deleted = delete_prediction(
        db_path = current_app.config["DATABASE_PATH"],
        pred_id = pred_id,
    )
    if deleted:
        flash(f"Prediction #{pred_id} deleted.", "success")
    else:
        flash(f"Prediction #{pred_id} not found.", "error")

    # Return to history unless the referer was the result page
    referer = request.referrer or ""
    if "history" in referer:
        return redirect(url_for("main.history"))
    return redirect(url_for("main.index"))


# ---------------------------------------------------------------------------
# API endpoint — model status (useful for polling from JS if needed)
# ---------------------------------------------------------------------------

@main.route("/api/status")
def api_status():
    status = list_available_models(current_app.config["MODELS_DIR"])
    return jsonify({
        "models":    status,
        "any_ready": any(status.values()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@main.app_errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        code=404,
        title="Page Not Found",
        description="The page you requested does not exist.",
    ), 404


@main.app_errorhandler(413)
def too_large(e):
    return render_template(
        "error.html",
        code=413,
        title="File Too Large",
        description="The uploaded file exceeds the 10 MB limit.",
    ), 413


@main.app_errorhandler(500)
def server_error(e):
    return render_template(
        "error.html",
        code=500,
        title="Server Error",
        description="Something went wrong on our end. Please try again.",
    ), 500
