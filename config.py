"""
config.py — Flask application configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

    # ------------------------------------------------------------------ #
    # File uploads
    # ------------------------------------------------------------------ #
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024          # 10 MB hard limit
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

    # ------------------------------------------------------------------ #
    # SQLite database
    # ------------------------------------------------------------------ #
    INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
    DATABASE_PATH = os.path.join(INSTANCE_DIR, "predictions.db")

    # ------------------------------------------------------------------ #
    # Trained model directory (populated after AWS training)
    # ------------------------------------------------------------------ #
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # ------------------------------------------------------------------ #
    # Pagination
    # ------------------------------------------------------------------ #
    HISTORY_PER_PAGE = 20


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")   # must be set in env


# Active config — override with APP_ENV=production
_configs = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
}
ActiveConfig = _configs.get(os.environ.get("APP_ENV", "development"), DevelopmentConfig)
