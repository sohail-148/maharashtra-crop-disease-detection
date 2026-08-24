"""
app/__init__.py — Flask application factory
"""
import os
from flask import Flask
from config import ActiveConfig


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(config or ActiveConfig)

    # ------------------------------------------------------------------ #
    # Ensure required directories exist
    # ------------------------------------------------------------------ #
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INSTANCE_DIR"],  exist_ok=True)

    # ------------------------------------------------------------------ #
    # Initialise database
    # ------------------------------------------------------------------ #
    from app.database import init_db
    init_db(app.config["DATABASE_PATH"])

    # ------------------------------------------------------------------ #
    # Register blueprints / routes
    # ------------------------------------------------------------------ #
    from app.routes import main
    app.register_blueprint(main)

    return app
