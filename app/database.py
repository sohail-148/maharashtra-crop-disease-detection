"""
app/database.py — SQLite schema and helper functions

Schema
------
predictions
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    created_at  TEXT     NOT NULL   -- ISO-8601 UTC  e.g. "2026-08-24T10:30:00"
    crop        TEXT     NOT NULL   -- e.g. "Tomato"
    experiment  TEXT     NOT NULL   -- model used,  e.g. "T1"
    disease     TEXT     NOT NULL   -- e.g. "Early Blight"
    confidence  REAL     NOT NULL   -- 0.0 – 1.0
    image_path  TEXT     NOT NULL   -- relative path under static/uploads/
    is_placeholder  INTEGER DEFAULT 0  -- 1 when no trained model was available
"""

import sqlite3
import contextlib
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL,
    crop           TEXT    NOT NULL,
    experiment     TEXT    NOT NULL,
    disease        TEXT    NOT NULL,
    confidence     REAL    NOT NULL,
    image_path     TEXT    NOT NULL,
    is_placeholder INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Create the database and table if they do not already exist."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def get_db(db_path: str):
    """Context manager that yields a sqlite3.Connection and commits/closes."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_prediction(db_path: str, crop: str, experiment: str,
                    disease: str, confidence: float,
                    image_path: str, is_placeholder: bool = False) -> int:
    """
    Insert a prediction record and return its new id.

    Parameters
    ----------
    db_path      : absolute path to the SQLite file
    crop         : human-readable crop name, e.g. "Tomato"
    experiment   : experiment ID, e.g. "T1"
    disease      : predicted disease label
    confidence   : float in [0, 1]
    image_path   : path stored for display (relative to static/)
    is_placeholder : True when no real model was used
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with get_db(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (created_at, crop, experiment, disease, confidence,
                 image_path, is_placeholder)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, crop, experiment, disease, round(float(confidence), 4),
             image_path, int(is_placeholder)),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_recent_predictions(db_path: str, page: int = 1,
                            per_page: int = 20) -> tuple:
    """
    Return (rows, total_count) for paginated history display.
    Rows are ordered newest-first.
    """
    offset = (page - 1) * per_page
    with get_db(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, created_at, crop, experiment, disease,
                   confidence, image_path, is_placeholder
            FROM   predictions
            ORDER  BY id DESC
            LIMIT  ? OFFSET ?
            """,
            (per_page, offset),
        ).fetchall()
    return rows, total


def get_prediction_by_id(db_path: str, pred_id: int):
    """Return a single prediction row or None."""
    with get_db(db_path) as conn:
        return conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()


def delete_prediction(db_path: str, pred_id: int) -> bool:
    """Delete a record. Returns True if a row was removed."""
    with get_db(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM predictions WHERE id = ?", (pred_id,)
        )
        return cur.rowcount > 0
