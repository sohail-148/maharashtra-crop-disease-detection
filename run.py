"""
run.py — Application entry point

Development:
    python run.py

Production (gunicorn example):
    gunicorn "run:app" --bind 0.0.0.0:5000 --workers 2
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
