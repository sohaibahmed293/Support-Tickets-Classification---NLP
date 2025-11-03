"""Flask web application for interactive ticket classification."""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
from flask import Flask, flash, g, redirect, render_template, request, url_for, current_app

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def ensure_database(path: Path) -> None:
    """Create the SQLite database and predictions table if they do not yet exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.commit()


def get_db() -> sqlite3.Connection:
    """Return a request-scoped SQLite connection."""
    if "db" not in g:
        db_path = Path(current_app.config["DATABASE"])
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(exception: Optional[BaseException] = None) -> None:
    """Close the request-scoped SQLite connection."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def available_models(model_dir: Path) -> List[Path]:
    """Return the list of trained model artefacts found in the model directory."""
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.joblib"))


@lru_cache(maxsize=8)
def load_model(model_path: str):
    """Load and cache a trained model pipeline."""
    return joblib.load(model_path)


def predict_ticket(
    model,
    ticket_text: str,
) -> Tuple[str, Optional[List[Tuple[str, float]]]]:
    """Run inference and return the top prediction and optional probability pairs."""
    prediction = model.predict([ticket_text])[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([ticket_text])[0]
        class_labels: Iterable[str] = getattr(model, "classes_", [])
        ranked = sorted(
            zip(class_labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return prediction, list(ranked)
    return prediction, None


def fetch_recent_predictions(limit: int = 10) -> List[sqlite3.Row]:
    """Retrieve the most recent prediction rows."""
    db = get_db()
    cursor = db.execute(
        """
        SELECT id, ticket_text, predicted_label, confidence, model_name, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cursor.fetchall()


def create_app(test_config: Optional[dict] = None) -> Flask:
    """Application factory used by Flask tooling and WSGI servers."""
    app = Flask(__name__, template_folder="templates")
    default_database = ROOT_DIR / "webapp" / "predictions.db"
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev"),
        DATABASE=os.environ.get("FLASK_DATABASE", str(default_database)),
        MODEL_DIR=os.environ.get("MODEL_ARTEFACT_DIR", str(ROOT_DIR / "artifacts")),
    )

    if test_config:
        app.config.update(test_config)

    ensure_database(Path(app.config["DATABASE"]))

    app.teardown_appcontext(close_db)

    @app.route("/", methods=["GET", "POST"])
    def index():
        model_dir = Path(app.config["MODEL_DIR"])
        models = available_models(model_dir)
        selected_model = request.form.get("model_path") if request.method == "POST" else None
        prediction_result = None
        probabilities = None

        if request.method == "POST":
            ticket_text = (request.form.get("ticket_text") or "").strip()
            model_path = selected_model or (str(models[0]) if models else "")

            if not ticket_text:
                flash("Please provide a support ticket description to classify.", "error")
            elif not model_path:
                flash(
                    "No trained models found. Train a model and place the joblib artefact in the "
                    "artifacts directory.",
                    "error",
                )
            else:
                model = load_model(model_path)
                predicted_label, prob_pairs = predict_ticket(model, ticket_text)
                top_confidence = prob_pairs[0][1] if prob_pairs else None

                db = get_db()
                db.execute(
                    """
                    INSERT INTO predictions (ticket_text, predicted_label, confidence, model_name, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ticket_text,
                        predicted_label,
                        float(top_confidence) if top_confidence is not None else None,
                        Path(model_path).name,
                        datetime.utcnow().isoformat(),
                    ),
                )
                db.commit()

                prediction_result = {
                    "ticket_text": ticket_text,
                    "predicted_label": predicted_label,
                    "model_name": Path(model_path).name,
                    "confidence": top_confidence,
                }
                probabilities = prob_pairs[:5] if prob_pairs else None
                flash("Prediction completed successfully.", "success")
                selected_model = model_path

        recent_predictions = fetch_recent_predictions()
        return render_template(
            "index.html",
            models=models,
            selected_model=selected_model,
            prediction=prediction_result,
            probabilities=probabilities,
            history=recent_predictions,
        )

    @app.post("/clear")
    def clear_history():
        db = get_db()
        db.execute("DELETE FROM predictions")
        db.commit()
        flash("Prediction history cleared.", "success")
        return redirect(url_for("index"))

    return app


app = create_app()
