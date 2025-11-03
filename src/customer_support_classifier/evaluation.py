"""Evaluation utilities for classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, average: str = "weighted") -> Dict[str, float]:
    """
    Generate classification metrics for a fitted model.

    Parameters
    ----------
    model:
        Trained scikit-learn model or pipeline.
    X_test, y_test:
        Evaluation features and labels.
    average:
        Averaging strategy for precision/recall/f1.

    Returns
    -------
    dict
        Computed metrics.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }
    return metrics


def save_confusion_matrix(
    model,
    X_test,
    y_test,
    output_path: str | Path,
    labels: list[str] | None = None,
) -> Path:
    """
    Create and persist a confusion matrix plot for the model predictions.
    """
    matrix = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_model(model, path: str | Path) -> Path:
    """Persist a trained model to disk using joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path

