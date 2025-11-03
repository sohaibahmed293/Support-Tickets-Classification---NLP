"""Model building utilities."""

from __future__ import annotations

from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import create_tfidf_vectorizer


def build_model_pipelines(config: dict) -> Dict[str, Pipeline]:
    """
    Construct ML pipelines for the configured models.

    Parameters
    ----------
    config:
        Full configuration dictionary.

    Returns
    -------
    dict
        Mapping of model name to scikit-learn Pipeline.
    """
    preprocess_config = config.get("preprocessing", {})
    models_config = config.get("models", {})

    pipelines: Dict[str, Pipeline] = {}

    if "logistic_regression" in models_config:
        lr_params = models_config["logistic_regression"]
        pipelines["logistic_regression"] = Pipeline(
            steps=[
                ("vectorizer", create_tfidf_vectorizer(preprocess_config)),
                (
                    "classifier",
                    LogisticRegression(
                        C=lr_params.get("C", 1.0),
                        max_iter=lr_params.get("max_iter", 200),
                        class_weight=lr_params.get("class_weight"),
                        n_jobs=lr_params.get("n_jobs"),
                        solver="lbfgs",
                        multi_class="auto",
                    ),
                ),
            ]
        )

    if "random_forest" in models_config:
        rf_params = models_config["random_forest"]
        pipelines["random_forest"] = Pipeline(
            steps=[
                ("vectorizer", create_tfidf_vectorizer(preprocess_config)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=rf_params.get("n_estimators", 200),
                        max_depth=rf_params.get("max_depth"),
                        class_weight=rf_params.get("class_weight"),
                        n_jobs=rf_params.get("n_jobs", -1),
                    ),
                ),
            ]
        )

    if not pipelines:
        raise ValueError("No model pipelines were constructed. Check your configuration.")

    return pipelines
