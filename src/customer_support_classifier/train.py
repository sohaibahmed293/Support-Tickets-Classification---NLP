"""Command line interface for training support ticket classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from .config import load_config
from .data import load_ticket_data
from .evaluation import evaluate_model, save_confusion_matrix, save_model
from .modeling import build_model_pipelines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NLP models to classify customer support tickets."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        help="Optional model name to train. Trains all models if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="artifacts",
        help="Directory to store trained models and evaluation outputs.",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Disable grid search and train with the default hyperparameters.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})

    X, y = load_ticket_data(data_cfg)

    label_set = sorted(pd.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        stratify=y,
    )

    pipelines = build_model_pipelines(config)
    if args.model:
        if args.model not in pipelines:
            raise ValueError(f"Model '{args.model}' not defined in configuration.")
        pipelines = {args.model: pipelines[args.model]}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}

    for model_name, pipeline in pipelines.items():
        print(f"\nTraining model: {model_name}")
        estimator = pipeline
        best_params: Optional[dict] = None

        if not args.no_grid_search:
            grid_config = config.get("grid_search", {}).get(model_name)
            if grid_config:
                param_grid = {f"classifier__{key}": value for key, value in grid_config.items()}
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=5,
                    scoring="f1_weighted",
                    verbose=1,
                )
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
                best_params = search.best_params_
                print(f"Best params for {model_name}: {best_params}")
            else:
                estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train)

        metrics = evaluate_model(
            estimator,
            X_test,
            y_test,
            average=eval_cfg.get("average", "weighted"),
        )
        print(f"Metrics for {model_name}: {metrics}")

        model_path = save_model(estimator, output_dir / f"{model_name}.joblib")
        cm_path = save_confusion_matrix(
            estimator,
            X_test,
            y_test,
            output_dir / f"{model_name}_confusion_matrix.png",
            labels=label_set,
        )

        results[model_name] = {
            "metrics": metrics,
            "model_path": str(model_path),
            "confusion_matrix": str(cm_path),
        }
        if best_params:
            results[model_name]["best_params"] = best_params

    results_path = output_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nTraining artefacts saved to: {output_dir.resolve()}")
    print(results_path.read_text())


if __name__ == "__main__":
    main(parse_args())
