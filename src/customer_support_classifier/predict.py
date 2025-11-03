"""Utility script to run predictions with a trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict support ticket categories using a trained model."
    )
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        help="Path to the trained model artefact (joblib file).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        "-t",
        nargs="+",
        help="Ticket text snippets to classify.",
    )
    group.add_argument(
        "--input-file",
        "-i",
        help="Optional path to a text file. One ticket per line.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save predictions as JSON.",
    )
    return parser.parse_args()


def load_tickets_from_file(path: str | Path) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main(args: argparse.Namespace) -> None:
    model = joblib.load(args.model_path)

    if args.text:
        texts = list(args.text)
    else:
        texts = load_tickets_from_file(args.input_file)

    predictions = model.predict(texts)
    output = [{"text": text, "prediction": prediction} for text, prediction in zip(texts, predictions)]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        print(f"Predictions saved to {output_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(parse_args())
