"""Data access utilities for the customer support ticket classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_json(path: Path, record_key: str | None = None) -> pd.DataFrame:
    df_json = pd.read_json(path)
    if record_key:
        if record_key not in df_json.columns:
            raise ValueError(
                f"Expected record key '{record_key}' not present in JSON data."
            )
        records = pd.json_normalize(df_json[record_key])
    else:
        records = pd.json_normalize(df_json)
    return records


def load_ticket_data(data_cfg: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Load support ticket data according to configuration.

    Supports CSV files as well as JSON exports where relevant fields live within a
    nested record (e.g., `_source` as found in the CFPB complaint dataset).
    """
    if "raw_path" not in data_cfg:
        raise ValueError("Data configuration must include 'raw_path'.")

    path = Path(data_cfg["raw_path"])
    if not path.exists():
        raise FileNotFoundError(f"Ticket dataset not found at {path}.")

    fmt = data_cfg.get("format", "csv").lower()

    if fmt == "csv":
        df = _load_csv(path)
    elif fmt == "json":
        df = _load_json(path, record_key=data_cfg.get("record_key"))
    else:
        raise ValueError(f"Unsupported data format '{fmt}'.")

    text_column = data_cfg.get("text_column")
    label_column = data_cfg.get("label_column")

    if not text_column or not label_column:
        raise ValueError("Data configuration must include 'text_column' and 'label_column'.")

    missing_cols = {text_column, label_column}.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {', '.join(sorted(missing_cols))}"
        )

    df = df[[text_column, label_column]].copy()

    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[label_column] = df[label_column].fillna("").astype(str).str.strip()

    if data_cfg.get("drop_empty_text", True):
        df = df[df[text_column].str.len() > 0]
    if data_cfg.get("drop_missing_label", True):
        df = df[df[label_column].str.len() > 0]

    limit_records = data_cfg.get("limit_records")
    if isinstance(limit_records, int) and limit_records > 0:
        df = df.head(limit_records)

    df = df.drop_duplicates(subset=[text_column, label_column])

    min_label_count = data_cfg.get("min_label_count", 1)
    if isinstance(min_label_count, int) and min_label_count > 1:
        label_counts = df[label_column].value_counts()
        valid_labels = label_counts[label_counts >= min_label_count].index
        df = df[df[label_column].isin(valid_labels)]

    if df.empty:
        raise ValueError("No records available after applying filters.")

    X = df[text_column].astype(str)
    y = df[label_column].astype(str)
    return X, y
