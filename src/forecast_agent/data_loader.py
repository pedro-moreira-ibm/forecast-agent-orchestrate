"""Data loading utilities for the forecast agent."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATE_COLUMN = "date"


class DataValidationError(ValueError):
    """Raised when the incoming data does not meet expected assumptions."""


def load_saas_metrics(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load SaaS metrics from CSV and return a monthly indexed DataFrame.

    Args:
        path: Path to the CSV file.
        required_columns: Optional iterable of columns that must be present.

    Returns:
        DataFrame indexed by the month start date with float values.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if DATE_COLUMN not in df.columns:
        raise DataValidationError(f"Expected '{DATE_COLUMN}' column to be present")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        raise DataValidationError("Found invalid or missing dates in the dataset")

    df = df.sort_values(DATE_COLUMN)
    df = df.set_index(DATE_COLUMN)

    # Enforce monthly frequency
    df.index = df.index.to_period("M").to_timestamp()

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise DataValidationError(f"Columns contain non-numeric values: {missing}")

    if required_columns is not None:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {', '.join(sorted(missing_cols))}"
            )

    return df
