"""Load and validate raw cement kiln energy and CO2 datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_ENERGY_COLUMNS: tuple[str, ...] = (
    "SN",
    "Power (Kw)",
    "Power Consumption (MWH)",
    "Power Consumed (KWt/h)",
    "Absorbed Power (Kw)",
    "Power Consumption (Kw) Max",
    "Power (rotary drive) Kw",
    "Energy Consumption (kWh/t-clinker)",
)

REQUIRED_CO2_COLUMNS: tuple[str, ...] = (
    "SN",
    "Kiln Temperature (°C)",
    "Fuel Consumption (kg/h)",
    "Raw Meal Composition (% Limestone)",
    "Clinker Production Rate (tons/h)",
    "Oxygen Levels (%)",
    "Kiln Pressure (bar)",
    "Air Flow Rate (m³/h)",
    "Fuel",
    "CO2 Emissions (kg/h)",
)


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def load_and_merge_datasets(
    energy_csv: str | Path,
    co2_csv: str | Path,
    join_column: str = "SN",
    deduplicate: bool = False,
) -> pd.DataFrame:
    """Load energy and CO2 datasets, join explicitly on `join_column`, and clean rows.

    The `join_column` (e.g., `SN`) is dropped immediately after merging to prevent
    row-index data leakage during model training.
    """
    energy_path = Path(energy_csv)
    co2_path = Path(co2_csv)

    if not energy_path.exists():
        raise FileNotFoundError(f"Energy dataset not found: {energy_path}")
    if not co2_path.exists():
        raise FileNotFoundError(f"CO2 dataset not found: {co2_path}")

    energy_df = pd.read_csv(energy_path)
    co2_df = pd.read_csv(co2_path)

    _validate_columns(energy_df, REQUIRED_ENERGY_COLUMNS, "kiln_energy_dataset")
    _validate_columns(co2_df, REQUIRED_CO2_COLUMNS, "kiln_co2_dataset")

    merged = pd.merge(energy_df, co2_df, on=join_column, how="inner")

    if join_column in merged.columns:
        merged = merged.drop(columns=[join_column])

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if deduplicate:
        merged = merged.drop_duplicates().reset_index(drop=True)

    if merged.empty:
        raise ValueError("Merged dataset is empty after joining and dropping invalid rows.")

    return merged
