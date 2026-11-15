"""Preprocessing helpers for controllable plant variables and categorical fuel encoding."""

from __future__ import annotations

import pandas as pd

CONTROLLABLE_NUMERIC_FEATURES: tuple[str, ...] = (
    "Power (Kw)",
    "Power Consumption (MWH)",
    "Power Consumed (KWt/h)",
    "Absorbed Power (Kw)",
    "Power Consumption (Kw) Max",
    "Power (rotary drive) Kw",
    "Kiln Temperature (°C)",
    "Fuel Consumption (kg/h)",
    "Raw Meal Composition (% Limestone)",
    "Clinker Production Rate (tons/h)",
    "Oxygen Levels (%)",
    "Kiln Pressure (bar)",
    "Air Flow Rate (m³/h)",
)

FUEL_CATEGORIES: tuple[str, ...] = ("Coal", "Gas", "Oil")


def get_controllable_feature_names(include_fuel_encoding: bool = True) -> list[str]:
    """Return ordered list of controllable input features prior to engineering transforms."""
    features = list(CONTROLLABLE_NUMERIC_FEATURES)
    if include_fuel_encoding:
        features.extend(f"Fuel_{fuel}" for fuel in FUEL_CATEGORIES)
    return features


def encode_fuel_column(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the `Fuel` column into deterministic `Fuel_<category>` columns."""
    out = df.copy()
    if "Fuel" in out.columns:
        fuel_series = out["Fuel"].astype(str).str.strip()
        for fuel in FUEL_CATEGORIES:
            out[f"Fuel_{fuel}"] = (fuel_series == fuel).astype(float)
        out = out.drop(columns=["Fuel"])
    else:
        for fuel in FUEL_CATEGORIES:
            col = f"Fuel_{fuel}"
            if col not in out.columns:
                out[col] = 0.0
    return out


def prepare_modeling_dataframe(
    df: pd.DataFrame,
    energy_target: str = "Energy Consumption (kWh/t-clinker)",
    co2_target: str = "CO2 Emissions (kg/h)",
    include_fuel_encoding: bool = True,
) -> pd.DataFrame:
    """Prepare cleaned modeling DataFrame with controllable features and targets."""
    processed = (
        encode_fuel_column(df)
        if include_fuel_encoding
        else df.drop(columns=["Fuel"], errors="ignore")
    )
    feature_cols = get_controllable_feature_names(include_fuel_encoding=include_fuel_encoding)
    ordered_cols = feature_cols + [energy_target, co2_target]
    missing = [col for col in ordered_cols if col not in processed.columns]
    if missing:
        raise ValueError(f"Missing expected columns after preprocessing: {missing}")
    return processed[ordered_cols].copy()
