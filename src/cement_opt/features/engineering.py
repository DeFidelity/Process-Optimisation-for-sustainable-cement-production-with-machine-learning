"""Domain-specific chemical engineering feature transformations (Equations 3.13 - 3.17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURES: tuple[str, ...] = (
    "Theoretical_CO2",
    "Theoretical_Energy",
    "Heat_Efficiency",
    "O2_Deviation",
)


def add_engineering_features(
    df: pd.DataFrame,
    stoichiometric_co2_factor: float = 0.44,
    calcination_enthalpy_kj_per_kg: float = 1780.0,
    optimal_o2_percent: float = 4.0,
) -> pd.DataFrame:
    """Compute thermodynamic and stoichiometric features on a DataFrame copy.

    Equations from thesis Section 3.4:
      - Eq. 3.13: Theoretical_CO2 = RMC (% CaCO3) * CPR (tons/h) * 0.44
      - Eq. 3.14-3.15: Theoretical_Energy (kW) = (CPR * 1000 * calcination_enthalpy) / 3600
      - Eq. 3.16: Heat_Efficiency = Theoretical_Energy / Fuel_Consumption (kg/h)
      - Eq. 3.17: O2_Deviation = |Oxygen_Levels (%) - optimal_o2_percent|
    """
    out = df.copy()

    limestone_col = "Raw Meal Composition (% Limestone)"
    production_col = "Clinker Production Rate (tons/h)"
    fuel_col = "Fuel Consumption (kg/h)"
    oxygen_col = "Oxygen Levels (%)"

    if limestone_col in out.columns and production_col in out.columns:
        out["Theoretical_CO2"] = (
            out[limestone_col] * out[production_col] * stoichiometric_co2_factor
        )

    if production_col in out.columns and fuel_col in out.columns:
        out["Theoretical_Energy"] = (
            out[production_col] * 1000.0 * calcination_enthalpy_kj_per_kg
        ) / 3600.0
        out["Heat_Efficiency"] = np.where(
            out[fuel_col] != 0.0,
            out["Theoretical_Energy"] / out[fuel_col],
            0.0,
        )

    if oxygen_col in out.columns:
        out["O2_Deviation"] = np.abs(out[oxygen_col] - optimal_o2_percent)

    return out


class ChemicalEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer injecting chemical engineering features into a pipeline."""

    def __init__(
        self,
        stoichiometric_co2_factor: float = 0.44,
        calcination_enthalpy_kj_per_kg: float = 1780.0,
        optimal_o2_percent: float = 4.0,
    ) -> None:
        self.stoichiometric_co2_factor = stoichiometric_co2_factor
        self.calcination_enthalpy_kj_per_kg = calcination_enthalpy_kj_per_kg
        self.optimal_o2_percent = optimal_o2_percent
        self.feature_names_out_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> "ChemicalEngineeringTransformer":
        transformed = self._transform_df(X)
        self.feature_names_out_ = list(transformed.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform_df(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChemicalEngineeringTransformer requires a pandas DataFrame input.")
        return add_engineering_features(
            X,
            stoichiometric_co2_factor=self.stoichiometric_co2_factor,
            calcination_enthalpy_kj_per_kg=self.calcination_enthalpy_kj_per_kg,
            optimal_o2_percent=self.optimal_o2_percent,
        )
