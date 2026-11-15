"""Pymoo multi-objective problem over controllable cement kiln variables only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from cement_opt.data.preprocessing import CONTROLLABLE_NUMERIC_FEATURES, FUEL_CATEGORIES
from cement_opt.models.trainer import CementProcessModel


class CementOptimizationProblem(Problem):
    """Multi-objective optimization problem minimizing Energy Consumption and CO2 Emissions.

    Decision variables consist exclusively of controllable plant parameters (and an optional
    discrete fuel-choice index). Derived thermodynamic features (`Theoretical_CO2`,
    `Theoretical_Energy`, `Heat_Efficiency`, `O2_Deviation`) are dynamically recomputed
    inside the model pipelines during `_evaluate()` to guarantee physical consistency.
    """

    def __init__(
        self,
        energy_model: CementProcessModel,
        co2_model: CementProcessModel,
        min_production_rate: float | None = None,
        production_column: str = "Clinker Production Rate (tons/h)",
    ) -> None:
        if energy_model.feature_mins is None or energy_model.feature_maxs is None:
            raise ValueError(
                "energy_model must be trained before constructing optimization problem."
            )
        if energy_model.controllable_features != co2_model.controllable_features:
            raise ValueError("Energy and CO2 models must share identical controllable_features.")

        self.energy_model = energy_model
        self.co2_model = co2_model
        self.production_column = production_column

        self.numeric_features = [
            f for f in CONTROLLABLE_NUMERIC_FEATURES if f in energy_model.controllable_features
        ]
        self.fuel_columns = [f"Fuel_{fuel}" for fuel in FUEL_CATEGORIES]
        self.has_fuel_encoding = all(
            col in energy_model.controllable_features for col in self.fuel_columns
        )

        # Bounds for numeric controllable decision variables
        feat_idx = {feat: idx for idx, feat in enumerate(energy_model.controllable_features)}
        xl_list = [float(energy_model.feature_mins[feat_idx[f]]) for f in self.numeric_features]
        xu_list = [float(energy_model.feature_maxs[feat_idx[f]]) for f in self.numeric_features]

        if self.has_fuel_encoding:
            # Continuous proxy rounded to {0: Coal, 1: Gas, 2: Oil}
            xl_list.append(-0.49)
            xu_list.append(len(FUEL_CATEGORIES) - 0.51)

        medians = energy_model.feature_medians
        if min_production_rate is not None:
            self.min_production_rate = float(min_production_rate)
        elif medians is not None and production_column in medians:
            self.min_production_rate = float(medians[production_column])
        else:
            raise ValueError("min_production_rate could not be determined from model medians.")

        super().__init__(
            n_var=len(xl_list),
            n_obj=2,
            n_constr=1,
            xl=np.asarray(xl_list, dtype=float),
            xu=np.asarray(xu_list, dtype=float),
        )

    def decode_decision_variables(self, X: np.ndarray) -> pd.DataFrame:
        """Convert NSGA-II decision matrix `X` into a controllable-feature DataFrame."""
        X_arr = np.atleast_2d(X)
        n_num = len(self.numeric_features)
        df = pd.DataFrame(X_arr[:, :n_num], columns=self.numeric_features)

        if self.has_fuel_encoding:
            fuel_idx = np.clip(
                np.round(X_arr[:, n_num]), 0, len(FUEL_CATEGORIES) - 1
            ).astype(int)
            for idx, col in enumerate(self.fuel_columns):
                df[col] = (fuel_idx == idx).astype(float)

        return df[self.energy_model.controllable_features]

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        X_df = self.decode_decision_variables(X)

        energy = self.energy_model.predict(X_df)
        co2 = self.co2_model.predict(X_df)

        # Constraint g(x) <= 0  <=>  Clinker Production Rate >= min_production_rate (Eq. 3.22)
        g = self.min_production_rate - X_df[self.production_column].to_numpy(dtype=float)

        out["F"] = np.column_stack([energy, co2])
        out["G"] = g.reshape(-1, 1)
