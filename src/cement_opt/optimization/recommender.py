"""Recommendation engine completing ProcessOptimizer from ml_model.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cement_opt.data.preprocessing import CONTROLLABLE_NUMERIC_FEATURES
from cement_opt.models.trainer import CementProcessModel
from cement_opt.optimization.solver import OptimizationResult


class ProcessOptimizer:
    """Extracts Pareto-optimal operating regimes and recommends setpoint adjustments."""

    def __init__(
        self,
        energy_model: CementProcessModel,
        co2_model: CementProcessModel,
        optimization_result: OptimizationResult,
    ) -> None:
        self.energy_model = energy_model
        self.co2_model = co2_model
        self.result = optimization_result
        self.pareto_df = optimization_result.pareto_df.copy()
        self.energy_col = optimization_result.energy_target
        self.co2_col = optimization_result.co2_target

    def get_representative_indices(self) -> dict[str, int]:
        """Identify Low Energy, Balanced (knee point), and Low CO2 solution indices."""
        energy_vals = self.pareto_df[self.energy_col].to_numpy(dtype=float)
        co2_vals = self.pareto_df[self.co2_col].to_numpy(dtype=float)

        low_energy_idx = int(np.argmin(energy_vals))
        low_co2_idx = int(np.argmin(co2_vals))

        # Normalized Euclidean distance to utopia point (min energy, min CO2)
        e_span = max(float(energy_vals.max() - energy_vals.min()), 1e-9)
        c_span = max(float(co2_vals.max() - co2_vals.min()), 1e-9)
        e_norm = (energy_vals - energy_vals.min()) / e_span
        c_norm = (co2_vals - co2_vals.min()) / c_span
        utopia_dist = np.sqrt(e_norm**2 + c_norm**2)
        balanced_idx = int(np.argmin(utopia_dist))

        return {
            "Low Energy": low_energy_idx,
            "Balanced": balanced_idx,
            "Low CO2": low_co2_idx,
        }

    def get_representative_solutions(self) -> pd.DataFrame:
        """Build Table 4.1 containing the three representative Pareto-optimal operating points."""
        indices = self.get_representative_indices()
        rows = []
        for regime, idx in indices.items():
            row = self.pareto_df.iloc[idx].copy()
            row["Regime"] = regime
            rows.append(row)

        table_df = pd.DataFrame(rows)
        cols = ["Regime"] + [c for c in table_df.columns if c != "Regime"]
        return table_df[cols].reset_index(drop=True)

    def get_balanced_solution(self) -> pd.Series:
        """Return the knee-point Balanced Pareto-optimal solution."""
        idx = self.get_representative_indices()["Balanced"]
        return self.pareto_df.iloc[idx]

    def recommend_adjustments(self, regime: str = "Balanced") -> pd.DataFrame:
        """Compare a chosen optimal regime against historical plant medians."""
        if self.energy_model.feature_medians is None:
            raise ValueError("Baseline medians unavailable on energy_model.")

        indices = self.get_representative_indices()
        if regime not in indices:
            raise ValueError(f"Unknown regime '{regime}'. Choose from {list(indices.keys())}.")

        target_row = self.pareto_df.iloc[indices[regime]]
        medians = self.energy_model.feature_medians

        records = []
        for feat in CONTROLLABLE_NUMERIC_FEATURES:
            if feat in medians and feat in target_row:
                baseline = float(medians[feat])
                recommended = float(target_row[feat])
                delta = recommended - baseline
                pct_change = (delta / baseline * 100.0) if baseline != 0 else 0.0
                records.append(
                    {
                        "Parameter": feat,
                        "Plant_Median": round(baseline, 2),
                        f"Recommended_{regime.replace(' ', '_')}": round(recommended, 2),
                        "Delta": round(delta, 2),
                        "Change_%": round(pct_change, 2),
                    }
                )

        return pd.DataFrame(records)
