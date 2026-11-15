"""Evaluation metrics and plant validation comparison tables (Section 3.2.3 & Table 4.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    from cement_opt.models.trainer import CementProcessModel


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R2, RMSE, MAE, and MAPE for regression predictions."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    non_zero = y_true_arr != 0
    if np.any(non_zero):
        mape = float(
            np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero]))
            * 100.0
        )
    else:
        mape = 0.0

    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def _percent_deviation(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0
    return ((value - reference) / reference) * 100.0


def build_validation_table(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    modeling_df: pd.DataFrame,
    balanced_solution: pd.Series,
    energy_target: str = "Energy Consumption (kWh/t-clinker)",
    co2_target: str = "CO2 Emissions (kg/h)",
    production_col: str = "Clinker Production Rate (tons/h)",
) -> pd.DataFrame:
    """Construct Table 4.2 comparing Predicted, Balanced Optimized, and Plant Median values."""
    avg_features = modeling_df[energy_model.controllable_features].mean().to_frame().T

    pred_energy = float(energy_model.predict(avg_features)[0])
    pred_co2 = float(co2_model.predict(avg_features)[0])
    pred_prod = float(modeling_df[production_col].median())

    plant_energy = float(modeling_df[energy_target].median())
    plant_co2 = float(modeling_df[co2_target].median())
    plant_prod = float(modeling_df[production_col].median())

    opt_energy = float(balanced_solution[energy_target])
    opt_co2 = float(balanced_solution[co2_target])
    opt_prod = float(balanced_solution[production_col])

    energy_test_rmse = energy_model.metrics.rmse_test if energy_model.metrics else float("nan")
    co2_test_rmse = co2_model.metrics.rmse_test if co2_model.metrics else float("nan")

    rows = [
        {
            "Parameter": energy_target,
            "Model Predicted Value": round(pred_energy, 2),
            "Model Deviation (%)": round(_percent_deviation(pred_energy, plant_energy), 2),
            "Test RMSE": round(energy_test_rmse, 4),
            "Optimized Value (Balanced)": round(opt_energy, 2),
            "Optimized Deviation (%)": round(_percent_deviation(opt_energy, plant_energy), 2),
            "Plant Median Value": round(plant_energy, 2),
        },
        {
            "Parameter": co2_target,
            "Model Predicted Value": round(pred_co2, 2),
            "Model Deviation (%)": round(_percent_deviation(pred_co2, plant_co2), 2),
            "Test RMSE": round(co2_test_rmse, 4),
            "Optimized Value (Balanced)": round(opt_co2, 2),
            "Optimized Deviation (%)": round(_percent_deviation(opt_co2, plant_co2), 2),
            "Plant Median Value": round(plant_co2, 2),
        },
        {
            "Parameter": production_col,
            "Model Predicted Value": round(pred_prod, 2),
            "Model Deviation (%)": round(_percent_deviation(pred_prod, plant_prod), 2),
            "Test RMSE": 0.0,
            "Optimized Value (Balanced)": round(opt_prod, 2),
            "Optimized Deviation (%)": round(_percent_deviation(opt_prod, plant_prod), 2),
            "Plant Median Value": round(plant_prod, 2),
        },
    ]
    return pd.DataFrame(rows)
