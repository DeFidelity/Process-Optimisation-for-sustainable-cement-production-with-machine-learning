"""Tests for NSGA-II constraint satisfaction, thermodynamic consistency, and ProcessOptimizer."""

import numpy as np
import pytest

from cement_opt.config import OptimizationConfig, load_config
from cement_opt.data import (
    get_controllable_feature_names,
    load_and_merge_datasets,
    prepare_modeling_dataframe,
)
from cement_opt.models import CementProcessModel
from cement_opt.optimization import CementOptimizationProblem, ProcessOptimizer, run_nsga2


def test_nsga2_respects_production_constraint_and_thermodynamics() -> None:
    cfg = load_config("configs/default.yaml")
    merged = load_and_merge_datasets(cfg.paths.energy_csv, cfg.paths.co2_csv)
    modeling_df = prepare_modeling_dataframe(merged, include_fuel_encoding=True)
    features = get_controllable_feature_names(include_fuel_encoding=True)

    energy_model = CementProcessModel(
        cfg.data.energy_target, features, cfg.model, cfg.thermodynamics
    )
    energy_model.train(modeling_df)

    co2_model = CementProcessModel(
        cfg.data.co2_target, features, cfg.model, cfg.thermodynamics
    )
    co2_model.train(modeling_df)

    problem = CementOptimizationProblem(
        energy_model=energy_model,
        co2_model=co2_model,
        production_column=cfg.data.production_column,
    )
    fast_opt = OptimizationConfig(
        pop_size=20, n_gen=20, seed=42, eliminate_duplicates=True
    )
    result = run_nsga2(problem, fast_opt, cfg.thermodynamics)

    # 1. Production constraint g(x) <= 0
    assert (result.pareto_df["Constraint_Violation_G"] <= 1e-6).all()
    assert (
        result.pareto_df[cfg.data.production_column] >= problem.min_production_rate - 1e-6
    ).all()

    # 2. Thermodynamic consistency on Pareto solutions
    recomputed_tco2 = (
        result.pareto_df["Raw Meal Composition (% Limestone)"]
        * result.pareto_df["Clinker Production Rate (tons/h)"]
        * cfg.thermodynamics.stoichiometric_co2_factor
    )
    assert np.allclose(result.pareto_df["Theoretical_CO2"], recomputed_tco2)

    # 3. Valid one-hot fuel choice
    fuel_sum = (
        result.pareto_df["Fuel_Coal"]
        + result.pareto_df["Fuel_Gas"]
        + result.pareto_df["Fuel_Oil"]
    )
    assert np.allclose(fuel_sum, 1.0)

    # 4. ProcessOptimizer extracts all three regimes and adjustments
    optimizer = ProcessOptimizer(energy_model, co2_model, result)
    rep_df = optimizer.get_representative_solutions()
    assert list(rep_df["Regime"]) == ["Low Energy", "Balanced", "Low CO2"]

    recs = optimizer.recommend_adjustments("Balanced")
    assert not recs.empty
    assert {"Parameter", "Plant_Median", "Recommended_Balanced", "Delta", "Change_%"} == set(
        recs.columns
    )
    assert np.isfinite(result.pareto_F).all()
    low_e = rep_df.loc[rep_df["Regime"] == "Low Energy", cfg.data.energy_target].iloc[0]
    assert pytest.approx(low_e) == result.pareto_F[:, 0].min()
