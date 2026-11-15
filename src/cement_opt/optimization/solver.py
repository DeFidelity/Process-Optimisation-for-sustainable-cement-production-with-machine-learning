"""NSGA-II multi-objective solver runner and result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from cement_opt.config import OptimizationConfig, ThermodynamicsConfig
from cement_opt.data.preprocessing import FUEL_CATEGORIES
from cement_opt.features.engineering import add_engineering_features
from cement_opt.optimization.problem import CementOptimizationProblem


@dataclass
class OptimizationResult:
    """Structured container for NSGA-II Pareto-optimal solutions and metadata."""

    pareto_df: pd.DataFrame
    pareto_F: np.ndarray
    pareto_G: np.ndarray
    energy_target: str
    co2_target: str
    raw_result: Any


def _decode_fuel_label(row: pd.Series) -> str:
    for fuel in FUEL_CATEGORIES:
        col = f"Fuel_{fuel}"
        if col in row and row[col] >= 0.5:
            return fuel
    return "Unknown"


def run_nsga2(
    problem: CementOptimizationProblem,
    opt_config: OptimizationConfig,
    thermo_config: ThermodynamicsConfig,
    energy_target: str = "Energy Consumption (kWh/t-clinker)",
    co2_target: str = "CO2 Emissions (kg/h)",
) -> OptimizationResult:
    """Run NSGA-II on the cement optimization problem and return decoded Pareto solutions."""
    algorithm = NSGA2(
        pop_size=opt_config.pop_size,
        eliminate_duplicates=opt_config.eliminate_duplicates,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", opt_config.n_gen),
        seed=opt_config.seed,
        save_history=True,
        verbose=False,
    )

    if res.X is None or len(res.X) == 0:
        raise RuntimeError("NSGA-II failed to find feasible Pareto-optimal solutions.")

    controllable_df = problem.decode_decision_variables(res.X)
    full_df = add_engineering_features(
        controllable_df,
        stoichiometric_co2_factor=thermo_config.stoichiometric_co2_factor,
        calcination_enthalpy_kj_per_kg=thermo_config.calcination_enthalpy_kj_per_kg,
        optimal_o2_percent=thermo_config.optimal_o2_percent,
    )

    if problem.has_fuel_encoding:
        full_df["Fuel"] = full_df.apply(_decode_fuel_label, axis=1)

    F = np.atleast_2d(res.F)
    G = np.atleast_2d(res.G) if res.G is not None else np.zeros((len(F), 1))

    full_df[energy_target] = F[:, 0]
    full_df[co2_target] = F[:, 1]
    full_df["Constraint_Violation_G"] = G[:, 0]

    # Sort monotonically along Energy Consumption for clean inspection & plotting
    sort_order = np.argsort(full_df[energy_target].to_numpy())
    full_df = full_df.iloc[sort_order].reset_index(drop=True)
    F_sorted = F[sort_order]
    G_sorted = G[sort_order]

    return OptimizationResult(
        pareto_df=full_df,
        pareto_F=F_sorted,
        pareto_G=G_sorted,
        energy_target=energy_target,
        co2_target=co2_target,
        raw_result=res,
    )
