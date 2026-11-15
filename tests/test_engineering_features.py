"""Unit tests for chemical engineering feature equations (Eq. 3.13 - 3.17)."""

import numpy as np
import pandas as pd
import pytest

from cement_opt.features.engineering import (
    ChemicalEngineeringTransformer,
    add_engineering_features,
)


def test_add_engineering_features_matches_equations() -> None:
    df = pd.DataFrame(
        {
            "Raw Meal Composition (% Limestone)": [85.0, 80.0],
            "Clinker Production Rate (tons/h)": [90.0, 100.0],
            "Fuel Consumption (kg/h)": [200.0, 0.0],
            "Oxygen Levels (%)": [3.5, 4.5],
        }
    )

    out = add_engineering_features(df)

    # Eq. 3.13: RMC * CPR * 0.44
    assert out["Theoretical_CO2"].iloc[0] == pytest.approx(85.0 * 90.0 * 0.44)
    assert out["Theoretical_CO2"].iloc[1] == pytest.approx(80.0 * 100.0 * 0.44)

    # Eq. 3.14-3.15: (CPR * 1000 * 1780) / 3600
    expected_te_0 = (90.0 * 1000.0 * 1780.0) / 3600.0
    assert out["Theoretical_Energy"].iloc[0] == pytest.approx(expected_te_0)

    # Eq. 3.16: Heat_Efficiency = TE / FC (0 when FC == 0)
    assert out["Heat_Efficiency"].iloc[0] == pytest.approx(expected_te_0 / 200.0)
    assert out["Heat_Efficiency"].iloc[1] == 0.0

    # Eq. 3.17: |O2 - 4.0|
    assert out["O2_Deviation"].iloc[0] == pytest.approx(0.5)
    assert out["O2_Deviation"].iloc[1] == pytest.approx(0.5)


def test_transformer_sklearn_interface() -> None:
    df = pd.DataFrame(
        {
            "Raw Meal Composition (% Limestone)": [84.0],
            "Clinker Production Rate (tons/h)": [92.0],
            "Fuel Consumption (kg/h)": [195.0],
            "Oxygen Levels (%)": [4.0],
        }
    )
    transformer = ChemicalEngineeringTransformer()
    transformed = transformer.fit_transform(df)
    assert list(transformer.get_feature_names_out()) == list(transformed.columns)
    assert np.isfinite(transformed.to_numpy()).all()
