"""Integration tests for data loading, SN leakage prevention, and model training."""

from pathlib import Path

import numpy as np

from cement_opt.config import load_config
from cement_opt.data import (
    get_controllable_feature_names,
    load_and_merge_datasets,
    prepare_modeling_dataframe,
)
from cement_opt.models import CementProcessModel, load_artifacts, save_artifacts


def test_loader_excludes_sn_and_encodes_fuel() -> None:
    cfg = load_config("configs/default.yaml")
    merged = load_and_merge_datasets(
        cfg.paths.energy_csv,
        cfg.paths.co2_csv,
        join_column=cfg.data.join_column,
        deduplicate=False,
    )
    assert "SN" not in merged.columns
    assert len(merged) == 100

    modeling_df = prepare_modeling_dataframe(
        merged,
        energy_target=cfg.data.energy_target,
        co2_target=cfg.data.co2_target,
        include_fuel_encoding=True,
    )
    for col in ("Fuel_Coal", "Fuel_Gas", "Fuel_Oil"):
        assert col in modeling_df.columns
    assert not modeling_df.isna().any().any()


def test_model_training_and_serialization(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    merged = load_and_merge_datasets(cfg.paths.energy_csv, cfg.paths.co2_csv)
    modeling_df = prepare_modeling_dataframe(merged, include_fuel_encoding=True)
    features = get_controllable_feature_names(include_fuel_encoding=True)

    energy_model = CementProcessModel(
        cfg.data.energy_target, features, cfg.model, cfg.thermodynamics
    )
    e_metrics = energy_model.train(modeling_df)

    co2_model = CementProcessModel(
        cfg.data.co2_target, features, cfg.model, cfg.thermodynamics
    )
    c_metrics = co2_model.train(modeling_df)

    assert e_metrics.r2_test > 0.85
    assert c_metrics.r2_test > 0.90
    assert np.isfinite(e_metrics.rmse_test)
    assert np.isfinite(c_metrics.rmse_test)

    save_artifacts(tmp_path, energy_model, co2_model)
    loaded_e, loaded_c = load_artifacts(tmp_path)

    sample = modeling_df[features].iloc[:3]
    assert np.allclose(energy_model.predict(sample), loaded_e.predict(sample))
    assert np.allclose(co2_model.predict(sample), loaded_c.predict(sample))
