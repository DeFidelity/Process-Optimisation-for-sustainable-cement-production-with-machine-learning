"""Serialize and load trained model artifacts and feature bounds."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from cement_opt.models.trainer import CementProcessModel


def save_artifacts(
    models_dir: str | Path,
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
) -> dict[str, Path]:
    """Save trained Random Forest pipelines and feature bound metadata."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    energy_path = out_dir / "energy_rf_model.joblib"
    co2_path = out_dir / "co2_rf_model.joblib"
    metadata_path = out_dir / "model_metadata.json"

    joblib.dump(energy_model, energy_path)
    joblib.dump(co2_model, co2_path)

    metadata: dict[str, Any] = {
        "controllable_features": energy_model.controllable_features,
        "feature_bounds": {
            feat: {
                "min": float(lo),
                "max": float(hi),
                "median": float(energy_model.feature_medians[feat])
                if energy_model.feature_medians is not None
                else None,
            }
            for feat, lo, hi in zip(
                energy_model.controllable_features,
                energy_model.feature_mins if energy_model.feature_mins is not None else [],
                energy_model.feature_maxs if energy_model.feature_maxs is not None else [],
            )
        },
        "energy_metrics": asdict(energy_model.metrics) if energy_model.metrics else None,
        "co2_metrics": asdict(co2_model.metrics) if co2_model.metrics else None,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "energy_model": energy_path,
        "co2_model": co2_path,
        "metadata": metadata_path,
    }


def load_artifacts(models_dir: str | Path) -> tuple[CementProcessModel, CementProcessModel]:
    """Load persisted Energy and CO2 CementProcessModel objects."""
    base_dir = Path(models_dir)
    energy_path = base_dir / "energy_rf_model.joblib"
    co2_path = base_dir / "co2_rf_model.joblib"

    if not energy_path.exists() or not co2_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {base_dir}. Run the training stage first."
        )

    energy_model = joblib.load(energy_path)
    co2_model = joblib.load(co2_path)
    return energy_model, co2_model
