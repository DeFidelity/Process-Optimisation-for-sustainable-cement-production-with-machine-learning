"""Configuration management for the cement optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    energy_csv: Path
    co2_csv: Path
    processed_csv: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    dashboard_html: Path = Path("outputs/visual_explainer.html")


@dataclass(frozen=True)
class DataConfig:
    join_column: str
    deduplicate: bool
    include_fuel_encoding: bool
    energy_target: str
    co2_target: str
    production_column: str


@dataclass(frozen=True)
class ThermodynamicsConfig:
    stoichiometric_co2_factor: float
    calcination_enthalpy_kj_per_kg: float
    optimal_o2_percent: float


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    max_depth: int | None
    test_size: float
    random_state: int
    cv_folds: int


@dataclass(frozen=True)
class OptimizationConfig:
    pop_size: int
    n_gen: int
    seed: int
    eliminate_duplicates: bool


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    data: DataConfig
    thermodynamics: ThermodynamicsConfig
    model: ModelConfig
    optimization: OptimizationConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path | None = None) -> "ProjectConfig":
        root = base_dir if base_dir is not None else Path.cwd()
        p = raw["paths"]
        paths = PathsConfig(
            energy_csv=root / p["energy_csv"],
            co2_csv=root / p["co2_csv"],
            processed_csv=root / p["processed_csv"],
            models_dir=root / p["models_dir"],
            figures_dir=root / p["figures_dir"],
            tables_dir=root / p["tables_dir"],
            dashboard_html=root / p.get("dashboard_html", "outputs/visual_explainer.html"),
        )
        return cls(
            paths=paths,
            data=DataConfig(**raw["data"]),
            thermodynamics=ThermodynamicsConfig(**raw["thermodynamics"]),
            model=ModelConfig(**raw["model"]),
            optimization=OptimizationConfig(**raw["optimization"]),
        )


def load_config(config_path: str | Path = "configs/default.yaml") -> ProjectConfig:
    """Load YAML configuration and resolve project paths relative to repo root."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # If config lives inside configs/, resolve paths relative to its parent directory
    base_dir = path.parent.parent if path.parent.name == "configs" else path.parent
    return ProjectConfig.from_dict(raw, base_dir=base_dir)
