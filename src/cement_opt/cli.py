"""Command-line interface for preprocessing, training, optimization, and reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cement_opt.config import ProjectConfig, load_config
from cement_opt.data import (
    get_controllable_feature_names,
    load_and_merge_datasets,
    prepare_modeling_dataframe,
)
from cement_opt.models import (
    CementProcessModel,
    build_validation_table,
    load_artifacts,
    save_artifacts,
)
from cement_opt.optimization import (
    CementOptimizationProblem,
    ProcessOptimizer,
    run_nsga2,
)
from cement_opt.visualization import (
    plot_feature_importance,
    plot_nsga2_convergence,
    plot_pareto_front,
    plot_physics_sensitivity,
    plot_r2_comparison,
    plot_regime_comparison,
    plot_setpoint_deltas,
    save_architecture_visualizer,
    save_system_architecture_svg,
    save_visual_explainer,
)


def run_preprocess(cfg: ProjectConfig) -> pd.DataFrame:
    """Merge raw kiln datasets, drop SN index, and persist processed modeling dataset."""
    merged = load_and_merge_datasets(
        energy_csv=cfg.paths.energy_csv,
        co2_csv=cfg.paths.co2_csv,
        join_column=cfg.data.join_column,
        deduplicate=cfg.data.deduplicate,
    )
    modeling_df = prepare_modeling_dataframe(
        merged,
        energy_target=cfg.data.energy_target,
        co2_target=cfg.data.co2_target,
        include_fuel_encoding=cfg.data.include_fuel_encoding,
    )
    cfg.paths.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(cfg.paths.processed_csv, index=False)
    print(f"[preprocess] Saved {len(modeling_df)} rows to {cfg.paths.processed_csv}")
    return modeling_df


def _load_processed_or_preprocess(cfg: ProjectConfig) -> pd.DataFrame:
    if cfg.paths.processed_csv.exists():
        return pd.read_csv(cfg.paths.processed_csv)
    return run_preprocess(cfg)


def run_train(cfg: ProjectConfig) -> tuple[CementProcessModel, CementProcessModel]:
    """Train Energy and CO2 Random Forest pipelines and save model artifacts."""
    modeling_df = _load_processed_or_preprocess(cfg)
    features = get_controllable_feature_names(
        include_fuel_encoding=cfg.data.include_fuel_encoding
    )

    energy_model = CementProcessModel(
        target_var=cfg.data.energy_target,
        controllable_features=features,
        model_config=cfg.model,
        thermo_config=cfg.thermodynamics,
    )
    e_metrics = energy_model.train(modeling_df)

    co2_model = CementProcessModel(
        target_var=cfg.data.co2_target,
        controllable_features=features,
        model_config=cfg.model,
        thermo_config=cfg.thermodynamics,
    )
    c_metrics = co2_model.train(modeling_df)

    paths = save_artifacts(cfg.paths.models_dir, energy_model, co2_model)
    print(
        f"[train] Energy RF -> Test R2: {e_metrics.r2_test:.4f}, "
        f"RMSE: {e_metrics.rmse_test:.4f}, "
        f"CV R2: {e_metrics.cv_r2_mean:.4f} ± {e_metrics.cv_r2_std:.4f}"
    )
    print(
        f"[train] CO2 RF    -> Test R2: {c_metrics.r2_test:.4f}, "
        f"RMSE: {c_metrics.rmse_test:.4f}, "
        f"CV R2: {c_metrics.cv_r2_mean:.4f} ± {c_metrics.cv_r2_std:.4f}"
    )
    print(f"[train] Saved artifacts in {paths['energy_model'].parent}")
    return energy_model, co2_model


def run_optimize(
    cfg: ProjectConfig,
    energy_model: CementProcessModel | None = None,
    co2_model: CementProcessModel | None = None,
) -> None:
    """Run NSGA-II multi-objective optimization, export Chapter 4 tables, and render plots."""
    modeling_df = _load_processed_or_preprocess(cfg)
    if energy_model is None or co2_model is None:
        energy_model, co2_model = load_artifacts(cfg.paths.models_dir)

    problem = CementOptimizationProblem(
        energy_model=energy_model,
        co2_model=co2_model,
        production_column=cfg.data.production_column,
    )
    opt_result = run_nsga2(
        problem=problem,
        opt_config=cfg.optimization,
        thermo_config=cfg.thermodynamics,
        energy_target=cfg.data.energy_target,
        co2_target=cfg.data.co2_target,
    )

    optimizer = ProcessOptimizer(energy_model, co2_model, opt_result)
    rep_table = optimizer.get_representative_solutions()
    balanced = optimizer.get_balanced_solution()
    validation_table = build_validation_table(
        energy_model=energy_model,
        co2_model=co2_model,
        modeling_df=modeling_df,
        balanced_solution=balanced,
        energy_target=cfg.data.energy_target,
        co2_target=cfg.data.co2_target,
        production_col=cfg.data.production_column,
    )
    recommendations = optimizer.recommend_adjustments(regime="Balanced")

    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    pareto_all_path = cfg.paths.tables_dir / "pareto_front_results.csv"
    table_4_1_path = cfg.paths.tables_dir / "table_4_1_pareto_solutions.csv"
    table_4_2_path = cfg.paths.tables_dir / "table_4_2_model_validation.csv"
    rec_path = cfg.paths.tables_dir / "balanced_setpoint_recommendations.csv"

    opt_result.pareto_df.to_csv(pareto_all_path, index=False)
    rep_table.to_csv(table_4_1_path, index=False)
    validation_table.to_csv(table_4_2_path, index=False)
    recommendations.to_csv(rec_path, index=False)

    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_r2_comparison(
        energy_model,
        co2_model,
        cfg.paths.figures_dir / "r2_performance_chart.png",
    )
    plot_feature_importance(
        energy_model,
        "Energy Consumption Model",
        cfg.paths.figures_dir / "feature_importance_energy.png",
    )
    plot_feature_importance(
        co2_model,
        "CO2 Emissions Model",
        cfg.paths.figures_dir / "feature_importance_co2.png",
    )
    plot_pareto_front(
        opt_result,
        optimizer,
        modeling_df,
        cfg.paths.figures_dir / "pareto_front.png",
    )
    plot_setpoint_deltas(
        optimizer,
        cfg.paths.figures_dir / "setpoint_deltas.png",
        regime="Balanced",
    )
    plot_regime_comparison(
        optimizer,
        modeling_df,
        cfg.paths.figures_dir / "regime_comparison.png",
    )
    plot_nsga2_convergence(
        opt_result,
        modeling_df,
        cfg.paths.figures_dir / "nsga2_convergence.png",
    )
    plot_physics_sensitivity(
        energy_model,
        co2_model,
        modeling_df,
        cfg.paths.figures_dir / "physics_sensitivity.png",
        optimal_o2_percent=cfg.thermodynamics.optimal_o2_percent,
    )
    dashboard_path = save_visual_explainer(
        energy_model,
        co2_model,
        opt_result,
        optimizer,
        modeling_df,
        cfg.paths.dashboard_html,
    )
    arch_html_path = save_architecture_visualizer(
        energy_model,
        co2_model,
        opt_result,
        optimizer,
        modeling_df,
        cfg.paths.architecture_html,
    )
    arch_svg_path = save_system_architecture_svg(
        energy_model,
        co2_model,
        opt_result,
        optimizer,
        modeling_df,
        cfg.paths.figures_dir / "system_architecture.svg",
    )

    print(f"[optimize] Found {len(opt_result.pareto_df)} Pareto-optimal solutions.")
    print(f"[optimize] Saved tables to {cfg.paths.tables_dir}")
    print(f"[optimize] Saved figures to {cfg.paths.figures_dir}")
    print(f"[optimize] Saved interactive visual explainer to {dashboard_path}")
    print(f"[optimize] Saved interactive architecture visualizer to {arch_html_path}")
    print(f"[optimize] Saved system architecture SVG to {arch_svg_path}")
    print("\n--- Table 4.2: Model Validation & Deviation Analysis ---")
    print(validation_table.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cement-opt",
        description="Sustainable Cement Production ML & NSGA-II Optimization Pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preprocess", help="Merge raw datasets and prepare modeling data.")
    subparsers.add_parser("train", help="Train Random Forest models and save artifacts.")
    subparsers.add_parser("optimize", help="Run NSGA-II optimization and generate outputs.")
    subparsers.add_parser("run-all", help="Execute preprocess -> train -> optimize end-to-end.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "preprocess":
        run_preprocess(cfg)
    elif args.command == "train":
        run_train(cfg)
    elif args.command == "optimize":
        run_optimize(cfg)
    elif args.command == "run-all":
        run_preprocess(cfg)
        energy_model, co2_model = run_train(cfg)
        run_optimize(cfg, energy_model=energy_model, co2_model=co2_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
