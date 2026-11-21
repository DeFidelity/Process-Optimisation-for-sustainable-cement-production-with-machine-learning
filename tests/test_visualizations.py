"""Tests for publication plots and interactive HTML visual explainer generation."""

from pathlib import Path

from cement_opt.config import OptimizationConfig, load_config
from cement_opt.data import (
    get_controllable_feature_names,
    load_and_merge_datasets,
    prepare_modeling_dataframe,
)
from cement_opt.models import CementProcessModel
from cement_opt.optimization import CementOptimizationProblem, ProcessOptimizer, run_nsga2
from cement_opt.visualization import (
    plot_nsga2_convergence,
    plot_physics_sensitivity,
    plot_regime_comparison,
    plot_setpoint_deltas,
    save_architecture_visualizer,
    save_system_architecture_svg,
    save_visual_explainer,
)


def test_extended_visualizations_and_dashboard(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    merged = load_and_merge_datasets(cfg.paths.energy_csv, cfg.paths.co2_csv)
    df = prepare_modeling_dataframe(merged, include_fuel_encoding=True)
    features = get_controllable_feature_names(include_fuel_encoding=True)

    energy_model = CementProcessModel(
        cfg.data.energy_target, features, cfg.model, cfg.thermodynamics
    )
    energy_model.train(df)

    co2_model = CementProcessModel(
        cfg.data.co2_target, features, cfg.model, cfg.thermodynamics
    )
    co2_model.train(df)

    problem = CementOptimizationProblem(
        energy_model, co2_model, production_column=cfg.data.production_column
    )
    fast_opt = OptimizationConfig(pop_size=16, n_gen=12, seed=42, eliminate_duplicates=True)
    opt_result = run_nsga2(problem, fast_opt, cfg.thermodynamics)
    optimizer = ProcessOptimizer(energy_model, co2_model, opt_result)

    p1 = plot_setpoint_deltas(optimizer, tmp_path / "setpoint_deltas.png")
    p2 = plot_regime_comparison(optimizer, df, tmp_path / "regime_comparison.png")
    p3 = plot_nsga2_convergence(opt_result, df, tmp_path / "nsga2_convergence.png")
    p4 = plot_physics_sensitivity(energy_model, co2_model, df, tmp_path / "physics.png")
    html_file = save_visual_explainer(
        energy_model, co2_model, opt_result, optimizer, df, tmp_path / "explainer.html"
    )
    arch_html = save_architecture_visualizer(
        energy_model, co2_model, opt_result, optimizer, df, tmp_path / "arch.html"
    )
    arch_svg = save_system_architecture_svg(
        energy_model, co2_model, opt_result, optimizer, df, tmp_path / "arch.svg"
    )

    for artifact in (p1, p2, p3, p4, html_file, arch_html, arch_svg):
        assert artifact.exists()
        assert artifact.stat().st_size > 1000

    html_text = html_file.read_text(encoding="utf-8")
    assert "explainer-root" in html_text
    assert "pareto-svg" in html_text
    assert "stage-drawer" in html_text

    arch_text = arch_html.read_text(encoding="utf-8")
    assert "arch-svg" in arch_text
    assert "scada-svg" in arch_text
    assert "pareto-scatter-svg" in arch_text
    assert "view-switch" in arch_text
    assert "live-stream-btn" in arch_text
    assert "baseline-compare-btn" in arch_text
    assert "inspector-panel" in arch_text
    assert "<svg" in arch_svg.read_text(encoding="utf-8")
