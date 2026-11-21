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
    assert "whatif-panel" in arch_text
    assert "whatif-sliders-grid" in arch_text
    assert "disturb-overburn" in arch_text
    assert "disturb-draft" in arch_text
    assert "disturb-cpr" in arch_text
    assert "ai-optimize-btn" in arch_text
    assert "ablation-toggle-btn" in arch_text
    assert "impact-strip" in arch_text
    assert "play-evolution-btn" in arch_text
    assert "tradeoff-zones-btn" in arch_text
    assert "pareto-zoom-btn" in arch_text
    assert "evolution-legend" in arch_text
    assert "zone-callout-bar" in arch_text
    assert "\"cal_anchors\":" in arch_text
    assert "1029.18" in arch_text
    assert "1030.42" in arch_text
    assert "920.42" in arch_text
    assert "1049.82" in arch_text
    assert "913.51" in arch_text
    assert "1168.21" in arch_text
    assert "1005.78" in arch_text
    assert "\"forests\":" in arch_text
    assert "\"ablation\":" in arch_text
    assert '"forest_features":' in arch_text
    assert "start-tour-btn" in arch_text
    assert "tour-hud" in arch_text
    assert "tour-prev-btn" in arch_text
    assert "tour-next-btn" in arch_text
    assert "tour-autoplay-btn" in arch_text
    assert "tour-exit-btn" in arch_text
    assert "validation-table-btn" in arch_text
    assert "act-fuel" in arch_text
    assert "act-idfan" in arch_text
    assert "act-rawmeal" in arch_text
    assert "act-drive" in arch_text
    assert "feedback-pulses-layer" in arch_text
    assert "TOUR_STOPS" in arch_text
    assert "triggerFeedbackPulse" in arch_text
    assert "unit-clinker" in arch_text
    assert "coupling-toggle-btn" in arch_text
    assert "copilot-toggle-btn" in arch_text
    assert "plant-profile-select" in arch_text
    assert '"coupling":' in arch_text
    assert '"plant_profiles":' in arch_text
    assert "rf-confidence-pill" in arch_text
    assert "horizon-oscilloscope-svg" in arch_text
    assert "clinker-soft-sensor" in arch_text
    assert "svg-fcao-val" in arch_text
    assert "supervisory-mode-switch" in arch_text
    assert "mode-auto-btn" in arch_text
    assert "mode-advisory-btn" in arch_text
    assert "advisory-toast" in arch_text
    assert "approve-advisory-btn" in arch_text
    assert "reject-advisory-btn" in arch_text
    assert "simulate-ood-btn" in arch_text
    assert "isa-interlock-banner" in arch_text
    assert "reset-interlock-btn" in arch_text
    assert "SLEW_LIMITS" in arch_text
    assert "evalCompactForestStats" in arch_text
    assert "executeRampedSetpointMove" in arch_text
    assert "triggerOodSensorFault" in arch_text
    assert "auto-mode-bar" in arch_text
    assert "mode-autonomous-btn" in arch_text
    assert "mode-manual-btn" in arch_text
    assert "auto-pause-btn" in arch_text
    assert "auto-inject-btn" in arch_text
    assert "ai-control-toggle-btn" in arch_text
    assert "dcs-oscilloscope-svg" in arch_text
    assert "dcs-status-ticker" in arch_text
    assert "dcs-early-warning-badge" in arch_text
    assert "Auto-Balance with NSGA-II" in arch_text
    assert "AUTONOMOUS_UPSETS" in arch_text
    assert "renderDcsOscilloscope" in arch_text
    assert "stepAutonomousSimulation" in arch_text
    assert "injectAutonomousDisturbanceNow" in arch_text
    assert "academic-metrics-btn" in arch_text
    assert "academic-modal-backdrop" in arch_text
    assert "acad-tab-btn-val" in arch_text
    assert "acad-tab-btn-pareto" in arch_text
    assert "acad-tab-btn-math" in arch_text
    assert "renderAcademicModal" in arch_text
    assert "openAcademicModal" in arch_text
    assert "closeAcademicModal" in arch_text
    assert "heat-roi-btn" in arch_text
    assert "heat-roi-modal-backdrop" in arch_text
    assert "computeKilnHeatBalance" in arch_text
    assert "renderHeatRoiModal" in arch_text
    assert "openHeatRoiModal" in arch_text
    assert "closeHeatRoiModal" in arch_text
    assert "rf-view-tabs" in arch_text
    assert "rf-view-shap-btn" in arch_text
    assert "rf-shap-svg" in arch_text
    assert "computeLocalShap" in arch_text
    assert "renderShapWaterfall" in arch_text
    assert "historian-replay-btn" in arch_text
    assert "historian-modal-backdrop" in arch_text
    assert "ASHAKA_24H_SHIFT_LOG" in arch_text
    assert "shift-replay-play-btn" in arch_text
    assert "csv-dropzone" in arch_text
    assert "download-csv-template-btn" in arch_text
    assert "opcua-console" in arch_text
    assert "ns=2;s=Ashaka.Kiln1.BurningZoneTemp" in arch_text
    assert "ns=2;s=Ashaka.Kiln1.GepolIDFan.Power" in arch_text
    assert "ns=2;s=Ashaka.Optimizer.NSGA2.Status" in arch_text
    assert "export-audit-btn" in arch_text
    assert "audit-modal-backdrop" in arch_text
    assert "print-audit-sheet" in arch_text
    assert "exportShiftAuditReport" in arch_text
    assert "Guided Tour" in arch_text
    assert "<svg" in arch_svg.read_text(encoding="utf-8")
