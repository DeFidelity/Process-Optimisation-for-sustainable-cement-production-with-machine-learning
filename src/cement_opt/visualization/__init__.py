"""Publication-ready visualizations and interactive visual explainer."""

from cement_opt.visualization.dashboard import (
    build_architecture_visualizer_html,
    build_visual_explainer_html,
    save_architecture_visualizer,
    save_system_architecture_svg,
    save_visual_explainer,
)
from cement_opt.visualization.plots import (
    plot_feature_importance,
    plot_nsga2_convergence,
    plot_pareto_front,
    plot_physics_sensitivity,
    plot_r2_comparison,
    plot_regime_comparison,
    plot_setpoint_deltas,
)

__all__ = [
    "build_architecture_visualizer_html",
    "build_visual_explainer_html",
    "plot_feature_importance",
    "plot_nsga2_convergence",
    "plot_pareto_front",
    "plot_physics_sensitivity",
    "plot_r2_comparison",
    "plot_regime_comparison",
    "plot_setpoint_deltas",
    "save_architecture_visualizer",
    "save_system_architecture_svg",
    "save_visual_explainer",
]
