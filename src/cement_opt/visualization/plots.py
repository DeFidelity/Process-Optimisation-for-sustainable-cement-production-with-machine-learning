"""Publication-ready matplotlib/seaborn figures matching Chapter 4 of the thesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cement_opt.models.trainer import CementProcessModel
from cement_opt.optimization.recommender import ProcessOptimizer
from cement_opt.optimization.solver import OptimizationResult

sns.set_theme(style="whitegrid", context="notebook")


def plot_r2_comparison(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    output_path: str | Path,
) -> Path:
    """Figure 4.1: Test R2 performance and parity scatter plots for Energy and CO2 models."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: R2 comparison bar chart
    models = ["Energy Consumption\n(kWh/t-clinker)", "CO2 Emissions\n(kg/h)"]
    r2_vals = [
        energy_model.metrics.r2_test if energy_model.metrics else 0.0,
        co2_model.metrics.r2_test if co2_model.metrics else 0.0,
    ]
    bars = axes[0].bar(models, r2_vals, color=["#1f77b4", "#2ca02c"], width=0.5)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Coefficient of Determination (R²)")
    axes[0].set_title("Model R² Comparison (Test Set)")
    for bar, val in zip(bars, r2_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel B: Energy parity plot
    if energy_model.y_test_ is not None and energy_model.y_pred_ is not None:
        y_t = energy_model.y_test_.to_numpy()
        y_p = energy_model.y_pred_
        axes[1].scatter(y_t, y_p, alpha=0.8, color="#1f77b4", edgecolor="white", s=55)
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        axes[1].plot(lims, lims, "r--", lw=1.5, label="Ideal Fit (y = x)")
        axes[1].set_xlabel("Actual Energy (kWh/t-clinker)")
        axes[1].set_ylabel("Predicted Energy (kWh/t-clinker)")
        axes[1].set_title(f"Energy Parity (R² = {r2_vals[0]:.4f})")
        axes[1].legend()

    # Panel C: CO2 parity plot
    if co2_model.y_test_ is not None and co2_model.y_pred_ is not None:
        y_t = co2_model.y_test_.to_numpy()
        y_p = co2_model.y_pred_
        axes[2].scatter(y_t, y_p, alpha=0.8, color="#2ca02c", edgecolor="white", s=55)
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        axes[2].plot(lims, lims, "r--", lw=1.5, label="Ideal Fit (y = x)")
        axes[2].set_xlabel("Actual CO2 Emissions (kg/h)")
        axes[2].set_ylabel("Predicted CO2 Emissions (kg/h)")
        axes[2].set_title(f"CO2 Parity (R² = {r2_vals[1]:.4f})")
        axes[2].legend()

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(
    model: CementProcessModel,
    title: str,
    output_path: str | Path,
) -> Path:
    """Figures 4.2 & 4.3: Horizontal bar chart of Random Forest feature importances."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    imp_df = model.get_feature_importances().sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * len(imp_df))))
    colors = sns.color_palette("viridis", n_colors=len(imp_df))
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_xlabel("Relative Feature Importance")
    ax.set_title(f"Feature Importance: {title}")

    for idx, val in enumerate(imp_df["importance"]):
        ax.text(val + 0.003, idx, f"{val:.3f}", va="center", fontsize=8.5)

    ax.set_xlim(0, float(imp_df["importance"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_front(
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Figure 4.4: Pareto front trade-off between Energy Consumption and CO2 Emissions."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    F = opt_result.pareto_F
    rep_df = optimizer.get_representative_solutions()

    plant_energy = float(modeling_df[opt_result.energy_target].median())
    plant_co2 = float(modeling_df[opt_result.co2_target].median())

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(F[:, 0], F[:, 1], color="#1f77b4", alpha=0.45, linewidth=1.5, zorder=1)
    ax.scatter(
        F[:, 0],
        F[:, 1],
        c="#1f77b4",
        s=55,
        edgecolor="white",
        label="Pareto-Optimal Front (NSGA-II)",
        zorder=2,
    )

    regime_styles = {
        "Low Energy": {"color": "#ff7f0e", "marker": "s", "offset": (8, 8)},
        "Balanced": {"color": "#2ca02c", "marker": "*", "offset": (10, -14)},
        "Low CO2": {"color": "#9467bd", "marker": "D", "offset": (-65, -18)},
    }

    for _, row in rep_df.iterrows():
        regime = str(row["Regime"])
        style = regime_styles.get(regime, {"color": "black", "marker": "o", "offset": (8, 8)})
        x_val = float(row[opt_result.energy_target])
        y_val = float(row[opt_result.co2_target])
        ax.scatter(
            x_val,
            y_val,
            c=style["color"],
            marker=style["marker"],
            s=180 if style["marker"] == "*" else 110,
            edgecolor="black",
            zorder=4,
            label=f"{regime} ({x_val:.1f} kWh/t, {y_val:.1f} kg/h)",
        )
        ax.annotate(
            regime,
            (x_val, y_val),
            textcoords="offset points",
            xytext=style["offset"],
            fontsize=9,
            fontweight="bold",
        )

    ax.scatter(
        plant_energy,
        plant_co2,
        c="#d62728",
        marker="X",
        s=150,
        edgecolor="black",
        zorder=5,
        label=f"Plant Median ({plant_energy:.1f} kWh/t, {plant_co2:.1f} kg/h)",
    )

    ax.set_title("Pareto Front: Energy Consumption vs CO2 Emissions")
    ax.set_xlabel("Energy Consumption (kWh/t-clinker)")
    ax.set_ylabel("CO2 Emissions (kg/h)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
