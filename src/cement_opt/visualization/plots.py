"""Publication-ready matplotlib/seaborn figures matching Chapter 4 & visual explainers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cement_opt.data.preprocessing import FUEL_CATEGORIES
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


def plot_setpoint_deltas(
    optimizer: ProcessOptimizer,
    output_path: str | Path,
    regime: str = "Balanced",
) -> Path:
    """Operator control-knob delta chart showing % change from Plant Median to Optimal."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rec_df = optimizer.recommend_adjustments(regime=regime).copy()
    rec_col = f"Recommended_{regime.replace(' ', '_')}"
    rec_df = rec_df.sort_values("Change_%", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.48 * len(rec_df))))
    colors = ["#1f77b4" if val < 0 else "#2ca02c" for val in rec_df["Change_%"]]

    ax.barh(rec_df["Parameter"], rec_df["Change_%"], color=colors, height=0.65, alpha=0.9)
    ax.axvline(0.0, color="#333333", linewidth=1.2)

    min_pct = float(rec_df["Change_%"].min())
    max_pct = float(rec_df["Change_%"].max())
    span = max(abs(min_pct), abs(max_pct), 5.0)
    ax.set_xlim(min_pct - span * 0.45, max_pct + span * 0.45)

    for idx, row in rec_df.iterrows():
        pct = float(row["Change_%"])
        base = float(row["Plant_Median"])
        rec = float(row[rec_col])
        label = f" {pct:+.2f}%  ({base:,.1f} -> {rec:,.1f})"
        if pct >= 0:
            ax.text(pct + 0.3, idx, label, va="center", ha="left", fontsize=8.5)
        else:
            ax.text(pct - 0.3, idx, f"({base:,.1f} -> {rec:,.1f})  {pct:+.2f}% ",
                    va="center", ha="right", fontsize=8.5)

    ax.set_xlabel("Recommended Adjustment vs. Plant Median (%)")
    ax.set_title(
        f"Kiln Operator Control-Knob Adjustments ({regime} Optimum vs. Plant Median)"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regime_comparison(
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Side-by-side comparison of Plant Median vs. all 3 Pareto regimes across key metrics."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rep_df = optimizer.get_representative_solutions()
    energy_col = optimizer.energy_col
    co2_col = optimizer.co2_col
    prod_col = "Clinker Production Rate (tons/h)"

    baseline = {
        "Regime": "Plant Median",
        energy_col: float(modeling_df[energy_col].median()),
        co2_col: float(modeling_df[co2_col].median()),
        prod_col: float(modeling_df[prod_col].median()),
    }
    comp_df = pd.concat([pd.DataFrame([baseline]), rep_df], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    specs = [
        (energy_col, "Energy Consumption\n(kWh/t-clinker — Lower is Better)", "{:.1f}"),
        (co2_col, "CO2 Emissions\n(kg/h — Lower is Better)", "{:.1f}"),
        (prod_col, "Clinker Production Rate\n(tons/h — Higher/Equal is Better)", "{:.2f}"),
    ]

    for ax, (col, title, fmt) in zip(axes, specs):
        vals = comp_df[col].to_numpy(dtype=float)
        base_val = vals[0]
        bars = ax.bar(comp_df["Regime"], vals, color=palette, width=0.58)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", rotation=18)
        y_min = min(vals) * 0.90
        y_max = max(vals) * 1.07
        ax.set_ylim(y_min, y_max)

        for idx, (bar, val) in enumerate(zip(bars, vals)):
            pct = ((val - base_val) / base_val * 100.0) if base_val else 0.0
            suffix = "" if idx == 0 else f"\n({pct:+.1f}%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + (y_max - y_min) * 0.015,
                fmt.format(val) + suffix,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if idx == 2 else "normal",
            )

    fig.suptitle(
        "Before vs. After: Historical Plant Median vs. NSGA-II Operating Regimes",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_nsga2_convergence(
    opt_result: OptimizationResult,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """4-panel snapshot showing NSGA-II evolving from Gen 1 to the final Pareto frontier."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    history = getattr(opt_result.raw_result, "history", None) or []
    if not history:
        raise ValueError("NSGA-II result does not contain optimization history.")

    total_gens = len(history)
    gen_indices = sorted({0, max(0, min(9, total_gens - 1)), total_gens // 2 - 1, total_gens - 1})
    while len(gen_indices) < 4:
        gen_indices.append(total_gens - 1)
    gen_indices = gen_indices[:4]

    hist_energy = modeling_df[opt_result.energy_target].to_numpy(dtype=float)
    hist_co2 = modeling_df[opt_result.co2_target].to_numpy(dtype=float)
    med_energy = float(np.median(hist_energy))
    med_co2 = float(np.median(hist_co2))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    gen_colors = ["#ff7f0e", "#17becf", "#9467bd", "#2ca02c"]

    for ax, gen_idx, col in zip(axes_flat, gen_indices, gen_colors):
        pop_F = history[gen_idx].pop.get("F")
        ax.scatter(
            hist_energy,
            hist_co2,
            color="#b0b0b0",
            alpha=0.45,
            s=32,
            label="Historical Plant Records",
        )
        ax.scatter(
            med_energy,
            med_co2,
            color="#d62728",
            marker="X",
            s=115,
            edgecolor="black",
            zorder=4,
            label="Plant Median",
        )
        ax.scatter(
            pop_F[:, 0],
            pop_F[:, 1],
            color=col,
            edgecolor="black",
            linewidth=0.5,
            s=48,
            zorder=5,
            label=f"NSGA-II Population (Gen {gen_idx + 1})",
        )
        ax.set_title(f"Generation {gen_idx + 1} of {total_gens}", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.45)

    axes[1, 0].set_xlabel("Energy Consumption (kWh/t-clinker)")
    axes[1, 1].set_xlabel("Energy Consumption (kWh/t-clinker)")
    axes[0, 0].set_ylabel("CO2 Emissions (kg/h)")
    axes[1, 0].set_ylabel("CO2 Emissions (kg/h)")

    fig.suptitle(
        "How NSGA-II Discovers Optimal Kiln Setpoints (Evolution Toward Bottom-Left)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_physics_sensitivity(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
    optimal_o2_percent: float = 4.0,
) -> Path:
    """2-panel chemical engineering explainer: O2 combustion target & Production-Limestone map."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    medians = modeling_df[energy_model.controllable_features].median()

    # Panel A: Sweep Oxygen Levels (%) across observed range for each Fuel type
    o2_col = "Oxygen Levels (%)"
    o2_grid = np.linspace(
        float(modeling_df[o2_col].min()),
        float(modeling_df[o2_col].max()),
        60,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fuel_colors = {"Coal": "#374151", "Gas": "#0284c7", "Oil": "#d97706"}
    has_fuel = all(f"Fuel_{f}" in energy_model.controllable_features for f in FUEL_CATEGORIES)

    if has_fuel:
        for fuel in FUEL_CATEGORIES:
            sweep_df = pd.DataFrame([medians.to_dict()] * len(o2_grid))
            sweep_df[o2_col] = o2_grid
            for f in FUEL_CATEGORIES:
                sweep_df[f"Fuel_{f}"] = 1.0 if f == fuel else 0.0
            pred_co2 = co2_model.predict(sweep_df)
            axes[0].plot(
                o2_grid,
                pred_co2,
                color=fuel_colors[fuel],
                linewidth=2.2,
                label=f"Fuel: {fuel}",
            )
    else:
        sweep_df = pd.DataFrame([medians.to_dict()] * len(o2_grid))
        sweep_df[o2_col] = o2_grid
        axes[0].plot(o2_grid, co2_model.predict(sweep_df), color="#2ca02c", linewidth=2.2)

    axes[0].axvline(
        optimal_o2_percent,
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Optimal O2 Target ({optimal_o2_percent:.1f}%, Eq. 3.17)",
    )
    axes[0].set_xlabel("Oxygen Levels (%)")
    axes[0].set_ylabel("Predicted CO2 Emissions (kg/h)")
    axes[0].set_title("Combustion O2 & Fuel-Type Sensitivity (at Median Settings)")
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.45)

    # Panel B: 2D Heatmap of Clinker Production Rate vs Raw Meal Limestone %
    cpr_col = "Clinker Production Rate (tons/h)"
    rmc_col = "Raw Meal Composition (% Limestone)"
    cpr_vals = np.linspace(float(modeling_df[cpr_col].min()), float(modeling_df[cpr_col].max()), 35)
    rmc_vals = np.linspace(float(modeling_df[rmc_col].min()), float(modeling_df[rmc_col].max()), 35)

    cc, rr = np.meshgrid(cpr_vals, rmc_vals)
    grid_df = pd.DataFrame([medians.to_dict()] * cc.size)
    grid_df[cpr_col] = cc.ravel()
    grid_df[rmc_col] = rr.ravel()

    z_co2 = co2_model.predict(grid_df).reshape(cc.shape)
    contour = axes[1].contourf(cc, rr, z_co2, levels=16, cmap="YlOrRd")
    cbar = fig.colorbar(contour, ax=axes[1])
    cbar.set_label("Predicted CO2 Emissions (kg/h)")

    axes[1].axvline(
        float(modeling_df[cpr_col].median()),
        color="#1d4ed8",
        linestyle="--",
        linewidth=1.5,
        label=f"Min Production Constraint ({modeling_df[cpr_col].median():.1f} tph)",
    )
    axes[1].set_xlabel("Clinker Production Rate (tons/h)")
    axes[1].set_ylabel("Raw Meal Composition (% Limestone)")
    axes[1].set_title("Stoichiometric Calcination Response Surface (Eq. 3.13)")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
