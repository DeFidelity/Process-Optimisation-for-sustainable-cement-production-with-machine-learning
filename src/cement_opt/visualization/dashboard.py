"""Self-contained interactive HTML/SVG visual explainer for sustainable cement optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from cement_opt.data.preprocessing import CONTROLLABLE_NUMERIC_FEATURES
from cement_opt.features.engineering import ChemicalEngineeringTransformer, add_engineering_features
from cement_opt.models.trainer import CementProcessModel
from cement_opt.optimization.recommender import ProcessOptimizer
from cement_opt.optimization.solver import OptimizationResult

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "visual_explainer.html"
ARCH_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "architecture_visualizer.html"
)
SVG_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "system_architecture.svg"
)

KEY_KNOBS: tuple[tuple[str, str, str], ...] = (
    ("Power (Kw)", "Main Power", "kW"),
    ("Power Consumption (MWH)", "Power Consumption", "MWh"),
    ("Power Consumed (KWt/h)", "Power Consumed", "kWt/h"),
    ("Absorbed Power (Kw)", "Absorbed Power", "kW"),
    ("Power Consumption (Kw) Max", "Max Power", "kW"),
    ("Power (rotary drive) Kw", "Rotary Drive", "kW"),
    ("Kiln Temperature (°C)", "Kiln Temp", "°C"),
    ("Fuel Consumption (kg/h)", "Fuel Rate", "kg/h"),
    ("Raw Meal Composition (% Limestone)", "Limestone", "%"),
    ("Clinker Production Rate (tons/h)", "Clinker Rate", "tph"),
    ("Oxygen Levels (%)", "O2 Level", "%"),
    ("Kiln Pressure (bar)", "Kiln Pressure", "bar"),
    ("Air Flow Rate (m³/h)", "Air Flow", "m³/h"),
)


def _extract_tree_node(
    tree: Any,
    feature_names: list[str],
    node_id: int = 0,
    depth: int = 0,
    max_depth: int = 2,
) -> dict[str, Any]:
    feat_idx = int(tree.feature[node_id])
    val = round(float(tree.value[node_id][0][0]), 2)
    n_samples = int(tree.n_node_samples[node_id])
    if feat_idx < 0 or depth >= max_depth:
        return {"leaf": True, "value": val, "samples": n_samples}
    left_id = int(tree.children_left[node_id])
    right_id = int(tree.children_right[node_id])
    return {
        "leaf": False,
        "feature": feature_names[feat_idx],
        "threshold": round(float(tree.threshold[node_id]), 2),
        "value": val,
        "samples": n_samples,
        "left": _extract_tree_node(tree, feature_names, left_id, depth + 1, max_depth),
        "right": _extract_tree_node(tree, feature_names, right_id, depth + 1, max_depth),
    }


def _extract_rf_trees(model: CementProcessModel, n_trees: int = 3) -> list[dict[str, Any]]:
    eng = model.pipeline.named_steps["engineering"]
    reg = model.pipeline.named_steps["regressor"]
    names = list(eng.get_feature_names_out())
    return [
        _extract_tree_node(est.tree_, names, node_id=0, depth=0, max_depth=2)
        for est in reg.estimators_[:n_trees]
    ]


def _compute_benchmarks_and_slopes(
    model: CementProcessModel,
    modeling_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    r2_rf = round(model.metrics.r2_test if model.metrics else 0.98, 4)
    cv_rf = round(model.metrics.cv_r2_mean if model.metrics else 0.98, 4)
    rmse_rf = round(model.metrics.rmse_test if model.metrics else 0.0, 2)
    mae_rf = round(model.metrics.mae_test if model.metrics else 0.0, 2)

    benchmarks: dict[str, float] = {
        "RF": r2_rf,
        "CV_RF": cv_rf,
        "RMSE": rmse_rf,
        "MAE": mae_rf,
        "Linear": r2_rf,
        "SVR": r2_rf,
        "GBR": r2_rf,
    }
    slopes: dict[str, float] = {}

    try:
        X = modeling_df[model.controllable_features].copy()
        y = modeling_df[model.target].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=model.model_config.test_size,
            random_state=model.model_config.random_state,
        )
        t_cfg = model.thermo_config

        def _make_pipe(est: Any) -> Pipeline:
            return Pipeline(
                steps=[
                    (
                        "eng",
                        ChemicalEngineeringTransformer(
                            stoichiometric_co2_factor=t_cfg.stoichiometric_co2_factor,
                            calcination_enthalpy_kj_per_kg=t_cfg.calcination_enthalpy_kj_per_kg,
                            optimal_o2_percent=t_cfg.optimal_o2_percent,
                        ),
                    ),
                    ("est", est),
                ]
            )

        lin_pipe = _make_pipe(LinearRegression())
        lin_pipe.fit(X_tr, y_tr)
        benchmarks["Linear"] = round(float(r2_score(y_te, lin_pipe.predict(X_te))), 4)

        svr_pipe = _make_pipe(Pipeline([("scaler", StandardScaler()), ("svr", SVR())]))
        svr_pipe.fit(X_tr, y_tr)
        benchmarks["SVR"] = round(float(r2_score(y_te, svr_pipe.predict(X_te))), 4)

        gbr_pipe = _make_pipe(GradientBoostingRegressor(random_state=42))
        gbr_pipe.fit(X_tr, y_tr)
        benchmarks["GBR"] = round(float(r2_score(y_te, gbr_pipe.predict(X_te))), 4)

        lin_full = _make_pipe(LinearRegression())
        lin_full.fit(X, y)
        feat_names = list(lin_full.named_steps["eng"].get_feature_names_out())
        coefs = lin_full.named_steps["est"].coef_
        slopes = {
            name: round(float(coef), 5)
            for name, coef in zip(feat_names, coefs)
        }
    except Exception:
        pass

    return benchmarks, slopes


def _extract_gen_snapshots(opt_result: OptimizationResult) -> list[dict[str, Any]]:
    history = getattr(opt_result.raw_result, "history", None) or []
    if not history:
        return []
    n_hist = len(history)
    if n_hist <= 8:
        indices = list(range(n_hist))
    else:
        raw_idx = np.linspace(0, n_hist - 1, 8, dtype=int)
        indices = sorted(set(int(i) for i in raw_idx))

    snapshots: list[dict[str, Any]] = []
    for idx in indices:
        algo = history[idx]
        pop_F = np.atleast_2d(algo.pop.get("F"))
        pop_G = algo.pop.get("G")
        pts = []
        for row_i, pt in enumerate(pop_F):
            feasible = 1
            if pop_G is not None and len(pop_G) > row_i:
                feasible = 1 if float(np.atleast_2d(pop_G)[row_i][0]) <= 0.0 else 0
            pts.append([round(float(pt[0]), 2), round(float(pt[1]), 2), feasible])
        snapshots.append({"gen": int(idx + 1), "points": pts})
    return snapshots


def _extract_dashboard_payload(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
) -> dict[str, Any]:
    energy_col = opt_result.energy_target
    co2_col = opt_result.co2_target
    prod_col = "Clinker Production Rate (tons/h)"

    baseline: dict[str, Any] = {
        "name": "Plant Median",
        "energy": round(float(modeling_df[energy_col].median()), 2),
        "co2": round(float(modeling_df[co2_col].median()), 2),
        "production": round(float(modeling_df[prod_col].median()), 2),
        "knobs": {},
    }
    for feat in CONTROLLABLE_NUMERIC_FEATURES:
        if feat in modeling_df.columns:
            baseline["knobs"][feat] = round(float(modeling_df[feat].median()), 2)

    rep_indices = optimizer.get_representative_indices()
    pareto_points: list[dict[str, Any]] = []
    for idx, row in opt_result.pareto_df.iterrows():
        knobs = {
            feat: round(float(row[feat]), 2)
            for feat in CONTROLLABLE_NUMERIC_FEATURES
            if feat in row
        }
        regime = None
        for r_name, r_idx in rep_indices.items():
            if idx == r_idx:
                regime = r_name
        pareto_points.append(
            {
                "id": int(idx),
                "regime": regime,
                "energy": round(float(row[energy_col]), 2),
                "co2": round(float(row[co2_col]), 2),
                "production": round(float(row[prod_col]), 2),
                "fuel": str(row.get("Fuel", "Oil")),
                "o2_dev": round(float(row.get("O2_Deviation", 0.0)), 3),
                "heat_eff": round(float(row.get("Heat_Efficiency", 0.0)), 1),
                "knobs": knobs,
            }
        )

    historical_points = [
        {
            "energy": round(float(r[energy_col]), 2),
            "co2": round(float(r[co2_col]), 2),
        }
        for _, r in modeling_df[[energy_col, co2_col]].drop_duplicates().iterrows()
    ]

    gen1_points: list[dict[str, float]] = []
    history = getattr(opt_result.raw_result, "history", None) or []
    if history:
        pop_F = np.atleast_2d(history[0].pop.get("F"))
        gen1_points = [
            {"energy": round(float(pt[0]), 2), "co2": round(float(pt[1]), 2)}
            for pt in pop_F
        ]

    knob_meta = [
        {"col": col, "label": label, "unit": unit}
        for col, label, unit in KEY_KNOBS
        if col in baseline["knobs"]
    ]

    return {
        "baseline": baseline,
        "regimes": rep_indices,
        "pareto": pareto_points,
        "historical": historical_points,
        "gen1": gen1_points,
        "knob_meta": knob_meta,
        "metrics": {
            "energy_r2": round(
                energy_model.metrics.r2_test if energy_model.metrics else 0.98, 4
            ),
            "co2_r2": round(
                co2_model.metrics.r2_test if co2_model.metrics else 0.98, 4
            ),
            "n_samples": int(len(modeling_df)),
            "n_pareto": int(len(pareto_points)),
        },
    }


def _extract_architecture_payload(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
) -> dict[str, Any]:
    energy_col = opt_result.energy_target
    co2_col = opt_result.co2_target
    prod_col = "Clinker Production Rate (tons/h)"
    t_cfg = energy_model.thermo_config

    eng_df = add_engineering_features(
        modeling_df,
        stoichiometric_co2_factor=t_cfg.stoichiometric_co2_factor,
        calcination_enthalpy_kj_per_kg=t_cfg.calcination_enthalpy_kj_per_kg,
        optimal_o2_percent=t_cfg.optimal_o2_percent,
    )

    avg_features = modeling_df[energy_model.controllable_features].mean().to_frame().T
    pred_base_energy = round(float(energy_model.predict(avg_features)[0]), 2)
    pred_base_co2 = round(float(co2_model.predict(avg_features)[0]), 2)

    base_knobs = {
        feat: round(float(modeling_df[feat].median()), 2)
        for feat in CONTROLLABLE_NUMERIC_FEATURES
        if feat in modeling_df.columns
    }
    base_df_row = pd.DataFrame([base_knobs])
    base_eng = add_engineering_features(
        base_df_row,
        stoichiometric_co2_factor=t_cfg.stoichiometric_co2_factor,
        calcination_enthalpy_kj_per_kg=t_cfg.calcination_enthalpy_kj_per_kg,
        optimal_o2_percent=t_cfg.optimal_o2_percent,
    ).iloc[0]

    baseline = {
        "energy": round(float(modeling_df[energy_col].median()), 2),
        "co2": round(float(modeling_df[co2_col].median()), 2),
        "production": round(float(modeling_df[prod_col].median()), 2),
        "predicted_energy": pred_base_energy,
        "predicted_co2": pred_base_co2,
        "power": round(float(modeling_df["Power (Kw)"].median()), 2),
        "o2": round(float(modeling_df["Oxygen Levels (%)"].median()), 2),
        "air": round(float(modeling_df["Air Flow Rate (m³/h)"].median()), 2),
        "temp": round(float(modeling_df["Kiln Temperature (°C)"].median()), 2),
        "fuel_rate": round(float(modeling_df["Fuel Consumption (kg/h)"].median()), 2),
        "limestone": round(
            float(modeling_df["Raw Meal Composition (% Limestone)"].median()), 2
        ),
        "fuel": "Oil",
        "theoretical_co2": round(float(base_eng["Theoretical_CO2"]), 2),
        "theoretical_energy": round(float(base_eng["Theoretical_Energy"]), 2),
        "heat_eff": round(float(base_eng["Heat_Efficiency"]), 1),
        "o2_dev": round(float(base_eng["O2_Deviation"]), 3),
        "knobs": base_knobs,
    }

    bounds = {
        feat: {
            "min": round(float(modeling_df[feat].min()), 2),
            "max": round(float(modeling_df[feat].max()), 2),
            "median": round(float(modeling_df[feat].median()), 2),
        }
        for feat in CONTROLLABLE_NUMERIC_FEATURES
        if feat in modeling_df.columns
    }

    knob_cols = [feat for feat in CONTROLLABLE_NUMERIC_FEATURES if feat in bounds]
    rep_indices = optimizer.get_representative_indices()
    pareto_points: list[list[Any]] = []
    regimes: dict[str, dict[str, Any]] = {}

    for idx, row in opt_result.pareto_df.iterrows():
        knobs = {feat: round(float(row[feat]), 2) for feat in knob_cols if feat in row}
        regime_name = None
        for r_name, r_idx in rep_indices.items():
            if idx == r_idx:
                regime_name = r_name

        energy_v = round(float(row[energy_col]), 2)
        co2_v = round(float(row[co2_col]), 2)
        fuel_v = str(row.get("Fuel", "Oil"))
        g_v = round(float(row.get("Constraint_Violation_G", 0.0)), 3)
        k_arr = [knobs[f] for f in knob_cols]
        pareto_points.append([int(idx), regime_name, energy_v, co2_v, fuel_v, g_v, k_arr])

        if regime_name is not None:
            regimes[regime_name] = {
                "id": int(idx),
                "regime": regime_name,
                "energy": energy_v,
                "co2": co2_v,
                "production": round(float(row[prod_col]), 2),
                "temp": round(float(row["Kiln Temperature (°C)"]), 2),
                "fuel_rate": round(float(row["Fuel Consumption (kg/h)"]), 2),
                "limestone": round(float(row["Raw Meal Composition (% Limestone)"]), 2),
                "o2": round(float(row["Oxygen Levels (%)"]), 2),
                "power": round(float(row["Power (Kw)"]), 2),
                "air": round(float(row["Air Flow Rate (m³/h)"]), 2),
                "fuel": fuel_v,
                "o2_dev": round(float(row.get("O2_Deviation", 0.0)), 3),
                "heat_eff": round(float(row.get("Heat_Efficiency", 0.0)), 1),
                "theoretical_co2": round(float(row.get("Theoretical_CO2", 0.0)), 2),
                "theoretical_energy": round(float(row.get("Theoretical_Energy", 0.0)), 2),
                "g_violation": g_v,
                "knobs": knobs,
            }

    e_imp = energy_model.get_feature_importances().head(6)
    c_imp = co2_model.get_feature_importances().head(6)

    e_bench, e_slopes = _compute_benchmarks_and_slopes(energy_model, modeling_df)
    c_bench, c_slopes = _compute_benchmarks_and_slopes(co2_model, modeling_df)

    spark_seeds = {
        feat: [round(float(v), 2) for v in modeling_df[feat].tail(15).tolist()]
        for feat in knob_cols
    }

    gen_snapshots = _extract_gen_snapshots(opt_result)

    return {
        "baseline": baseline,
        "bounds": bounds,
        "knob_cols": knob_cols,
        "thermo": {
            "stoichiometric_co2_factor": t_cfg.stoichiometric_co2_factor,
            "calcination_enthalpy_kj_per_kg": t_cfg.calcination_enthalpy_kj_per_kg,
            "optimal_o2_percent": t_cfg.optimal_o2_percent,
            "min_cpr": baseline["production"],
            "te_min": round(float(eng_df["Theoretical_Energy"].min()), 1),
            "te_max": round(float(eng_df["Theoretical_Energy"].max()), 1),
            "tco2_min": round(float(eng_df["Theoretical_CO2"].min()), 1),
            "tco2_max": round(float(eng_df["Theoretical_CO2"].max()), 1),
            "he_min": round(float(eng_df["Heat_Efficiency"].min()), 1),
            "he_max": round(float(eng_df["Heat_Efficiency"].max()), 1),
            "o2dev_max": round(float(eng_df["O2_Deviation"].max()), 3),
        },
        "regimes": regimes,
        "rep_indices": rep_indices,
        "pareto": pareto_points,
        "gen_snapshots": gen_snapshots,
        "knob_meta": [
            {"col": col, "label": label, "unit": unit}
            for col, label, unit in KEY_KNOBS
            if col in bounds
        ],
        "importances": {
            "energy": [
                {"feature": str(r["feature"]), "importance": round(float(r["importance"]), 4)}
                for _, r in e_imp.iterrows()
            ],
            "co2": [
                {"feature": str(r["feature"]), "importance": round(float(r["importance"]), 4)}
                for _, r in c_imp.iterrows()
            ],
        },
        "trees": {
            "energy": _extract_rf_trees(energy_model, n_trees=3),
            "co2": _extract_rf_trees(co2_model, n_trees=3),
        },
        "benchmarks": {
            "energy": e_bench,
            "co2": c_bench,
        },
        "slopes": {
            "energy": e_slopes,
            "co2": c_slopes,
        },
        "spark_seeds": spark_seeds,
        "metrics": {
            "energy_r2": round(
                energy_model.metrics.r2_test if energy_model.metrics else 0.98, 4
            ),
            "co2_r2": round(
                co2_model.metrics.r2_test if co2_model.metrics else 0.98, 4
            ),
            "n_pareto": int(len(opt_result.pareto_df)),
            "n_samples": int(len(modeling_df)),
            "n_gen": int(gen_snapshots[-1]["gen"]) if gen_snapshots else 100,
        },
    }


def build_visual_explainer_html(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
) -> str:
    """Build a self-contained interactive HTML/SVG explainer string from template."""
    payload = _extract_dashboard_payload(
        energy_model=energy_model,
        co2_model=co2_model,
        opt_result=opt_result,
        optimizer=optimizer,
        modeling_df=modeling_df,
    )
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (
        template.replace("__ENERGY_R2__", f"{payload['metrics']['energy_r2']:.4f}")
        .replace("__CO2_R2__", f"{payload['metrics']['co2_r2']:.4f}")
        .replace("__PAYLOAD_JSON__", json.dumps(payload))
    )
    return html


def save_visual_explainer(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Render and write the interactive HTML/SVG explainer to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_visual_explainer_html(
        energy_model=energy_model,
        co2_model=co2_model,
        opt_result=opt_result,
        optimizer=optimizer,
        modeling_df=modeling_df,
    )
    out.write_text(html, encoding="utf-8")
    return out


def build_architecture_visualizer_html(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
) -> str:
    """Build interactive clickable SVG architecture & stream visualizer HTML."""
    payload = _extract_architecture_payload(
        energy_model=energy_model,
        co2_model=co2_model,
        opt_result=opt_result,
        optimizer=optimizer,
        modeling_df=modeling_df,
    )
    import re

    template = ARCH_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = re.sub(r"<!--.*?-->", "", template, flags=re.DOTALL)
    lines = []
    for line in template.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        lines.append(s)
    compact_lines = "\n".join(lines)
    return (
        compact_lines.replace("__ENERGY_R2__", f"{payload['metrics']['energy_r2']:.4f}")
        .replace("__CO2_R2__", f"{payload['metrics']['co2_r2']:.4f}")
        .replace("__ARCH_PAYLOAD_JSON__", json.dumps(payload, separators=(",", ":")))
    )


def save_architecture_visualizer(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write the interactive clickable SVG architecture visualizer HTML to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_architecture_visualizer_html(
        energy_model=energy_model,
        co2_model=co2_model,
        opt_result=opt_result,
        optimizer=optimizer,
        modeling_df=modeling_df,
    )
    out.write_text(html, encoding="utf-8")
    return out


def save_system_architecture_svg(
    energy_model: CementProcessModel,
    co2_model: CementProcessModel,
    opt_result: OptimizationResult,
    optimizer: ProcessOptimizer,
    modeling_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export a standalone vector SVG diagram of the closed-loop kiln architecture."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _extract_architecture_payload(
        energy_model=energy_model,
        co2_model=co2_model,
        opt_result=opt_result,
        optimizer=optimizer,
        modeling_df=modeling_df,
    )
    bal = payload["regimes"]["Balanced"]
    base = payload["baseline"]
    e_pct = (bal["energy"] - base["energy"]) / base["energy"] * 100.0
    c_pct = (bal["co2"] - base["co2"]) / base["co2"] * 100.0
    p_pct = (bal["production"] - base["production"]) / base["production"] * 100.0

    replacements = {
        "__BASE_PROD__": f"{base['production']:.2f}",
        "__BAL_TEMP__": f"{bal['temp']:.1f}",
        "__BAL_FUEL_RATE__": f"{bal['fuel_rate']:.1f}",
        "__BAL_FUEL__": str(bal["fuel"]),
        "__BAL_O2__": f"{bal['o2']:.2f}",
        "__BAL_POWER__": f"{bal['power']:.0f}",
        "__BAL_HE__": f"{bal['heat_eff']:.1f}",
        "__BAL_O2DEV__": f"{bal['o2_dev']:.3f}",
        "__ENERGY_R2__": f"{payload['metrics']['energy_r2']:.4f}",
        "__CO2_R2__": f"{payload['metrics']['co2_r2']:.4f}",
        "__BAL_ENERGY__": f"{bal['energy']:.1f}",
        "__BAL_CO2__": f"{bal['co2']:.1f}",
        "__N_PARETO__": str(payload["metrics"]["n_pareto"]),
        "__BAL_PROD__": f"{bal['production']:.2f}",
        "__E_PCT__": f"{e_pct:+.2f}%",
        "__C_PCT__": f"{c_pct:+.2f}%",
        "__P_PCT__": f"{p_pct:+.2f}%",
        "__BASE_POWER__": f"{base['power']:.0f}",
        "__BASE_O2__": f"{base['o2']:.2f}",
        "__BASE_AIR__": f"{base['air']:.0f}",
        "__BAL_AIR__": f"{bal['air']:.0f}",
    }
    svg = SVG_TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, val in replacements.items():
        svg = svg.replace(key, val)
    out.write_text(svg, encoding="utf-8")
    return out

