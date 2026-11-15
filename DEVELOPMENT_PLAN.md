# Step-by-Step Development Plan: Process Optimization for Sustainable Cement Production

This document outlines the architecture and phased execution plan to rebuild and refactor the single-script prototype ([`ml_model.py`](ml_model.py) and [`process_o/process.py`](process_o/process.py)) into a modular, reproducible, production-grade Machine Learning and Multi-Objective Optimization package aligned with the thesis in [`process optimization for sustainable cement production using machine learning.md`](process%20optimization%20for%20sustainable%20cement%20production%20using%20machine%20learning.md).

---

## 1. Audit of Current Codebase & Key Issues to Fix

Before restructuring, the rebuild will resolve several critical issues identified in [`ml_model.py`](ml_model.py) and [`process_o/process.py`](process_o/process.py):

1. **Incomplete Script & Broken Imports**: [`ml_model.py`](ml_model.py) is truncated mid-class at `ProcessOptimizer.__init__` and expects [`kiln_energy_dataset.csv`](process_o/kiln_energy_dataset.csv) to contain process columns (`Clinker Production Rate (tons/h)`, `Raw Meal Composition (% Limestone)`) before merging with [`kiln_co2_dataset.csv`](process_o/kiln_co2_dataset.csv).
2. **Data Leakage & Feature Alignment Bug**: In [`process_o/process.py`](process_o/process.py), `SN` (Serial Number) is not excluded from `all_features`, allowing the Random Forest model to train and optimize on row indices. Additionally, `energy_data` and `co2_data` must be joined on `SN` (or timestamp/batch ID) **before** computing cross-dataset chemical engineering features.
3. **Thermodynamic Inconsistency During NSGA-II Evaluation**: In `CementOptimizationProblem._evaluate()`, derived features (`Theoretical_CO2`, `Theoretical_Energy`, `Heat_Efficiency`, `O2_Deviation`) are treated as independent decision variables mutated directly by NSGA-II. Instead, NSGA-II should only mutate **controllable process & power variables**, and dynamically recompute the derived chemical engineering features inside `_evaluate()` so physical laws are respected on every generation.
4. **Dropped Categorical Fuel Feature**: [`kiln_co2_dataset.csv`](process_o/kiln_co2_dataset.csv) includes a `Fuel` column (`Coal`, `Gas`, `Oil`) that is currently dropped via `merged_data[col].dtype != 'object'`. Encoding fuel type directly fulfills **Chapter 5 Recommendation (iii)**.
5. **Reproducibility & Validation Pipeline**: The validation table metrics (`R²`, `RMSE`, `% Deviation` in Table 4.2) and Pareto-optimal solution selection (`Low Energy`, `Balanced`, `Low CO2` in Table 4.1) need automated calculation and export pipelines rather than manual console copy-pasting.

---

## 2. Target Project Directory Layout

The repository will follow a standard Python Data Science / ML package layout (`src/` layout with `pyproject.toml`):

```text
Process-Optimisation-for-sustainable-cement-production-with-machine-learning/
├── README.md                                 # Project overview, setup, and usage guide
├── DEVELOPMENT_PLAN.md                       # This step-by-step development roadmap
├── pyproject.toml                            # Dependencies, build system, ruff/pytest config
├── Makefile                                  # Shortcuts: make setup, make train, make optimize, make test
├── .gitignore                                # Ignore venv, __pycache__, artifacts, .env
│
├── configs/
│   └── default.yaml                          # Hyperparameters, thermodynamic constants, NSGA-II settings
│
├── data/
│   ├── raw/                                  # Immutable raw plant datasets
│   │   ├── kiln_energy_dataset.csv
│   │   └── kiln_co2_dataset.csv
│   └── processed/                            # Merged, cleaned, and feature-engineered datasets
│       └── kiln_merged_processed.parquet
│
├── docs/
│   └── thesis_report.md                      # Relocated and formatted project documentation
│
├── models/                                   # Serialized trained models & metadata (via joblib)
│   ├── energy_rf_model.joblib
│   ├── co2_rf_model.joblib
│   └── feature_bounds.json
│
├── notebooks/                                # Exploratory analysis & interactive reporting
│   ├── 01_eda_and_thermodynamics.ipynb
│   ├── 02_model_training_and_shap.ipynb
│   └── 03_pareto_optimization_analysis.ipynb
│
├── outputs/
│   ├── figures/                              # High-DPI publication-ready charts
│   │   ├── r2_performance_chart.png
│   │   ├── feature_importance_energy.png
│   │   ├── feature_importance_co2.png
│   │   └── pareto_front.png
│   └── tables/                               # Generated CSV/Markdown tables matching Chapter 4
│       ├── table_4_1_pareto_solutions.csv
│       └── table_4_2_model_validation.csv
│
├── src/
│   └── cement_opt/
│       ├── __init__.py
│       ├── config.py                         # Pydantic/dataclass configuration loader
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py                     # Load & validate raw CSV schemas, join on SN
│       │   └── preprocessing.py              # Deduplication, outlier handling, train/test split
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py                # Stoichiometric CO2, enthalpy balance, O2 deviation
│       ├── models/
│       │   ├── __init__.py
│       │   ├── trainer.py                    # RandomForestRegressor wrapper, cross-validation
│       │   ├── evaluator.py                  # R², RMSE, MAE, % Deviation against plant medians
│       │   └── registry.py                   # Save/load model artifacts and feature bounds
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── problem.py                    # Pymoo Problem over controllable variables only
│       │   ├── solver.py                     # NSGA-II runner with convergence tracking
│       │   └── recommender.py                # ProcessOptimizer: Knee-point & priority selection
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── plots.py                      # Pareto front, feature importance, parity plots
│       └── cli.py                            # Command-line interface (preprocess, train, optimize, report)
│
└── tests/
    ├── __init__.py
    ├── test_engineering_features.py          # Verify Eq. 3.13 - 3.17 calculations
    ├── test_model_pipeline.py                # Verify training, serialization, and R² checks
    └── test_optimization.py                  # Verify NSGA-II constraints (g(x) <= 0) & Pareto sorting
```

---

## 3. Phased Step-by-Step Implementation Plan

### Phase 1: Project Scaffolding & Environment Setup
**Goal:** Establish a clean, reproducible Python environment and directory structure without breaking existing files until migration is complete.

- [x] **Step 1.1: Initialize Packaging & Dependencies (`pyproject.toml`)**
  - Define project metadata (`cement-opt`), Python version (`>=3.10`), and core dependencies: `numpy`, `pandas`, `scikit-learn`, `pymoo`, `matplotlib`, `seaborn`, `joblib`, `pyyaml`, `typer` (or `argparse`).
  - Add development dependencies: `pytest`, `ruff`, `jupyter`.
- [x] **Step 1.2: Create Directory Structure & Migrate Assets**
  - Create `src/cement_opt/`, `data/raw/`, `data/processed/`, `configs/`, `models/`, `outputs/figures/`, `outputs/tables/`, and `tests/`.
  - Move [`process_o/kiln_energy_dataset.csv`](process_o/kiln_energy_dataset.csv) and [`process_o/kiln_co2_dataset.csv`](process_o/kiln_co2_dataset.csv) into `data/raw/`.
  - Move [`process optimization for sustainable cement production using machine learning.md`](process%20optimization%20for%20sustainable%20cement%20production%20using%20machine%20learning.md) into `docs/thesis_report.md` (or keep a clean reference in `README.md`).
- [x] **Step 1.3: Centralize Configuration (`configs/default.yaml` & `src/cement_opt/config.py`)**
  - Extract all magic numbers from the scripts into `configs/default.yaml`:
    - Stoichiometric $\text{CO}_2$ factor: `0.44` (Eq. 3.13)
    - Calcination enthalpy: `1780.0` kJ/kg-clinker (Eq. 3.14)
    - Optimal $\text{O}_2$ target: `4.0` % (Eq. 3.17)
    - Random Forest settings: `n_estimators: 100`, `max_depth: 8`, `test_size: 0.2`, `random_state: 42`
    - NSGA-II settings: `pop_size: 50`, `n_gen: 100`, `seed: 42`

---

### Phase 2: Data Ingestion, Merging & Chemical Engineering Feature Pipeline
**Goal:** Build a reliable data pipeline that merges the energy and $\text{CO}_2$ plant datasets properly and applies domain-specific chemical engineering transformations.

- [x] **Step 2.1: Schema Validation & Dataset Merger (`src/cement_opt/data/loader.py`)**
  - Load `kiln_energy_dataset.csv` and `kiln_co2_dataset.csv`.
  - Inner/outer join explicitly on `SN` and drop `SN` before feature selection so row indices never leak into model training.
  - Handle missing values, infinite values, and duplicate cyclic rows (with configurable option to retain full plant log or deduplicate).
- [x] **Step 2.2: Chemical Engineering Feature Transformer (`src/cement_opt/features/engineering.py`)**
  - Implement a scikit-learn compatible `BaseEstimator, TransformerMixin` class (`ChemicalEngineeringTransformer`) that computes:
    1. **Theoretical $\text{CO}_2$ (Eq. 3.13):**
       $$\text{T}_{\text{CO}_2} = \text{RMC} \times \text{CPR} \times 0.44$$
    2. **Theoretical Energy (Eq. 3.14–3.15):**
       $$\text{TE} = \frac{\text{CPR} \times 1000 \times 1780}{3600}$$
    3. **Heat Efficiency (Eq. 3.16):**
       $$\text{HE} = \frac{\text{TE}}{\text{FC}}$$
    4. **Combustion Efficiency / $\text{O}_2$ Deviation (Eq. 3.17):**
       $$\text{O}_2\text{ Deviation} = |\text{O}_2\text{ Level (\%)} - 4.0|$$
  - Add optional encoding for the `Fuel` column (`Coal`, `Gas`, `Oil`) so fuel carbon intensity can be modeled.

---

### Phase 3: Predictive Modeling & Validation Engine (Random Forest Regressor)
**Goal:** Train, evaluate, and persist the Energy Consumption and $\text{CO}_2$ Emissions prediction models with full validation metrics matching Chapter 3 & Chapter 4.

- [x] **Step 3.1: Model Wrapper & Training Pipeline (`src/cement_opt/models/trainer.py`)**
  - Refactor `CementProcessModel` to wrap `RandomForestRegressor` inside a scikit-learn `Pipeline` (combining `ChemicalEngineeringTransformer` + `RandomForestRegressor`).
  - Why this matters: Passing raw controllable variables into the pipeline automatically computes the engineered features prior to prediction, preventing train/inference skew and making NSGA-II evaluation trivial.
  - Record min/max bounds (`feature_mins`, `feature_maxs`) and baseline medians strictly for the controllable decision variables.
- [x] **Step 3.2: Comprehensive Evaluation & Deviation Analysis (`src/cement_opt/models/evaluator.py`)**
  - Compute test-set metrics: $R^2$ (Eq. 3.20), $\text{RMSE}$, $\text{MAE}$, and cross-validation scores.
  - Implement the **Plant Validation & Deviation Analysis** formulas from Section 3.2.3 to automatically generate **Table 4.2** (comparing Model Predicted Value, Optimized Value, and Plant Median Value with `% Deviation` and `RMSE`).
- [x] **Step 3.3: Model Serialization (`src/cement_opt/models/registry.py`)**
  - Save trained pipelines and feature bound metadata to `models/energy_rf_model.joblib` and `models/co2_rf_model.joblib`.

---

### Phase 4: Multi-Objective Optimization (NSGA-II) & Recommendation Engine
**Goal:** Formulate the multi-objective optimization problem with physical consistency and complete the unfinished `ProcessOptimizer` recommendation engine from [`ml_model.py`](ml_model.py).

- [x] **Step 4.1: Physically Consistent Optimization Problem (`src/cement_opt/optimization/problem.py`)**
  - Refactor `CementOptimizationProblem(Problem)` so decision variables $X$ consist **only** of controllable plant parameters (Kiln Temperature, Fuel Consumption, Raw Meal Composition, Clinker Production Rate, Oxygen Levels, Kiln Pressure, Air Flow Rate, and Power parameters).
  - Inside `_evaluate(self, X, out)`:
    - Recompute `Theoretical_CO2`, `Theoretical_Energy`, `Heat_Efficiency`, and `O2_Deviation` dynamically from candidate $X$ using `ChemicalEngineeringTransformer`.
    - Evaluate objectives:
      $$\min F(X) = [\text{EC}(X),\ \text{CO}_2\text{E}(X)]$$
    - Evaluate production constraint (Eq. 3.22):
      $$g(X) = \text{CuPR}_{\text{median}} - X_{\text{CPR}} \le 0$$
- [x] **Step 4.2: NSGA-II Solver Runner (`src/cement_opt/optimization/solver.py`)**
  - Configure `NSGA2(pop_size=50, eliminate_duplicates=True)` and execute `pymoo.optimize.minimize` with history enabled for convergence diagnostics.
- [x] **Step 4.3: Complete the `ProcessOptimizer` Recommendation Engine (`src/cement_opt/optimization/recommender.py`)**
  - Finish the truncated `ProcessOptimizer` class in [`ml_model.py`](ml_model.py) to:
    1. Sort the Pareto front by Energy Consumption and $\text{CO}_2$ Emissions.
    2. Automatically extract the three operating regimes highlighted in **Table 4.1**:
       - **Low Energy Regime** (minimum energy consumption)
       - **Balanced / Knee-Point Regime** (minimum normalized Euclidean distance to the utopia point)
       - **Low $\text{CO}_2$ Regime** (minimum $\text{CO}_2$ emission rate)
    3. Compute actionable set-point adjustments ($\Delta$ from current plant median operating state) for kiln operators.

---

### Phase 5: Automated Visualization, Reporting & CLI
**Goal:** Generate all thesis figures and tables with a single command.

- [x] **Step 5.1: Publication-Quality Visualizations (`src/cement_opt/visualization/plots.py`)**
  - Upgrade plotting functions from [`process_o/process.py`](process_o/process.py) with clean labels, tight layouts (`plt.tight_layout()`), so x-axis tick labels are never clipped (fixing the overlapping text in the original `feature_importance_*.png` images):
    - **Figure 4.1:** $R^2$ Parity / Performance comparison chart for Energy and $\text{CO}_2$ models.
    - **Figure 4.2 & 4.3:** Horizontal bar charts for Feature Importance (Energy & $\text{CO}_2$ models).
    - **Figure 4.4:** Annotated Pareto Front scatter plot highlighting the **Low Energy**, **Balanced**, and **Low $\text{CO}_2$** solutions alongside the baseline plant operating point.
- [x] **Step 5.2: Unified CLI Entrypoint (`src/cement_opt/cli.py`)**
  - Provide clean CLI commands:
    - `python -m cement_opt.cli preprocess` — Merges raw CSVs and outputs processed dataset.
    - `python -m cement_opt.cli train` — Trains RF models, saves artifacts to `models/`, and prints $R^2$/RMSE metrics.
    - `python -m cement_opt.cli optimize` — Runs NSGA-II, saves Table 4.1 & Table 4.2 to `outputs/tables/`, and generates all plots in `outputs/figures/`.
    - `python -m cement_opt.cli run-all` — Runs the complete end-to-end pipeline.

---

### Phase 6: Testing, Documentation & Legacy Cleanup
**Goal:** Ensure every equation and constraint is unit-tested and the repository is ready for portfolio/academic presentation.

- [x] **Step 6.1: Unit & Integration Tests (`tests/`)**
  - `test_engineering_features.py`: Verify Equations 3.13–3.17 against hand-calculated chemical engineering values.
  - `test_model_pipeline.py`: Ensure `SN` is excluded, no `NaN`/`inf` leaks occur, and $R^2$ exceeds threshold.
  - `test_optimization.py`: Assert all returned Pareto solutions satisfy $g(x) \le 0$ (Clinker Production Rate $\ge$ baseline median).
- [x] **Step 6.2: Update `README.md` & Remove Legacy Clutter**
  - Write a comprehensive `README.md` with project architecture, chemical engineering background, quickstart commands, and embedded links to generated results and charts.
  - Archive or remove the redundant `process_o/` folder and broken single-file [`ml_model.py`](ml_model.py) once the new `src/cement_opt/` pipeline is verified.
