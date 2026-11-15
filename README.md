# Process Optimisation for Sustainable Cement Production with Machine Learning

**Research work conducted by:** Amusat Haki Adeyemi  
**Institution:** Department of Chemical/Petrochemical Engineering, University of Calabar  

A modular machine learning and multi-objective optimization framework for sustainable cement kiln operations. The framework integrates chemical engineering thermodynamics, **Random Forest Regression** surrogate modeling, and **NSGA-II** multi-objective evolutionary optimization to simultaneously minimize **Energy Consumption (kWh/t-clinker)** and **CO₂ Emissions (kg/h)** while maintaining or improving **Clinker Production Rate (tons/h)**.

---

## Key Verified Results

### Predictive Model Performance
- **Energy Consumption Random Forest:** Test $R^2 = 0.9757$, Test $\text{RMSE} = 11.94\text{ kWh/t-clinker}$, 5-Fold CV $R^2 = 0.9843 \pm 0.0047$
- **$\text{CO}_2$ Emissions Random Forest:** Test $R^2 = 0.9762$, Test $\text{RMSE} = 11.68\text{ kg/h}$, 5-Fold CV $R^2 = 0.9474 \pm 0.0703$

### Validation & Optimization Comparison (Table 4.2)

| Parameter | Model Predicted Value | Model Deviation (%) | Test RMSE | Balanced Optimized Value | Optimized Deviation (%) | Plant Median Value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Energy Consumption (kWh/t-clinker) | 1165.42 | +0.39% | 11.9365 | 1031.26 | **-11.16%** | 1160.87 |
| $\text{CO}_2$ Emissions (kg/h) | 1006.19 | +3.36% | 11.6775 | 960.25 | **-1.36%** | 973.45 |
| Clinker Production Rate (tons/h) | 91.80 | 0.00% | 0.0000 | 93.11 | **+1.43%** | 91.80 |

---

## Project Architecture

```text
├── configs/
│   └── default.yaml                 # Thermodynamic constants, RF & NSGA-II hyperparameters
├── data/
│   ├── raw/                         # Immutable Ashaka Cement Plant datasets
│   └── processed/                   # Merged & cleaned modeling dataset (SN leakage removed)
├── docs/
│   └── thesis_report.md             # Full B.Eng thesis documentation
├── models/                          # Serialized Random Forest pipelines & metadata (.joblib/.json)
├── notebooks/                       # Interactive EDA, model evaluation & Pareto analysis notebooks
├── outputs/
│   ├── figures/                     # Publication-ready charts (Figures 4.1–4.4)
│   └── tables/                      # Automated Table 4.1, Table 4.2 & setpoint deltas
├── src/cement_opt/
│   ├── config.py                    # YAML configuration loader
│   ├── data/                        # Schema validation, SN merge & fuel encoding
│   ├── features/                    # ChemicalEngineeringTransformer (Eq. 3.13–3.17)
│   ├── models/                      # RandomForest pipeline, CV metrics & Table 4.2 validator
│   ├── optimization/                # Thermodynamically consistent NSGA-II & ProcessOptimizer
│   ├── visualization/               # R² parity, feature importance & Pareto front plots
│   └── cli.py                       # Unified CLI (`preprocess`, `train`, `optimize`, `run-all`)
└── tests/                           # Unit & integration test suite (pytest)
```

---

## Quickstart

### 1. Environment Setup
```bash
make setup
# or manually:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Run the Full End-to-End Pipeline
```bash
make run-all
# or via installed CLI:
.venv/bin/cement-opt run-all
```

Individual stages can also be executed separately:
```bash
.venv/bin/cement-opt preprocess
.venv/bin/cement-opt train
.venv/bin/cement-opt optimize
```

### 3. Run Tests & Linter
```bash
make test
make lint
```

---

## Chemical Engineering Integration

Before model training and during every NSGA-II candidate evaluation, the scikit-learn `ChemicalEngineeringTransformer` computes domain-specific features directly from controllable kiln variables:

1. **Theoretical $\text{CO}_2$ (Eq. 3.13):** $\text{T}_{\text{CO}_2} = \text{RMC (\% Limestone)} \times \text{CPR (tons/h)} \times 0.44$
2. **Theoretical Energy (Eq. 3.14–3.15):** $\text{TE} = \frac{\text{CPR} \times 1000 \times 1780}{3600}$ (kW)
3. **Heat Efficiency (Eq. 3.16):** $\text{HE} = \frac{\text{TE}}{\text{Fuel Consumption (kg/h)}}$
4. **Combustion Efficiency ($\text{O}_2$ Deviation, Eq. 3.17):** $|\text{O}_2 (\%) - 4.0|$
