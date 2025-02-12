# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# ========================================================================
# 1. DATA PREPROCESSING WITH CHEMICAL ENGINEERING FEATURES
# ========================================================================
def add_engineering_features(df):
    """Add domain-specific features using chemical engineering principles"""
    # Theoretical CO2 from limestone calcination (CaCO3 -> CaO + CO2)
    df['Theoretical_CO2'] = df['Raw Meal Composition (% Limestone)'] * df['Clinker Production Rate (tons/h)'] * 0.44
    
    # Heat balance calculations (simplified enthalpy balance)
    calcination_enthalpy = 1780  # kJ/kg-clinker
    df['Theoretical_Energy'] = (df['Clinker Production Rate (tons/h)'] * 1000 * calcination_enthalpy) / 3600  # kW
    df['Heat_Efficiency'] = df['Theoretical_Energy'] / df['Fuel Consumption (kg/h)']
    
    # Combustion efficiency indicator
    df['O2_Deviation'] = np.abs(df['Oxygen Levels (%)'] - 4.0)  # Optimal O2% target
    
    return df

# Load and preprocess datasets
energy_data = add_engineering_features(pd.read_csv('kiln_energy_dataset.csv'))
co2_data = add_engineering_features(pd.read_csv('kiln_co2_dataset.csv'))

# ========================================================================
# 2. MACHINE LEARNING MODEL TRAINING
# ========================================================================
class CementProcessModel:
    def __init__(self, target_var):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=8)
        self.target = target_var
        self.features = [
            'Kiln Temperature (°C)', 
            'Fuel Consumption (kg/h)',
            'Raw Meal Composition (% Limestone)',
            'Clinker Production Rate (tons/h)',
            'Oxygen Levels (%)',
            'Air Flow Rate (m³/h)',
            'Theoretical_CO2',
            'Heat_Efficiency',
            'O2_Deviation'
        ]
        
    def train(self, data):
        X = data[self.features]
        y = data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Save feature ranges for optimization constraints
        self.feature_mins = X.min().values
        self.feature_maxs = X.max().values
        
        print(f"{self.target} model trained. R²: {self.model.score(X_test, y_test):.2f}")

# Initialize and train models
energy_model = CementProcessModel('Energy Consumption (kWh/t-clinker)')
energy_model.train(energy_data)

co2_model = CementProcessModel('CO2 Emissions (kg/h)')
co2_model.train(co2_data)

# ========================================================================
# 3. MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
# ========================================================================
class CementOptimizationProblem(Problem):
    def __init__(self, energy_model, co2_model):
        super().__init__(n_var=len(energy_model.features),
                         n_obj=2,
                         n_constr=1,
                         xl=energy_model.feature_mins,
                         xu=energy_model.feature_maxs)
        
        self.energy_model = energy_model
        self.co2_model = co2_model
        self.current_production = energy_data['Clinker Production Rate (tons/h)'].median()
        
    def _evaluate(self, X, out, *args, **kwargs):
        # Predict objectives
        energy = self.energy_model.model.predict(X)
        co2 = self.co2_model.model.predict(X)
        
        # Constraint: Production rate >= current value (index 3 in features)
        g = self.current_production - X[:, 3]  # g <= 0 required
        
        out["F"] = np.column_stack([energy, co2])
        out["G"] = g.reshape(-1,1)

# Initialize optimization problem
problem = CementOptimizationProblem(energy_model, co2_model)
algorithm = NSGA2(pop_size=50)

# Run optimization
result = minimize(problem,
                  algorithm,
                  ('n_gen', 100),
                  seed=42,
                  verbose=False)

# ========================================================================
# 4. RECOMMENDATION ENGINE WITH PROCESS CONSTRAINTS
# ========================================================================
class ProcessOptimizer:
    def __init__(self, energy_model, co2_model, result):
        self.energy


