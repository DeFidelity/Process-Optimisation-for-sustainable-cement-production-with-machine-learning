# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================================
# 1. DATA PREPROCESSING WITH CHEMICAL ENGINEERING FEATURES
# ========================================================================
def add_engineering_features(df):
    """Add domain-specific features using chemical engineering principles."""

    # Theoretical CO2 (only if the necessary columns are present)
    if 'Raw Meal Composition (% Limestone)' in df.columns and 'Clinker Production Rate (tons/h)' in df.columns:
        df['Theoretical_CO2'] = df['Raw Meal Composition (% Limestone)'] * df['Clinker Production Rate (tons/h)'] * 0.44

    # Heat balance calculations (only if necessary columns are present)
    if 'Clinker Production Rate (tons/h)' in df.columns and 'Fuel Consumption (kg/h)' in df.columns:
        calcination_enthalpy = 1780  # kJ/kg-clinker
        df['Theoretical_Energy'] = (df['Clinker Production Rate (tons/h)'] * 1000 * calcination_enthalpy) / 3600  # kW
        df['Heat_Efficiency'] = np.where(df['Fuel Consumption (kg/h)'] != 0, df['Theoretical_Energy'] / df['Fuel Consumption (kg/h)'], 0)

    # Combustion efficiency indicator (only if the column is present)
    if 'Oxygen Levels (%)' in df.columns:
        df['O2_Deviation'] = np.abs(df['Oxygen Levels (%)'] - 4.0)  # Optimal O2% target
    return df


# Load datasets
energy_data = pd.read_csv('kiln_energy_dataset.csv')
co2_data = pd.read_csv('kiln_co2_dataset.csv')

# Add engineering features
energy_data = add_engineering_features(energy_data)
co2_data = add_engineering_features(co2_data)

# Combine the DataFrames *before* defining the feature list
# Use a 'outer' merge to keep all columns.  Fill NaN values appropriately.
merged_data = pd.merge(energy_data, co2_data, how='outer')

# Create a unified feature list *after* merging
all_features = [col for col in merged_data.columns if col not in ['Energy Consumption (kWh/t-clinker)', 'CO2 Emissions (kg/h)'] and merged_data[col].dtype != 'object']

#Select columns using all_features
merged_data = merged_data[all_features + ['Energy Consumption (kWh/t-clinker)','CO2 Emissions (kg/h)']]

# Drop rows with NaN or infinite values (do this *after* merging and feature engineering)
merged_data = merged_data.replace([np.inf, -np.inf], np.nan).dropna()

# Separate the merged data back into energy and co2 data for model training
energy_data = merged_data[merged_data['Energy Consumption (kWh/t-clinker)'].notna()].copy()
co2_data = merged_data[merged_data['CO2 Emissions (kg/h)'].notna()].copy()

# ========================================================================
# 2. MACHINE LEARNING MODEL TRAINING
# ========================================================================
class CementProcessModel:
    def __init__(self, target_var, features):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        self.target = target_var
        self.features = features

    def train(self, data):

        X = data[self.features]
        y = data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Save feature ranges
        self.feature_mins = X.min().values
        self.feature_maxs = X.max().values

        print(f"{self.target} model trained.  R²: {self.model.score(X_test, y_test):.2f}")


# Initialize and train models. Use the *unified* feature list.
energy_model = CementProcessModel('Energy Consumption (kWh/t-clinker)', all_features)
energy_model.train(energy_data)

co2_model = CementProcessModel('CO2 Emissions (kg/h)', all_features)
co2_model.train(co2_data)



# ========================================================================
# 3. MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
# ========================================================================
class CementOptimizationProblem(Problem):
    def __init__(self, energy_model, co2_model):
        super().__init__(n_var=len(energy_model.features),  # Use the unified feature length
                         n_obj=2,
                         n_constr=1,
                         xl=energy_model.feature_mins,  # Use mins/maxs from energy model (they should now be aligned)
                         xu=energy_model.feature_maxs)

        self.energy_model = energy_model
        self.co2_model = co2_model
        #Production constraint
        if 'Clinker Production Rate (tons/h)' in energy_data.columns:
          self.current_production = energy_data['Clinker Production Rate (tons/h)'].median()
        elif 'Clinker Production Rate (tons/h)' in co2_data.columns:
          self.current_production = co2_data['Clinker Production Rate (tons/h)'].median()
        else:
          self.current_production = 100

    def _evaluate(self, X, out, *args, **kwargs):
        # Create DataFrames with the correct feature names for prediction
        X_energy = pd.DataFrame(X, columns=self.energy_model.features)
        X_co2 = pd.DataFrame(X, columns=self.co2_model.features)
        # Predict objectives
        energy = self.energy_model.model.predict(X_energy)
        co2 = self.co2_model.model.predict(X_co2)

        # Constraint: Production Rate
        if 'Clinker Production Rate (tons/h)' in self.energy_model.features:
          g = self.current_production - X[:, self.energy_model.features.index('Clinker Production Rate (tons/h)')]
        else:
          g = np.zeros(len(X))

        out["F"] = np.column_stack([energy, co2])
        out["G"] = g.reshape(-1, 1)



# Initialize optimization problem
problem = CementOptimizationProblem(energy_model, co2_model)
algorithm = NSGA2(pop_size=50, eliminate_duplicates=True)

# Run optimization
result = minimize(problem,
                  algorithm,
                  ('n_gen', 100),
                  seed=42,
                  verbose=False)

# ========================================================================
# 4. VISUALIZATION OF RESULTS
# ========================================================================
def plot_pareto_front(result):
    """Plot the Pareto front and save to a file."""
    F = result.F
    plt.figure(figsize=(10, 6))
    plt.scatter(F[:, 0], F[:, 1], c='blue', label='Pareto Front')
    plt.title('Pareto Front: Energy Consumption vs CO2 Emissions')
    plt.xlabel('Energy Consumption (kWh/t-clinker)')
    plt.ylabel('CO2 Emissions (kg/h)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_front.png')  # Save the plot
    # plt.show()  # Optionally keep this if you *also* have a display

def plot_feature_importance(model, title):
    """Plot feature importance and save to a file."""
    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance: {title}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [model.features[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.savefig(f'feature_importance_{title.replace(" ", "_")}.png')  # Save the plot
    # plt.show() # Optionally keep

# Plot feature importance
plot_pareto_front(result)
plot_feature_importance(energy_model, 'Energy Consumption Model')
plot_feature_importance(co2_model, 'CO2 Emissions Model')


# Save results
pareto_front_results = pd.DataFrame(result.F, columns=['Energy Consumption (kWh/t-clinker)', 'CO2 Emissions (kg/h)'])
pareto_front_results.to_csv('pareto_front_results.csv', index=False)
print("Pareto front results saved to 'pareto_front_results.csv'.")


# Create a DataFrame for easier handling
pareto_df = pd.DataFrame(result.X, columns=all_features)
pareto_df['Energy Consumption (kWh/t-clinker)'] = result.F[:, 0]
pareto_df['CO2 Emissions (kg/h)'] = result.F[:, 1]

# Select representative solutions (example - adjust indices as needed)
# Choose indices that visually represent different points on your Pareto front.
low_energy_solution = pareto_df.iloc[0]  # Example: First solution
balanced_solution = pareto_df.iloc[len(pareto_df) // 2]  # Example: Middle solution
low_co2_solution = pareto_df.iloc[-1]  # Example: Last solution

# Create the table for your report
table_data = pd.DataFrame([low_energy_solution, balanced_solution, low_co2_solution])
print(table_data) #This can now be directly copy and pasted.