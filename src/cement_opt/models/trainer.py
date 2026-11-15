"""Random Forest regression pipeline wrapping chemical engineering feature transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from cement_opt.config import ModelConfig, ThermodynamicsConfig
from cement_opt.features.engineering import ChemicalEngineeringTransformer
from cement_opt.models.evaluator import evaluate_predictions


@dataclass(frozen=True)
class TrainingMetrics:
    target: str
    r2_test: float
    rmse_test: float
    mae_test: float
    cv_r2_mean: float
    cv_r2_std: float
    n_train: int
    n_test: int


class CementProcessModel:
    """Trains and serves a RandomForestRegressor pipeline for a cement kiln target."""

    def __init__(
        self,
        target_var: str,
        controllable_features: list[str],
        model_config: ModelConfig,
        thermo_config: ThermodynamicsConfig,
    ) -> None:
        self.target = target_var
        self.controllable_features = list(controllable_features)
        self.model_config = model_config
        self.thermo_config = thermo_config

        self.pipeline = Pipeline(
            steps=[
                (
                    "engineering",
                    ChemicalEngineeringTransformer(
                        stoichiometric_co2_factor=thermo_config.stoichiometric_co2_factor,
                        calcination_enthalpy_kj_per_kg=thermo_config.calcination_enthalpy_kj_per_kg,
                        optimal_o2_percent=thermo_config.optimal_o2_percent,
                    ),
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=model_config.n_estimators,
                        max_depth=model_config.max_depth,
                        random_state=model_config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self.feature_mins: np.ndarray | None = None
        self.feature_maxs: np.ndarray | None = None
        self.feature_medians: pd.Series | None = None
        self.feature_means: pd.Series | None = None
        self.metrics: TrainingMetrics | None = None
        self.y_test_: pd.Series | None = None
        self.y_pred_: np.ndarray | None = None

    def train(self, data: pd.DataFrame) -> TrainingMetrics:
        """Split data, fit pipeline, compute holdout + CV metrics, and record bounds."""
        X = data[self.controllable_features].copy()
        y = data[self.target].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        eval_scores = evaluate_predictions(y_test.to_numpy(), y_pred)

        n_splits = min(self.model_config.cv_folds, len(X))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.model_config.random_state)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="r2")

        self.feature_mins = X.min().to_numpy(dtype=float)
        self.feature_maxs = X.max().to_numpy(dtype=float)
        self.feature_medians = X.median()
        self.feature_means = X.mean()
        self.y_test_ = y_test
        self.y_pred_ = y_pred

        self.metrics = TrainingMetrics(
            target=self.target,
            r2_test=eval_scores["r2"],
            rmse_test=eval_scores["rmse"],
            mae_test=eval_scores["mae"],
            cv_r2_mean=float(np.mean(cv_scores)),
            cv_r2_std=float(np.std(cv_scores)),
            n_train=len(X_train),
            n_test=len(X_test),
        )
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target from controllable features (auto-computes engineered features)."""
        return self.pipeline.predict(X[self.controllable_features])

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances sorted descending across all model features."""
        eng = self.pipeline.named_steps["engineering"]
        reg = self.pipeline.named_steps["regressor"]
        names = list(eng.get_feature_names_out())
        importances = reg.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
