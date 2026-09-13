"""Model definitions and group-aware hyperparameter tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import N_CV_SPLITS, N_ITER_SEARCH, RANDOM_STATE


def baseline_linear() -> Pipeline:
    """Scaled linear regression baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def ridge_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def random_forest(params: dict[str, Any] | None = None) -> RandomForestRegressor:
    defaults = dict(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if params:
        defaults.update(params)
    return RandomForestRegressor(**defaults)


def xgboost_regressor(params: dict[str, Any] | None = None) -> XGBRegressor:
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)


def tune_random_forest(X, y, groups, n_iter: int = N_ITER_SEARCH) -> RandomizedSearchCV:
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    cv = GroupKFold(n_splits=N_CV_SPLITS)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    return search


def tune_xgboost(X, y, groups, n_iter: int = N_ITER_SEARCH) -> RandomizedSearchCV:
    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    cv = GroupKFold(n_splits=N_CV_SPLITS)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    return search
