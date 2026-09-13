"""Regression metrics including NASA asymmetric RUL score."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def nasa_score(y_true, y_pred) -> float:
    """
    NASA PHM'08 asymmetric scoring function.

    d = y_pred - y_true
    Early prediction (d < 0): exp(-d/13) - 1
    Late prediction  (d >= 0): exp(d/10) - 1
    Summed over samples (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NASA_score": nasa_score(y_true, y_pred),
    }
