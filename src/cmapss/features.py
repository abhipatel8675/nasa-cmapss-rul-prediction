"""RUL labeling and causal rolling / window feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import CleaningReport
from .config import ROLLING_WINDOW, RUL_CAP


def add_rul_labels(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """
    Add uncapped RUL and optional piecewise-capped RUL for training.

    RUL_uncapped = max_cycle(unit) - cycle
    RUL = min(RUL_uncapped, cap) when cap is set.
    """
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["RUL_uncapped"] = max_cycle - out["cycle"]
    if cap is None:
        out["RUL"] = out["RUL_uncapped"]
    else:
        out["RUL"] = out["RUL_uncapped"].clip(upper=cap)
    return out


def _rolling_slope(values: np.ndarray) -> float:
    """OLS slope of values vs index; NaN-safe for short windows."""
    n = len(values)
    if n < 2 or np.any(~np.isfinite(values)):
        return 0.0
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    y = values.astype(float)
    y = y - y.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Causal rolling stats per engine: mean, std, slope over the past `window` cycles.

    Uses min_periods=1 and shift-free past-including-current windows so no future
    cycles leak into features.
    """
    out = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = out.groupby("unit_id", group_keys=False)

    for col in feature_cols:
        roll = grouped[col].rolling(window=window, min_periods=1)
        out[f"{col}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{col}_roll_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        out[f"{col}_roll_slope"] = (
            grouped[col]
            .rolling(window=window, min_periods=2)
            .apply(_rolling_slope, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    return out


def feature_column_list(report: CleaningReport, include_raw: bool = True) -> list[str]:
    """Columns used as model inputs after cleaning + rolling expansion."""
    base = list(report.kept_settings) + list(report.kept_sensors)
    cols: list[str] = []
    if include_raw:
        cols.extend(base)
    for c in base:
        cols.extend([f"{c}_roll_mean", f"{c}_roll_std", f"{c}_roll_slope"])
    return cols


def prepare_frame(
    df: pd.DataFrame,
    report: CleaningReport,
    *,
    add_rul: bool,
    rul_cap: int | None = RUL_CAP,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Clean → (optional RUL) → rolling features."""
    from .cleaning import apply_cleaning

    out = apply_cleaning(df, report)
    if add_rul:
        out = add_rul_labels(out, cap=rul_cap)
    base_cols = list(report.kept_settings) + list(report.kept_sensors)
    out = add_rolling_features(out, base_cols, window=window)
    return out
