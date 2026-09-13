"""Explicit data-cleaning helpers for C-MAPSS FD001."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SENSOR_COLS, SETTING_COLS


@dataclass
class CleaningReport:
    n_missing: int
    constant_sensors: list[str]
    near_constant_sensors: list[str]
    dropped_columns: list[str]
    kept_sensors: list[str]
    kept_settings: list[str]


def find_constant_columns(df: pd.DataFrame, cols: list[str], eps: float = 1e-9) -> list[str]:
    """Columns with near-zero variance."""
    constant = []
    for c in cols:
        if c not in df.columns:
            continue
        if float(df[c].std()) <= eps:
            constant.append(c)
    return constant


def cleaning_diagnostics(train: pd.DataFrame) -> CleaningReport:
    """Inspect missing values and constant sensors/settings on the training set."""
    n_missing = int(train.isna().sum().sum())
    constant_sensors = find_constant_columns(train, SENSOR_COLS)
    # Near-constant: very low relative variation
    near = []
    for c in SENSOR_COLS:
        if c in constant_sensors or c not in train.columns:
            continue
        std = float(train[c].std())
        rng = float(train[c].max() - train[c].min())
        if rng > 0 and std / rng < 1e-3:
            near.append(c)

    constant_settings = find_constant_columns(train, SETTING_COLS)
    dropped = sorted(set(constant_sensors + near + constant_settings))
    kept_sensors = [c for c in SENSOR_COLS if c not in dropped]
    kept_settings = [c for c in SETTING_COLS if c not in dropped]

    return CleaningReport(
        n_missing=n_missing,
        constant_sensors=constant_sensors,
        near_constant_sensors=near,
        dropped_columns=dropped,
        kept_sensors=kept_sensors,
        kept_settings=kept_settings,
    )


def apply_cleaning(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop columns identified as uninformative on the training diagnostics."""
    out = df.copy()
    drop = [c for c in report.dropped_columns if c in out.columns]
    return out.drop(columns=drop)
