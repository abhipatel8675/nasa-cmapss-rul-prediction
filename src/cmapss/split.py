"""Engine-wise train / validation split (no cross-engine row leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import RANDOM_STATE, VAL_ENGINE_FRACTION


def engine_train_val_split(
    df: pd.DataFrame,
    val_fraction: float = VAL_ENGINE_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fraction of engines for validation.

    Rows from the same unit_id never appear on both sides.
    """
    units = df["unit_id"].unique()
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    # Dummy y for API; groups are unit ids per row
    train_idx, val_idx = next(gss.split(df, groups=df["unit_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def group_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return GroupKFold (train_idx, val_idx) pairs by unit_id."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df["unit_id"]))


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final observed cycle for each engine (official test protocol)."""
    idx = df.groupby("unit_id")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit_id").reset_index(drop=True)
