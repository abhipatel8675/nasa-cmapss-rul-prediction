"""Load C-MAPSS FD001 tables with a consistent schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ALL_COLS, DATA_RAW, INDEX_COLS, SENSOR_COLS, SETTING_COLS


def load_fd001(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (train_df, test_df, test_rul) for FD001."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW

    train = pd.read_csv(
        raw_dir / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=ALL_COLS,
        engine="python",
    )
    test = pd.read_csv(
        raw_dir / "test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=ALL_COLS,
        engine="python",
    )
    # Trailing whitespace can create an empty column; drop if present
    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")

    rul = pd.read_csv(
        raw_dir / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python",
    )["RUL"]

    for df in (train, test):
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    return train, test, rul


def describe_schema() -> str:
    """Human-readable description of the sensor/time-series structure."""
    return (
        "Each row is one operational cycle for one turbofan engine unit.\n"
        f"- Index: {INDEX_COLS}\n"
        f"- Operating settings: {SETTING_COLS}\n"
        f"- Sensor measurements: {SENSOR_COLS}\n"
        "Training trajectories run until failure. Test trajectories are truncated;\n"
        "RUL_FD001.txt holds the true remaining cycles at the last test cycle of each engine."
    )
