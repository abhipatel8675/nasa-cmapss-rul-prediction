"""Project-wide configuration for C-MAPSS FD001 RUL pipeline."""

from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset
DATASET = "FD001"
RUL_CAP = 125
ROLLING_WINDOW = 5
RANDOM_STATE = 42
VAL_ENGINE_FRACTION = 0.20

# Column schema for C-MAPSS text files
INDEX_COLS = ["unit_id", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

FD001_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]

# Hyperparameter search budget
N_ITER_SEARCH = 12
N_CV_SPLITS = 3
