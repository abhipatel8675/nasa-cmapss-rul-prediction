"""Evaluation helpers and figure generation for RUL predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR
from .metrics import regression_metrics


def evaluate_predictions(y_true, y_pred, name: str = "") -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    if name:
        metrics = {"model": name, **metrics}
    return metrics


def plot_pred_vs_actual(y_true, y_pred, title: str, path: Path) -> Path:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1, label="Ideal")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred, title: str, path: Path) -> Path:
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid, bins=20, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("Prediction error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_vs_rul(y_true, y_pred, title: str, path: Path) -> Path:
    y_true = np.asarray(y_true)
    err = np.asarray(y_pred) - y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_true, err, alpha=0.7)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Error (pred - true)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def per_engine_errors(unit_ids, y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "unit_id": np.asarray(unit_ids),
            "true_RUL": np.asarray(y_true, dtype=float),
            "pred_RUL": np.asarray(y_pred, dtype=float),
        }
    )
    df["error"] = df["pred_RUL"] - df["true_RUL"]
    df["abs_error"] = df["error"].abs()
    return df.sort_values("abs_error", ascending=False).reset_index(drop=True)


def save_comparison_bar(metrics_rows: list[dict], path: Path) -> Path:
    df = pd.DataFrame(metrics_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "NASA_score"]):
        ax.bar(df["model"], df[metric], color=["#4C72B0", "#55A868", "#C44E52"][: len(df)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Test-set model comparison (FD001)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def default_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
