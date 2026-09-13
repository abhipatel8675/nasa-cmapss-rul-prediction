"""Model interpretation helpers (feature importance)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tree_feature_importance(model, feature_names: list[str], top_k: int = 20) -> pd.DataFrame:
    """Extract impurity-based importance for RF / XGB."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    else:
        raise AttributeError("Model has no feature_importances_")
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = importance_df.iloc[::-1]
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
