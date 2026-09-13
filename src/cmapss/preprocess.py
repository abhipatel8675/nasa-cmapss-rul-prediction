"""Train-only preprocessing (scaling) without leakage."""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train) -> StandardScaler:
    """Fit StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform(scaler: StandardScaler, X):
    return scaler.transform(X)
