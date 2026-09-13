#!/usr/bin/env python3
"""
End-to-end C-MAPSS FD001 RUL pipeline:

download → clean → features → engine-wise split → tune → evaluate → figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.cleaning import cleaning_diagnostics
from cmapss.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, ROLLING_WINDOW, RUL_CAP
from cmapss.download import download_fd001
from cmapss.evaluate import (
    evaluate_predictions,
    per_engine_errors,
    plot_error_vs_rul,
    plot_pred_vs_actual,
    plot_residuals,
    save_comparison_bar,
)
from cmapss.features import feature_column_list, prepare_frame
from cmapss.interpret import plot_feature_importance, tree_feature_importance
from cmapss.load import describe_schema, load_fd001
from cmapss.models import baseline_linear, tune_random_forest, tune_xgboost
from cmapss.split import engine_train_val_split, last_cycle_per_engine


def _xy(df: pd.DataFrame, feature_cols: list[str], target: str = "RUL"):
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df["unit_id"].to_numpy()
    return X, y, groups


def main() -> None:
    print(describe_schema())
    print("-" * 60)

    download_fd001()
    train_raw, test_raw, test_rul = load_fd001()

    report = cleaning_diagnostics(train_raw)
    print("Cleaning report:")
    print(f"  missing values: {report.n_missing}")
    print(f"  dropped columns: {report.dropped_columns}")
    print(f"  kept sensors: {report.kept_sensors}")
    print(f"  kept settings: {report.kept_settings}")

    train_feat = prepare_frame(train_raw, report, add_rul=True, rul_cap=RUL_CAP, window=ROLLING_WINDOW)
    test_feat = prepare_frame(test_raw, report, add_rul=False, window=ROLLING_WINDOW)

    feature_cols = feature_column_list(report, include_raw=True)
    # Drop any accidental non-finite rows from early rolling edge cases
    train_feat = train_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["RUL"])

    train_df, val_df = engine_train_val_split(train_feat)
    print(
        f"Engine-wise split: {train_df['unit_id'].nunique()} train engines, "
        f"{val_df['unit_id'].nunique()} val engines"
    )

    X_train, y_train, g_train = _xy(train_df, feature_cols)
    X_val, y_val, _ = _xy(val_df, feature_cols)
    X_all, y_all, _ = _xy(train_feat, feature_cols)

    results = {"cleaning": {
        "n_missing": report.n_missing,
        "dropped_columns": report.dropped_columns,
        "kept_sensors": report.kept_sensors,
        "kept_settings": report.kept_settings,
        "rul_cap": RUL_CAP,
        "rolling_window": ROLLING_WINDOW,
        "n_features": len(feature_cols),
    }}

    # --- Baseline: Linear Regression ---
    print("\nTraining Linear Regression baseline...")
    lr = baseline_linear()
    lr.fit(X_train, y_train)
    val_lr = evaluate_predictions(y_val, lr.predict(X_val), "LinearRegression")
    print(f"  Val: {val_lr}")

    # --- Random Forest (group-aware tuning on TRAIN engines only) ---
    print("\nTuning Random Forest (GroupKFold on train engines)...")
    rf_search = tune_random_forest(X_train, y_train, g_train)
    rf = rf_search.best_estimator_
    print(f"  Best RF params: {rf_search.best_params_}")
    val_rf = evaluate_predictions(y_val, rf.predict(X_val), "RandomForest")
    print(f"  Val: {val_rf}")

    # --- XGBoost (group-aware tuning on TRAIN engines only) ---
    print("\nTuning XGBoost (GroupKFold on train engines)...")
    xgb_search = tune_xgboost(X_train, y_train, g_train)
    xgb = xgb_search.best_estimator_
    print(f"  Best XGB params: {xgb_search.best_params_}")
    val_xgb = evaluate_predictions(y_val, xgb.predict(X_val), "XGBoost")
    print(f"  Val: {val_xgb}")

    results["validation"] = [val_lr, val_rf, val_xgb]
    results["best_params"] = {
        "RandomForest": rf_search.best_params_,
        "XGBoost": {k: (float(v) if hasattr(v, "item") else v) for k, v in xgb_search.best_params_.items()},
    }

    # Refit selected configs on all training-file engines before official test
    print("\nRefitting models on all training engines...")
    lr.fit(X_all, y_all)
    rf.set_params(**rf_search.best_params_)
    rf.fit(X_all, y_all)
    xgb.set_params(**xgb_search.best_params_)
    xgb.fit(X_all, y_all)

    # Official test: last cycle per engine vs RUL_FD001.txt
    test_last = last_cycle_per_engine(test_feat)
    # Align RUL vector to unit order 1..100
    test_last = test_last.sort_values("unit_id").reset_index(drop=True)
    y_test = test_rul.to_numpy(dtype=float)
    assert len(test_last) == len(y_test), "Test engine count must match RUL file"

    X_test = test_last[feature_cols].to_numpy(dtype=float)
    unit_ids = test_last["unit_id"].to_numpy()

    models = {
        "LinearRegression": lr,
        "RandomForest": rf,
        "XGBoost": xgb,
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_rows = []
    preds = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        row = evaluate_predictions(y_test, y_pred, name)
        test_rows.append(row)
        print(f"Test {name}: {row}")

        slug = name.lower()
        plot_pred_vs_actual(
            y_test, y_pred, f"{name}: Predicted vs Actual RUL (test)", FIGURES_DIR / f"{slug}_pred_vs_actual.png"
        )
        plot_residuals(
            y_test, y_pred, f"{name}: Residuals (test)", FIGURES_DIR / f"{slug}_residuals.png"
        )
        plot_error_vs_rul(
            y_test, y_pred, f"{name}: Error vs True RUL (test)", FIGURES_DIR / f"{slug}_error_vs_rul.png"
        )

    results["test"] = test_rows
    save_comparison_bar(test_rows, FIGURES_DIR / "model_comparison.png")

    # Error analysis on best test RMSE model
    best = min(test_rows, key=lambda r: r["RMSE"])
    best_name = best["model"]
    err_df = per_engine_errors(unit_ids, y_test, preds[best_name])
    err_path = REPORTS_DIR / "per_engine_errors.csv"
    err_df.to_csv(err_path, index=False)
    results["error_analysis"] = {
        "best_model_by_rmse": best_name,
        "worst_5_engines": err_df.head(5)[["unit_id", "true_RUL", "pred_RUL", "error", "abs_error"]].to_dict(
            orient="records"
        ),
        "mean_signed_error": float(err_df["error"].mean()),
        "pct_late_predictions": float((err_df["error"] > 0).mean() * 100),
        "pct_early_predictions": float((err_df["error"] < 0).mean() * 100),
    }
    print(f"\nBest model by test RMSE: {best_name}")
    print("Worst 5 engines:")
    print(err_df.head(5).to_string(index=False))

    # Interpretation
    rf_imp = tree_feature_importance(rf, feature_cols, top_k=20)
    xgb_imp = tree_feature_importance(xgb, feature_cols, top_k=20)
    rf_imp.to_csv(REPORTS_DIR / "rf_feature_importance.csv", index=False)
    xgb_imp.to_csv(REPORTS_DIR / "xgb_feature_importance.csv", index=False)
    plot_feature_importance(rf_imp, "Random Forest feature importance (top 20)", FIGURES_DIR / "rf_importance.png")
    plot_feature_importance(xgb_imp, "XGBoost feature importance (top 20)", FIGURES_DIR / "xgb_importance.png")
    results["top_features"] = {
        "RandomForest": rf_imp.head(10).to_dict(orient="records"),
        "XGBoost": xgb_imp.head(10).to_dict(orient="records"),
    }

    # Persist models + feature list
    artifacts = REPORTS_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, artifacts / "linear_regression.joblib")
    joblib.dump(rf, artifacts / "random_forest.joblib")
    joblib.dump(xgb, artifacts / "xgboost.joblib")
    joblib.dump(feature_cols, artifacts / "feature_cols.joblib")
    joblib.dump(report, artifacts / "cleaning_report.joblib")

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {metrics_path}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
