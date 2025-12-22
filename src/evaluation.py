# src/evaluation.py
"""
Evaluation utilities for GDP-per-capita regression.

This module defines a compact set of functions for:
1) computing standard regression metrics (RMSE and R²),
2) producing an across-model comparison table, and
3) generating diagnostic figures (true vs. predicted per model, and bar charts for RMSE/R²).

The module is intentionally designed to be *imported and called from `main.py`* rather than
executed as a standalone script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination (R²)."""
    return float(r2_score(y_true, y_pred))


def evaluate_one_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a fitted model on both training and test sets.

    Parameters
    ----------
    model
        A fitted regressor implementing `.predict(X)`.
    X_train, y_train
        Training set features and target.
    X_test, y_test
        Test set features and target.
    model_name : str
        Label used for reporting.

    Returns
    -------
    dict
        Dictionary containing training and test RMSE/R², as well as simple generalization gaps.
    """
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    train_rmse = rmse(y_train, yhat_train)
    test_rmse = rmse(y_test, yhat_test)

    train_r2 = r2(y_train, yhat_train)
    test_r2 = r2(y_test, yhat_test)

    return {
        "model": model_name,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "rmse_gap": test_rmse - train_rmse,   # positive values suggest worse generalization
        "r2_gap": train_r2 - test_r2,         # positive values suggest worse generalization
    }


def build_results_table(results: list[dict], sort_by: str = "test_rmse") -> pd.DataFrame:
    """
    Construct a model-comparison table from per-model evaluation dictionaries.

    Parameters
    ----------
    results : list of dict
        Output of repeated calls to `evaluate_one_model`.
    sort_by : str
        Column used to sort the table (default: test RMSE, ascending).

    Returns
    -------
    pd.DataFrame
        A sorted results table suitable for printing or exporting.
    """
    df = pd.DataFrame(results)
    ascending = sort_by.endswith("rmse")
    return df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def print_results_table(results_df: pd.DataFrame) -> None:
    """
    Print a formatted model-comparison table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of `build_results_table`.
    """
    cols = ["model", "test_rmse", "test_r2", "train_rmse", "train_r2", "rmse_gap", "r2_gap"]
    print("\n" + "=" * 80)
    print("Model comparison (target = log GDP per capita)")
    print("=" * 80)
    print(
        results_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print("=" * 80)


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Plot observed versus predicted outcomes with a 45-degree reference line.

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.
    y_pred : np.ndarray
        Predicted values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
 


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
) -> None:
    """
    Bar chart comparison across models for a selected metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model comparison table.
    metric_col : str
        Column to plot (e.g., 'test_r2' or 'test_rmse').
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    """
    df = results_df.copy()
    ascending = metric_col.endswith("rmse")
    df = df.sort_values(metric_col, ascending=ascending)

    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df[metric_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    
