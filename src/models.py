# src/models.py
"""
Model training utilities for GDP-per-capita regression.

This module defines a small set of baseline and non-linear regressors commonly used for
tabular prediction tasks. All functions return *fitted* scikit-learn estimators, intended
to be called from `main.py` to ensure a single, reproducible execution path.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42


def train_random_forest(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Fit a Random Forest regressor.

    Random Forests provide a flexible non-parametric benchmark that can capture non-linear
    relationships and interactions with limited feature engineering. The hyperparameters
    below are chosen to balance predictive performance and regularization on panel-style
    macroeconomic covariates.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training covariates.
    y_train : array-like of shape (n_samples,)
        Training target (e.g., log GDP per capita).
    random_state : int, default RANDOM_STATE
        Seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted Random Forest estimator.
    """
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=8,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_gbdt(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Fit a histogram-based Gradient Boosting regressor.

    Gradient boosting constructs an additive ensemble of shallow trees and is often
    competitive on tabular data. Early stopping is enabled to reduce overfitting by
    selecting the number of boosting iterations using a validation split.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training covariates.
    y_train : array-like of shape (n_samples,)
        Training target (e.g., log GDP per capita).
    random_state : int, default RANDOM_STATE
        Seed for reproducibility.

    Returns
    -------
    HistGradientBoostingRegressor
        Fitted gradient boosting estimator.
    """
    model = HistGradientBoostingRegressor(
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        max_iter=5000,
        learning_rate=0.03,
        max_depth=3,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors: int = 15):
    """
    Fit a k-Nearest Neighbors (kNN) regressor.

    kNN is a non-parametric baseline that predicts outcomes by averaging the targets of the
    nearest observations in feature space. Performance is sensitive to feature scaling and
    the choice of distance metric; here we use the scikit-learn default (Euclidean distance).

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training covariates.
    y_train : array-like of shape (n_samples,)
        Training target (e.g., log GDP per capita).
    n_neighbors : int, default 15
        Number of neighbors used in prediction.

    Returns
    -------
    KNeighborsRegressor
        Fitted kNN estimator.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_linear(X_train, y_train):
    """
    Fit a linear regression model (ordinary least squares).

    This estimator serves as a transparent baseline with additive effects. While it may be
    misspecified under non-linearities, it provides a useful reference point for evaluating
    the incremental value of non-linear machine learning models.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training covariates.
    y_train : array-like of shape (n_samples,)
        Training target (e.g., log GDP per capita).

    Returns
    -------
    LinearRegression
        Fitted OLS estimator.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
