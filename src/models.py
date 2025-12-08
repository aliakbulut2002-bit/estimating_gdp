# src/models.py
"""Model definitions and training for GDP per capita regression."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42


def train_random_forest(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Train a Random Forest regressor on (X_train, y_train).

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : RandomForestRegressor
        Fitted model.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors: int = 15):
    """
    Train a k-Nearest Neighbors regressor on (X_train, y_train).

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    n_neighbors : int
        Number of neighbors used in prediction.

    Returns
    -------
    model : KNeighborsRegressor
        Fitted model.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_linear(X_train, y_train):
    """
    Train a linear regression (OLS) model on (X_train, y_train).

    Returns
    -------
    model : LinearRegression
        Fitted model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
