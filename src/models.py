# src/models.py
"""Model definitions and training for GDP per capita regression."""

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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
    Train a Gradient Boosted Trees regressor (Histogram-based) on (X_train, y_train).

    This is often a strong alternative to Random Forest on tabular data.
    """
    model = HistGradientBoostingRegressor(
        # let the model stop before it overfits
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        max_iter=5000,

        # shrinkage
        learning_rate=0.03,

        # control capacity (key overfitting levers)
        max_depth=3,
        max_leaf_nodes=31,
        min_samples_leaf=50,

        # regularization
        l2_regularization=1.0,

        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors: int = 15):
    """
    Train a k-Nearest Neighbors regressor on (X_train, y_train).

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors used in prediction.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_linear(X_train, y_train):
    """Train a linear regression (OLS) model on (X_train, y_train)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
