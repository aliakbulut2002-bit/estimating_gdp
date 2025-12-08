# src/evaluation.py
"""Model evaluation and visualization for GDP per capita regression."""

from pyexpat import model
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from src.data_loader import load_and_split
from src.models import train_random_forest, train_knn, train_linear

RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a regression model on the test set and print results.

    Metrics:
        - RMSE on log GDP per capita
        - R² on log GDP per capita
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"RMSE (log GDPpc): {rmse:.3f}")
    print(f"R²:              {r2:.3f}")

    return rmse, r2


def check_overfitting(model, X_train, y_train, X_test, y_test, name: str):
    "Print train vs test R² to diagnose overfitting."
    train_r2 = model.score(X_train, y_train)   # R² sur train
    test_r2 = model.score(X_test, y_test)      # R² sur test
    gap = train_r2 - test_r2

    print(f"\n{name} – overfitting check:")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  Test  R²: {test_r2:.3f}")
    print(f"  Gap (train - test): {gap:.3f}")
    return train_r2, test_r2, gap

if __name__ == "__main__":
    # 1. Load data and split (Eritrea is held out inside load_and_split)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_eritrea,
        eritrea_meta,
    ) = load_and_split(test_size=0.2, random_state=RANDOM_STATE)

    # 2. Train the three models
    rf = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
    knn = train_knn(X_train, y_train, n_neighbors=15)
    lin = train_linear(X_train, y_train)

    # 3. Evaluate on the test set
    print("Test performance (target = log GDP per capita):")
    rmse_rf, r2_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    rmse_knn, r2_knn = evaluate_model(knn, X_test, y_test, "kNN")
    rmse_lin, r2_lin = evaluate_model(lin, X_test, y_test, "Linear regression")

    # 4. Simple visualization: true vs predicted for Random Forest
    y_pred_rf = rf.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.6)
    plt.xlabel("True log GDP per capita")
    plt.ylabel("Predicted log GDP per capita (RF)")
    plt.title("Random Forest: True vs Predicted (test set)")
    # 45-degree line
    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.tight_layout()
    plt.show()

   