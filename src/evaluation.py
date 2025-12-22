# src/evaluation.py
"""Model evaluation and visualization for GDP per capita regression."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_and_split
from src.models import train_random_forest, train_knn, train_linear, train_gbdt

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
    """
    Train vs Test comparison to diagnose overfitting.

    Overfitting signal:
      - Train RMSE much lower than Test RMSE
      - Train R² much higher than Test R²
    """
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, yhat_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, yhat_test))

    r2_train = r2_score(y_train, yhat_train)
    r2_test = r2_score(y_test, yhat_test)

    print(f"\n{name} — overfitting check:")
    print(f"  Train RMSE: {rmse_train:.3f} | Train R²: {r2_train:.3f}")
    print(f"  Test  RMSE: {rmse_test:.3f} | Test  R²: {r2_test:.3f}")
    print(f"  RMSE gap (test - train): {rmse_test - rmse_train:.3f}")
    print(f"  R² gap   (train - test): {r2_train - r2_test:.3f}")

    return rmse_train, rmse_test, r2_train, r2_test


def plot_true_vs_pred(y_true, y_pred, title: str, xlabel: str, ylabel: str):
    """Helper for True vs Predicted scatter plot with 45° line."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.show()


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

    # 2. Train the models (requested order)
    lin = train_linear(X_train, y_train)
    knn = train_knn(X_train, y_train, n_neighbors=15)
    gbdt = train_gbdt(X_train, y_train, random_state=RANDOM_STATE)
    rf = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)

    # 3. Evaluate on the test set (requested order)
    print("Test performance (target = log GDP per capita):")
    rmse_lin, r2_lin = evaluate_model(lin, X_test, y_test, "Linear regression")
    rmse_knn, r2_knn = evaluate_model(knn, X_test, y_test, "kNN")
    rmse_gbdt, r2_gbdt = evaluate_model(gbdt, X_test, y_test, "Gradient Boosted Trees")
    rmse_rf, r2_rf = evaluate_model(rf, X_test, y_test, "Random Forest")

    # 4. Overfitting checks (requested order)
    check_overfitting(lin, X_train, y_train, X_test, y_test, "Linear regression")
    check_overfitting(knn, X_train, y_train, X_test, y_test, "kNN")
    check_overfitting(gbdt, X_train, y_train, X_test, y_test, "Gradient Boosted Trees")
    check_overfitting(rf, X_train, y_train, X_test, y_test, "Random Forest")

    # 5. Plots: true vs predicted for each model (requested order)
    y_pred_lin = lin.predict(X_test)
    plot_true_vs_pred(
        y_test, y_pred_lin,
        title="Linear regression: True vs Predicted (test set)",
        xlabel="True log GDP per capita",
        ylabel="Predicted log GDP per capita (Linear)",
    )

    y_pred_knn = knn.predict(X_test)
    plot_true_vs_pred(
        y_test, y_pred_knn,
        title="kNN: True vs Predicted (test set)",
        xlabel="True log GDP per capita",
        ylabel="Predicted log GDP per capita (kNN)",
    )

    y_pred_gbdt = gbdt.predict(X_test)
    plot_true_vs_pred(
        y_test, y_pred_gbdt,
        title="Gradient Boosted Trees: True vs Predicted (test set)",
        xlabel="True log GDP per capita",
        ylabel="Predicted log GDP per capita (GBDT)",
    )

    y_pred_rf = rf.predict(X_test)
    plot_true_vs_pred(
        y_test, y_pred_rf,
        title="Random Forest: True vs Predicted (test set)",
        xlabel="True log GDP per capita",
        ylabel="Predicted log GDP per capita (RF)",
    )
