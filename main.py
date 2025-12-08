# main.py
"""
Main script to compare ML models for GDP per capita regression.
"""

from src.data_loader import load_and_split
from src.models import train_random_forest, train_knn, train_linear
from src.evaluation import check_overfitting, evaluate_model
import numpy as np

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("GDP per capita regression: Model comparison")
    print("=" * 60)

    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_eritrea,
        eritrea_meta,
    ) = load_and_split(test_size=0.2, random_state=RANDOM_STATE)

    print(f"   Train size: {X_train.shape}")
    print(f"   Test size:  {X_test.shape}")
    if X_eritrea is not None:
        print(f"   Eritrea rows kept for prediction: {X_eritrea.shape[0]}")

    # 2. Train models
    print("\n2. Training models...")
    rf_model = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
    knn_model = train_knn(X_train, y_train, n_neighbors=15)   # <- no random_state
    lr_model = train_linear(X_train, y_train)                 # <- no random_state
    print("   ✓ All models trained")

    # 3. Evaluate models
    print("\n3. Evaluating models on test set...")
    rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    knn_rmse, knn_r2 = evaluate_model(knn_model, X_test, y_test, "kNN")
    lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test, "Linear regression")

    # 4. Overfitting check (R² train vs test)
    check_overfitting(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    check_overfitting(knn_model, X_train, y_train, X_test, y_test, "kNN")
    check_overfitting(lr_model, X_train, y_train, X_test, y_test, "Linear regression")

    # 5. Conclusion
    results_r2 = {
        "Random Forest": rf_r2,
        "kNN": knn_r2,
        "Linear regression": lr_r2,
    }
    winner = max(results_r2, key=results_r2.get)

    print("\n" + "=" * 60)
    print(f"Best model (by R²): {winner} (R² = {results_r2[winner]:.3f})")
    print("=" * 60)


    #6. Predict GDP for Eritrea using the best model
    if X_eritrea is not None:
        print("\nEstimating GDP per capita for Eritrea...")

        # for now we know Random Forest is the winner, so we use rf_model
        best_model = rf_model   # or choose based on 'winner' if you want to generalize

        y_log_eri = best_model.predict(X_eritrea)   # predictions in log GDPpc
        y_eri = np.exp(y_log_eri)                   # back to level (USD per capita)

        print("\nPredictions for Eritrea (GDP per capita in USD):")
        for (country, year), gdp_pcap in zip(eritrea_meta.values, y_eri):
            print(f"{country} {year}: {gdp_pcap:,.0f} USD")

        print("=" * 60)
     # 6. Predict GDP for Eritrea using the best model
    print("\nDebug Eritrea in main():")
    print("  X_eritrea is None? ", X_eritrea is None)
    if X_eritrea is not None:
        print("  X_eritrea shape:   ", X_eritrea.shape)
        print("  eritrea_meta head:\n", eritrea_meta.head())

    if X_eritrea is not None and len(X_eritrea) > 0:
        print("\nEstimating GDP per capita for Eritrea...")

        best_model = rf_model   # or choose by 'winner'

        y_log_eri = best_model.predict(X_eritrea)   # log GDPpc
        y_eri = np.exp(y_log_eri)                   # back to level

        print("\nPredictions for Eritrea (GDP per capita in USD):")
        for (country, year), gdp_pcap in zip(eritrea_meta.values, y_eri):
            print(f"{country} {year}: {gdp_pcap:,.0f} USD")

        print("=" * 60)
    else:
        print("\nNo Eritrea predictions: X_eritrea is None or empty.")

if __name__ == "__main__":
    main()

