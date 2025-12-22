# main.py
"""
Main script to run the full pipeline:
raw data -> processed tables -> merged modeling panel -> train/evaluate -> Eritrea estimate
"""

import sys
import subprocess
import numpy as np

from src.data_loader import load_and_split
from src.models import train_random_forest, train_knn, train_linear, train_gbdt
from src.evaluation import check_overfitting, evaluate_model

RANDOM_STATE = 42


def run_module(module_name: str):
    """Run a Python module like: python -m src.viirs_processing"""
    print(f"  Running {module_name} ...")
    subprocess.run([sys.executable, "-m", module_name], check=True)


def run_data_pipeline():
    """
    Run all processing steps in a sensible order.
    Adjust order if your merge expects a different sequence.
    """
    print("\n0) Building dataset from raw files...")

    # Feature/inputs processing
    run_module("src.viirs_processing")
    run_module("src.population_processing")
    run_module("src.urbanpop_processing")
    run_module("src.land_area_processing")
    run_module("src.gdp_processing")

    # Final merge into a modeling panel (the file your data_loader reads)
    run_module("src.merge_data")

    print("   ✓ Data pipeline finished")


def main():
    print("=" * 60)
    print("GDP per capita regression: End-to-end pipeline")
    print("=" * 60)

    # 0) Build processed data + merged panel from raw data
    run_data_pipeline()

    # 1) Load and split data
    print("\n1) Loading modeling panel and splitting data...")
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

    # 2) Train models
    print("\n2) Training models...")
    lr_model = train_linear(X_train, y_train)
    knn_model = train_knn(X_train, y_train, n_neighbors=15)
    gbdt_model = train_gbdt(X_train, y_train, random_state=RANDOM_STATE)
    rf_model = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
    print("   ✓ All models trained")

    # 3) Evaluate models
    print("\n3) Evaluating models on test set (target = log GDP per capita)...")
    results = {}

    rmse_lr, r2_lr = evaluate_model(lr_model, X_test, y_test, "Linear regression")
    results["Linear regression"] = {"rmse": rmse_lr, "r2": r2_lr, "model": lr_model}

    rmse_knn, r2_knn = evaluate_model(knn_model, X_test, y_test, "kNN")
    results["kNN"] = {"rmse": rmse_knn, "r2": r2_knn, "model": knn_model}

    rmse_gbdt, r2_gbdt = evaluate_model(gbdt_model, X_test, y_test, "Gradient Boosted Trees")
    results["Gradient Boosted Trees"] = {"rmse": rmse_gbdt, "r2": r2_gbdt, "model": gbdt_model}

    rmse_rf, r2_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results["Random Forest"] = {"rmse": rmse_rf, "r2": r2_rf, "model": rf_model}

    # 4) Overfitting checks
    print("\n4) Overfitting checks (train vs test)...")
    check_overfitting(lr_model, X_train, y_train, X_test, y_test, "Linear regression")
    check_overfitting(knn_model, X_train, y_train, X_test, y_test, "kNN")
    check_overfitting(gbdt_model, X_train, y_train, X_test, y_test, "Gradient Boosted Trees")
    check_overfitting(rf_model, X_train, y_train, X_test, y_test, "Random Forest")

    # 5) Pick best model 
    winner = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[winner]["model"]

    print("\n" + "=" * 60)
    print("Leaderboard (sorted by RMSE, lower is better):")
    for name in sorted(results, key=lambda k: results[k]["rmse"]):
        print(f"  {name:24s}  RMSE={results[name]['rmse']:.3f}  R²={results[name]['r2']:.3f}")
    print("-" * 60)
    print(f"Best model (by RMSE): {winner} (RMSE = {results[winner]['rmse']:.3f})")
    print("=" * 60)

    # 6) Eritrea predictions (once)
    if X_eritrea is not None and len(X_eritrea) > 0:
        print("\nEstimating GDP per capita for Eritrea...")
        y_log_eri = best_model.predict(X_eritrea)
        y_eri = np.exp(y_log_eri)

        print("\nPredictions for Eritrea (GDP per capita in USD):")
        for (country, year), gdp_pcap in zip(eritrea_meta.values, y_eri):
            print(f"{country} {year}: {gdp_pcap:,.0f} USD")
        print("=" * 60)
    else:
        print("\nNo Eritrea predictions: X_eritrea is None or empty.")


if __name__ == "__main__":
    main()
