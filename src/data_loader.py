# src/data_loader.py
"""Data loading and preprocessing for the GDP estimation project."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import BASE_DIR

# ---- global config for reproducibility ----
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PANEL_PATH = BASE_DIR / "data" / "processed" / "model_panel.csv"


def load_and_split(test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Load model_panel.csv, build features, split into train/test, and
    also return Eritrea features for prediction.

    Training data:
        - all rows where country != "ERI"
    Test data:
        - random sample (size = test_size) of the training countries
          using sklearn.model_selection.train_test_split with the
          given random_state.

    Eritrea:
        - all rows where country == "ERI" are kept aside and NEVER used
          in training; they are only used later for prediction.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, n_features)
    X_test  : np.ndarray, shape (n_test, n_features)
    y_train : np.ndarray, shape (n_train,)
    y_test  : np.ndarray, shape (n_test,)
    X_eritrea : np.ndarray or None, shape (n_eri_years, n_features)
    eritrea_meta : pd.DataFrame with columns [country, year]
    """
    print(f"Reading panel from: {PANEL_PATH}")
    df = pd.read_csv(PANEL_PATH)

    # --- clean column names ---
    df = df.rename(columns={"gdp_pcap (US dollars)": "gdp_pcap"})

    if "gdp_pcap" not in df.columns:
        raise ValueError("Expected column 'gdp_pcap' in model_panel.csv")

    # --- target: log GDP per capita ---
    df["log_gdp_pcap"] = np.log(df["gdp_pcap"])
    target_col = "log_gdp_pcap"

    # --- feature engineering ---
    df["log_light"] = np.log1p(df["mean_light"])        # log(1 + light)
    df["log_pop"] = np.log(df["population"])
    df["log_land_area"] = np.log(df["land_area"])
    df["urban_share"] = df["urban_pop_rate"] / 100.0    # from % to 0–1

    feature_cols = ["log_light", "log_pop", "log_land_area", "urban_share"]

    # --- separate Eritrea (prediction target, never in training) ---
    eritrea_mask = df["country"] == "ERI"
    eritrea_meta = df.loc[eritrea_mask, ["country", "year"]].copy()

    X_eritrea = (
        df.loc[eritrea_mask, feature_cols].values if eritrea_mask.any() else None
    )

    # --- training pool: all other countries ---
    df_train = df[~eritrea_mask].dropna(subset=[target_col])

    X = df_train[feature_cols].values
    y = df_train[target_col].values

    # --- train / test split (random but reproducible) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # --- scale features (important for kNN, harmless for RF/OLS) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_eritrea_scaled = (
        scaler.transform(X_eritrea) if X_eritrea is not None else None
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_eritrea_scaled,
        eritrea_meta,
    )
