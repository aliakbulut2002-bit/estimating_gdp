# src/data_loader.py
"""Data ingestion and preprocessing routines for the GDP-per-capita estimation pipeline."""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import BASE_DIR

# Global reproducibility settings
RANDOM_STATE = 42


# Path to the consolidated modeling panel produced by the upstream preprocessing and merge steps
PANEL_PATH = BASE_DIR / "data" / "processed" / "model_panel.csv"


def load_and_split(test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Loads the merged country-year modeling panel, constructs features, and splits the training
    sample into train/test sets. Eritrea is systematically excluded from training and held
    out for out-of-sample prediction.

    Design
    ------
    Training pool:
        - All observations with country != "ERI" and non-missing target.
    Test set:
        - Random holdout from the training pool with size `test_size`, using the provided
          `random_state` for reproducibility.
    Eritrea holdout:
        - All observations with country == "ERI" are returned separately and are never
          used for model fitting.

    Parameters
    ----------
    test_size : float
        Proportion of the non-Eritrea sample reserved for testing.
    random_state : int
        Seed controlling the train/test split.

    Returns
    -------
    X_train : np.ndarray
        Standardized feature matrix for the training set, shape (n_train, n_features).
    X_test : np.ndarray
        Standardized feature matrix for the test set, shape (n_test, n_features).
    y_train : np.ndarray
        Target vector for the training set (log GDP per capita), shape (n_train,).
    y_test : np.ndarray
        Target vector for the test set (log GDP per capita), shape (n_test,).
    X_eritrea : np.ndarray or None
        Standardized Eritrea feature matrix for prediction, shape (n_eri_years, n_features),
        or None if Eritrea observations are not present in the panel.
    eritrea_meta : pd.DataFrame
        Metadata for Eritrea rows (country, year), aligned with X_eritrea where applicable.
    """
    print(f"Reading panel from: {PANEL_PATH}")
    df = pd.read_csv(PANEL_PATH)

    # Standardize GDP-per-capita column name for downstream consistency
    df = df.rename(columns={"gdp_pcap (US dollars)": "gdp_pcap"})

    if "gdp_pcap" not in df.columns:
        raise ValueError("Expected column 'gdp_pcap' in model_panel.csv")

    # Target definition: natural logarithm of GDP per capita
    df["log_gdp_pcap"] = np.log(df["gdp_pcap"])
    target_col = "log_gdp_pcap"

    # Feature construction (log transforms and unit normalization)
    df["log_light"] = np.log1p(df["mean_light"])        # log(1 + mean_light)
    df["log_pop"] = np.log(df["population"])
    df["log_land_area"] = np.log(df["land_area"])
    df["urban_share"] = df["urban_pop_rate"] / 100.0    # convert percentage to share in [0, 1]

    feature_cols = ["log_light", "log_pop", "log_land_area", "urban_share"]

    # Eritrea holdout sample (excluded from training; used only for prediction)
    eritrea_mask = df["country"] == "ERI"
    eritrea_meta = df.loc[eritrea_mask, ["country", "year"]].copy()

    X_eritrea = df.loc[eritrea_mask, feature_cols].values if eritrea_mask.any() else None

    # Training pool: all non-Eritrea observations with observed target
    df_train = df[~eritrea_mask].dropna(subset=[target_col])

    X = df_train[feature_cols].values
    y = df_train[target_col].values

    # Random train/test split (reproducible via `random_state`)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Feature standardization (required for distance-based methods; benign for tree-based and linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_eritrea_scaled = scaler.transform(X_eritrea) if X_eritrea is not None else None

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_eritrea_scaled,
        eritrea_meta,
    )
