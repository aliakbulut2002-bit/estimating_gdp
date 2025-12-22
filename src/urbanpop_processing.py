# src/urbanpop_processing.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TARGET_COUNTRIES

URBAN_XLSX_PATH = BASE_DIR / "data" / "raw" / "urbanpop_rate_raw.xlsx"


def build_urbanpop_panel(first_year: int = 2012) -> Path:
    """
    Construct a country–year panel of the urban population share from a WDI-style Excel extract.

    The raw input is assumed to follow the World Development Indicators (WDI) export structure:
      - a metadata block precedes the tabular header,
      - country identifiers are included in columns such as “Country Code”,
      - annual observations are provided in wide format with one column per year.

    The procedure:
      1) reads the raw Excel sheet,
      2) restricts the sample to the set of countries specified by `TARGET_COUNTRIES`,
      3) retains annual columns from `first_year` onward,
      4) reshapes the dataset from wide to long format,
      5) coerces types, validates ranges, and removes missing observations,
      6) persists the cleaned panel for downstream merges.

    Parameters
    ----------
    first_year : int, default 2012
        Lower bound (inclusive) for the annual columns to include.

    Returns
    -------
    Path
        Path to the processed CSV saved under `BASE_DIR/data/processed/urbanpop_panel.csv`.

    Output schema
    -------------
      - country : str
          Country identifier (from “Country Code”; typically ISO-3).
      - year : int
          Calendar year.
      - urban_pop_rate : float
          Urban population as a percentage of total population (conceptually bounded in [0, 100]).
    """
    print(f"Reading urban population rate data from: {URBAN_XLSX_PATH}")

    # WDI exports commonly place column headers after a short metadata block.
    # `header=3` indicates that the 4th row (0-indexed) contains the column names.
    df = pd.read_excel(
        URBAN_XLSX_PATH,
        sheet_name="Data",
        header=3,
    )

    # Restrict the sample to the project’s country scope.
    df = df[df["Country Code"].isin(TARGET_COUNTRIES)].copy()

    # Identify annual columns (e.g., 1960, "1960", 2012, "2012"). This relies on digit-only labels.
    year_cols = [c for c in df.columns if str(c).isdigit()]
    year_cols_filtered = [c for c in year_cols if int(str(c)) >= first_year]

    if not year_cols_filtered:
        raise RuntimeError(
            f"No year columns >= {first_year} found in urban population rate file."
        )

    # Retain only the country identifier and annual series columns.
    df = df[["Country Code"] + year_cols_filtered]

    # Reshape from wide (one column per year) to long (one row per country-year).
    long_df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols_filtered,
        var_name="year",
        value_name="urban_pop_rate",
    )

    # Coerce types robustly.
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").round().astype("Int64")
    long_df["urban_pop_rate"] = pd.to_numeric(long_df["urban_pop_rate"], errors="coerce")

    # Drop rows with missing keys or missing values.
    long_df = long_df.dropna(subset=["year", "urban_pop_rate"]).reset_index(drop=True)
    long_df["year"] = long_df["year"].astype(int)

    # Standardize naming conventions to match the remainder of the pipeline.
    long_df = long_df.rename(columns={"Country Code": "country"})

    # Basic plausibility check: urban population share should lie in [0, 100].
    # We do not drop automatically; we emit a warning to avoid silently altering the data.
    bad = long_df[(long_df["urban_pop_rate"] < 0) | (long_df["urban_pop_rate"] > 100)]
    if len(bad) > 0:
        print(f"Warning: {len(bad)} urban_pop_rate values fall outside [0, 100].")

    # Sort for readability and reproducibility.
    long_df = long_df.sort_values(["country", "year"]).reset_index(drop=True)

    # Lightweight diagnostics (useful for auditability and debugging).
    print(
        f"UrbanPop panel: rows={len(long_df):,} | countries={long_df['country'].nunique()} | "
        f"years={long_df['year'].min()}–{long_df['year'].max()}"
    )

    # Persist the processed panel for downstream merges.
    out_path = BASE_DIR / "data" / "processed" / "urbanpop_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)

    print(f"Saved urban population rate panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # Enable reproducible module execution (e.g., `python -m src.urbanpop_processing`).
    build_urbanpop_panel(first_year=2012)
