# src/population_processing.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TARGET_COUNTRIES

POP_XLSX_PATH = BASE_DIR / "data" / "raw" / "population_raw.xlsx"


def build_population_panel(first_year: int = 2012) -> Path:
    """
    Construct a country–year population panel from a WDI-style Excel extract.

    The raw input is assumed to follow the World Development Indicators (WDI) export structure:
      - a small metadata block precedes the tabular header,
      - country identifiers are provided in columns such as “Country Code”,
      - annual observations are provided in wide format (one column per year).

    The routine:
      1) reads the raw Excel sheet,
      2) restricts the sample to the set of countries specified by `TARGET_COUNTRIES`,
      3) retains annual columns from `first_year` onward,
      4) reshapes the dataset from wide to long format,
      5) coerces variable types and removes missing observations,
      6) persists the cleaned panel for downstream merges.

    Parameters
    ----------
    first_year : int, default 2012
        Lower bound (inclusive) for the annual columns to include in the output panel.

    Returns
    -------
    Path
        Path to the processed CSV saved under `BASE_DIR/data/processed/population_panel.csv`.

    Output schema
    -------------
      - country : str
          Country identifier (from “Country Code”; typically ISO-3).
      - year : int
          Calendar year.
      - population : float
          Total population for the given country-year (as reported in the source).
    """
    print(f"Reading population data from: {POP_XLSX_PATH}")

    # WDI exports often place the tabular header after a metadata block.
    # Here, `header=3` indicates the 4th row (0-indexed) contains column names.
    df = pd.read_excel(POP_XLSX_PATH, sheet_name="Data", header=3)

    # Restrict to the set of countries used in the project’s target scope.
    df = df[df["Country Code"].isin(TARGET_COUNTRIES)].copy()

    # Identify annual columns (expected to be digit-only labels such as "2012", "2013", ...).
    year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]

    # Retain annual columns from `first_year` onward.
    year_cols_filtered = [c for c in year_cols if int(c) >= first_year]

    if not year_cols_filtered:
        raise RuntimeError(f"No year columns >= {first_year} found in population file.")

    # Keep only the identifier and annual population series.
    df = df[["Country Code"] + year_cols_filtered]

    # Reshape from wide (one column per year) to long (one row per country-year).
    long_df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols_filtered,
        var_name="year",
        value_name="population",
    )

    # Coerce types: year as integer and population as numeric.
    long_df["year"] = long_df["year"].astype(int)
    long_df["population"] = pd.to_numeric(long_df["population"], errors="coerce")

    # Standardize naming to match the remainder of the pipeline.
    long_df = long_df.rename(columns={"Country Code": "country"})

    # Drop missing population observations.
    long_df = long_df.dropna(subset=["population"]).reset_index(drop=True)

    # Persist the processed panel for downstream merges.
    out_path = BASE_DIR / "data" / "processed" / "population_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)

    print(f"Saved population panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # Enable reproducible module execution (e.g., `python -m src.population_processing`).
    build_population_panel(first_year=2012)
