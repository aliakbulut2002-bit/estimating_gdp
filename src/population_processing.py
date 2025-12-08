# src/population_processing.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TARGET_COUNTRIES

POP_XLSX_PATH = BASE_DIR / "data" / "raw" / "population_raw.xlsx"


def build_population_panel(first_year: int = 2012) -> Path:
    """
    Read WDI population Excel, keep only TARGET_COUNTRIES and years >= first_year,
    and save a long panel CSV: country, year, population.
    """
    print(f"Reading population data from: {POP_XLSX_PATH}")

    # In the WDI Excel, the real header is on the 4th row (row index 3):
    # Country Name | Country Code | Indicator Name | Indicator Code | 1960 | 1961 | ...
    df = pd.read_excel(POP_XLSX_PATH, sheet_name="Data", header=3)

    # Keep only the countries we care about
    df = df[df["Country Code"].isin(TARGET_COUNTRIES)].copy()

    # Identify year columns (those whose name is all digits)
    year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()]

    # Convert to int and keep only years >= first_year
    year_cols_filtered = [c for c in year_cols if int(c) >= first_year]

    if not year_cols_filtered:
        raise RuntimeError(f"No year columns >= {first_year} found in population file.")

    # Keep only relevant columns
    df = df[["Country Code"] + year_cols_filtered]

    # Wide → long: one row per (country, year)
    long_df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols_filtered,
        var_name="year",
        value_name="population",
    )

    # Clean types
    long_df["year"] = long_df["year"].astype(int)
    long_df["population"] = pd.to_numeric(long_df["population"], errors="coerce")

    # Rename country column to match the rest of the project
    long_df = long_df.rename(columns={"Country Code": "country"})

    # Optionally drop rows with missing population
    long_df = long_df.dropna(subset=["population"]).reset_index(drop=True)

    # Save
    out_path = BASE_DIR / "data" / "processed" / "population_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)

    print(f"Saved population panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # From 2012 onwards 
    build_population_panel(first_year=2012)
