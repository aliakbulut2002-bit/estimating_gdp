# src/urbanpop_processing.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TARGET_COUNTRIES

URBAN_XLSX_PATH = BASE_DIR / "data" / "raw" / "urbanpop_rate_raw.xlsx"


def build_urbanpop_panel(first_year: int = 2012) -> Path:
    """
    Read WDI urban population rate Excel, keep only TARGET_COUNTRIES and years >= first_year,
    and save a long panel CSV: country, year, urban_pop_rate.

    urban_pop_rate is in % of total population (0–100).
    """
    print(f"Reading urban population rate data from: {URBAN_XLSX_PATH}")

    # WDI layout: real header on 4th row (index 3), data in sheet "Data"
    df = pd.read_excel(
        URBAN_XLSX_PATH,
        sheet_name="Data",
        header=3,
    )

    # Keep only the countries we care about
    df = df[df["Country Code"].isin(TARGET_COUNTRIES)].copy()

    # Year columns are the ones that are just digits like "1960", "2012", ...
    year_cols = [c for c in df.columns if str(c).isdigit()]
    year_cols_filtered = [c for c in year_cols if int(str(c)) >= first_year]

    if not year_cols_filtered:
        raise RuntimeError(
            f"No year columns >= {first_year} found in urban population rate file."
        )

    # Keep only useful columns
    df = df[["Country Code"] + year_cols_filtered]

    # Wide -> long format
    long_df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols_filtered,
        var_name="year",
        value_name="urban_pop_rate",  # percentage of total population
    )

    # Clean types
    long_df["year"] = long_df["year"].astype(int)
    long_df["urban_pop_rate"] = pd.to_numeric(long_df["urban_pop_rate"], errors="coerce")

    # Rename country column to match rest of project
    long_df = long_df.rename(columns={"Country Code": "country"})

    # Drop rows with missing values
    long_df = long_df.dropna(subset=["urban_pop_rate"]).reset_index(drop=True)

    # Save processed panel
    out_path = BASE_DIR / "data" / "processed" / "urbanpop_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)

    print(f"Saved urban population rate panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # from 2012 onward to match other panels
    build_urbanpop_panel(first_year=2012)
