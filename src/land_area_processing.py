# src/land_area_processing.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, TARGET_COUNTRIES

LAND_XLSX_PATH = BASE_DIR / "data" / "raw" / "land_area_raw.xlsx"


def build_land_area_panel(first_year: int = 2012) -> Path:
    """
    Read WDI land area Excel, keep only TARGET_COUNTRIES and years >= first_year,
    and save a long panel CSV: country, year, land_area.
    """
    print(f"Reading land area data from: {LAND_XLSX_PATH}")

    # WDI format: real header row is the 4th row (index 3), sheet "Data"
    df = pd.read_excel(
        LAND_XLSX_PATH,
        sheet_name="Data",
        header=3,
    )

    # Keep only countries of interest
    df = df[df["Country Code"].isin(TARGET_COUNTRIES)].copy()

    # Identify year columns (1960, 1961, ..., etc.)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    year_cols_filtered = [c for c in year_cols if int(str(c)) >= first_year]

    if not year_cols_filtered:
        raise RuntimeError(f"No year columns >= {first_year} found in land area file.")

    # Keep only relevant columns
    df = df[["Country Code"] + year_cols_filtered]

    # Wide -> long
    long_df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols_filtered,
        var_name="year",
        value_name="land_area",
    )

    # Clean types
    long_df["year"] = long_df["year"].astype(int)
    long_df["land_area"] = pd.to_numeric(long_df["land_area"], errors="coerce")

    # Rename to match rest of project
    long_df = long_df.rename(columns={"Country Code": "country"})

    # Drop missing
    long_df = long_df.dropna(subset=["land_area"]).reset_index(drop=True)

    # Save
    out_path = BASE_DIR / "data" / "processed" / "land_area_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)

    print(f"Saved land area panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # from 2012 onwards
    build_land_area_panel(first_year=2012)
