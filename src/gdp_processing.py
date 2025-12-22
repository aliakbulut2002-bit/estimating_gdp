from pathlib import Path

import pandas as pd

from src.config import TRAINING_COUNTRIES, BASE_DIR


def build_gdp_panel(
    raw_filename: str = "gdp_per_capita_raw.xlsx",
    first_year: int = 2012,
) -> Path:
    """
    Construct a cleaned country–year panel of GDP per capita from a raw Excel extract.

    The raw input is assumed to follow a typical World Bank-style layout in which:
      - country identifiers are provided in dedicated columns (e.g., “Country Name”, “Country Code”),
      - annual observations are stored in wide format with one column per year.

    The output is a tidy (long) panel suitable for downstream merges and modeling.

    Parameters
    ----------
    raw_filename : str, default "gdp_per_capita_raw.xlsx"
        Filename located under `BASE_DIR/data/raw/` containing the raw GDP-per-capita series.
    first_year : int, default 2012
        Lower bound for the set of annual columns to keep.

    Returns
    -------
    Path
        File path to the processed CSV saved under `BASE_DIR/data/processed/gdp_panel.csv`.

    Output schema
    -------------
    A CSV with (at minimum) the following columns:
      - country : str
          Country identifier (here, the “Country Code” from the raw file; typically ISO-3).
      - year : int
          Calendar year.
      - gdp_pcap (US dollars) : float
          GDP per capita in US dollars, excluding missing observations.

    Notes
    -----
    - The sample is restricted to `TRAINING_COUNTRIES` to maintain alignment with the modeling design.
    - Rows with missing GDP-per-capita values are dropped to avoid propagating NAs into the merged panel.
    """
    raw_path = BASE_DIR / "data" / "raw" / raw_filename
    print(f"Reading raw GDP file from: {raw_path}")

    # Load the data sheet. `skiprows=3` assumes a header/metadata block precedes the tabular content.
    df_raw = pd.read_excel(
        raw_path,
        sheet_name="Data",
        skiprows=3,
    )

    # Identify year columns at or after `first_year`.
    # The current logic assumes year headers are *strings* containing digits (e.g., "2012").
    year_cols = [
        c for c in df_raw.columns
        if isinstance(c, str) and c.isdigit() and int(c) >= first_year
    ]

    # Retain identifiers plus the selected year columns, then restrict the sample to training countries.
    df = df_raw[["Country Name", "Country Code"] + year_cols].copy()
    df = df[df["Country Code"].isin(TRAINING_COUNTRIES)].copy()

    # Reshape from wide (one column per year) to long (one row per country-year).
    df_long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_pcap (US dollars)",
    )

    # Standardize types and naming conventions.
    df_long["year"] = df_long["year"].astype(int)

    # “country” is used as the canonical country identifier in the remainder of the pipeline.
    # Here it corresponds to the raw “Country Code” (typically ISO-3), not the country name.
    df_long.rename(columns={"Country Code": "country"}, inplace=True)

    # Drop missing GDP-per-capita observations to produce a complete response variable panel.
    df_long = df_long.dropna(subset=["gdp_pcap (US dollars)"])

    # Select and order the final panel columns.
    panel = df_long[["country", "year", "gdp_pcap (US dollars)"]].sort_values(["country", "year"])

    # Persist the processed panel for downstream merges.
    out_path = BASE_DIR / "data" / "processed" / "gdp_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"Saved cleaned GDP panel to {out_path}")
    return out_path


if __name__ == "__main__":
    # Enable module execution (e.g., `python -m src.gdp_processing`) to reproduce the GDP panel artifact.
    build_gdp_panel()
