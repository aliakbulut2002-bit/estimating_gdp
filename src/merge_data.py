# src/merge_data.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR


def build_model_panel() -> Path:
    """
    Merge all processed panels (GDP, VIIRS, population, land area, urban pop)
    into a single panel on (country, year).

    Logic:
        - Start from VIIRS (where Eritrea is present).
        - LEFT join GDP: Eritrea is kept even if missing in gdp_panel,
          so its GDP will be NaN (to be predicted later).
        - INNER join population, land area, and urban pop: we require
          Eritrea to exist in these panels as well.

    Output:
        data/processed/model_panel.csv
    """
    processed_dir = BASE_DIR / "data" / "processed"

    # ---- read individual panels ----
    gdp = pd.read_csv(processed_dir / "gdp_panel.csv")
    viirs = pd.read_csv(processed_dir / "viirs_panel.csv")
    pop = pd.read_csv(processed_dir / "population_panel.csv")
    land = pd.read_csv(processed_dir / "land_area_panel.csv")
    urban = pd.read_csv(processed_dir / "urbanpop_panel.csv")

    # Ensure types are consistent
    for df in (gdp, viirs, pop, land, urban):
        df["country"] = df["country"].astype(str)
        df["year"] = df["year"].astype(int)

    # ---- merge step by step on (country, year) ----
    panel = (
        viirs
        .merge(gdp,   on=["country", "year"], how="left")
        .merge(pop,   on=["country", "year"], how="inner")
        .merge(land,  on=["country", "year"], how="inner")
        .merge(urban, on=["country", "year"], how="inner")
    )

    # Save final panel
    out_path = processed_dir / "model_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"Saved merged model panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    build_model_panel()
