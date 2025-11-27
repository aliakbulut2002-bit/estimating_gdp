# src/viirs_processing.py
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

from src.config import BASE_DIR, TRAINING_COUNTRIES

# Natural Earth Admin 0 – Countries shapefile
SHAPE_PATH = BASE_DIR / "data" / "raw" / "ne_10m_admin_0_countries.shp"
CODE_COL = "SOV_A3"  # 3-letter country code column in this shapefile


def compute_viirs_means_for_year(year: int) -> pd.DataFrame:
    """
    Compute mean VIIRS light for each training country for a given year.

    Returns a DataFrame with columns: country, year, mean_light
    """
    viirs_path = BASE_DIR / "data" / "raw" / f"viirs_{year}.tif"
    print(f"Reading VIIRS raster from: {viirs_path}")

    # Read countries and keep only those in TRAINING_COUNTRIES
    gdf = gpd.read_file(SHAPE_PATH)
    gdf = gdf[gdf[CODE_COL].isin(TRAINING_COUNTRIES)].copy()

    results: list[dict] = []

    with rasterio.open(viirs_path) as src:
        for _, row in gdf.iterrows():
            iso3 = row[CODE_COL]
            geom = [row["geometry"]]

            out_image, _ = mask(src, geom, crop=True)
            data = out_image[0]

            # mask nodata
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            # compute mean on valid pixels
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                mean_light = np.nan
            else:
                mean_light = float(valid.mean())

            results.append(
                {
                    "country": iso3,
                    "year": year,
                    "mean_light": mean_light,
                }
            )

    df_year = pd.DataFrame(results)
    return df_year


def build_viirs_panel(first_year: int = 2012, last_year: int = 2024) -> Path:
    """
    Loop over years, compute mean light per country, and save panel CSV.
    """
    dfs: list[pd.DataFrame] = []

    for year in range(first_year, last_year + 1):
        try:
            df_year = compute_viirs_means_for_year(year)
            dfs.append(df_year)
        except FileNotFoundError:
            print(f"No VIIRS file found for year {year}, skipping.")

    if not dfs:
        raise RuntimeError("No VIIRS files found for any year in the range.")

    panel = pd.concat(dfs, ignore_index=True)

    out_path = BASE_DIR / "data" / "processed" / "viirs_africa_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"Saved VIIRS panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # for now only 2012 since that's the file you have
    build_viirs_panel(first_year=2012, last_year=2012)
