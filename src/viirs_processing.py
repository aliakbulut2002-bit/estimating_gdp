# src/viirs_processing.py
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

from src.config import BASE_DIR, TARGET_COUNTRIES

# Natural Earth Admin 0 – Countries shapefile (country polygons used for raster aggregation).
SHAPE_PATH = BASE_DIR / "data" / "raw" / "ne_10m_admin_0_countries.shp"

# 3-letter country code attribute used as the merge key for country-level panels.
# (This follows the original implementation and assumes consistency with other project panels.)
CODE_COL = "SOV_A3"


def compute_viirs_means_for_year(year: int) -> pd.DataFrame:
    """
    Compute country-level mean VIIRS night-time lights intensity for a single year.

    This function summarizes an annual VIIRS raster (`viirs_{year}.tif`) to the country level by:
      1) reading country polygons from the Natural Earth Admin-0 dataset,
      2) restricting polygons to the project’s country scope (`TARGET_COUNTRIES`),
      3) masking the raster to each country geometry,
      4) excluding raster NoData values,
      5) computing the mean intensity over valid pixels.

    Parameters
    ----------
    year : int
        Calendar year corresponding to the raster file to be processed.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per country for the specified year, containing:
          - country : str
              Country identifier (taken from the shapefile field `SOV_A3`).
          - year : int
              Calendar year.
          - mean_light : float
              Mean night-time lights intensity for the country-year.

    Notes
    -----
    - The aggregation statistic is the arithmetic mean across pixels intersecting the country polygon.
      This is a standard baseline measure; alternative summaries (median, trimmed mean, etc.) could
      be considered depending on the distributional properties of the raster product.
    - The masking operation assumes that the raster and polygon geometries share a compatible CRS.
      If inputs are not aligned, results may be incorrect; CRS harmonization can be added if needed.
    """
    viirs_path = BASE_DIR / "data" / "raw" / f"viirs_{year}.tif"
    print(f"Reading VIIRS raster from: {viirs_path}")

    # Read country polygons and restrict to the project’s country scope.
    gdf = gpd.read_file(SHAPE_PATH)
    gdf = gdf[gdf[CODE_COL].isin(TARGET_COUNTRIES)].copy()

    results: list[dict] = []

    with rasterio.open(viirs_path) as src:
        for _, row in gdf.iterrows():
            iso3 = row[CODE_COL]
            geom = [row["geometry"]]

            # Spatially mask and crop the raster to the country polygon.
            out_image, _ = mask(src, geom, crop=True)
            data = out_image[0]

            # Replace NoData values with NaN to facilitate numeric aggregation.
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            # Compute the mean over valid pixels only.
            valid = data[~np.isnan(data)]
            mean_light = float(valid.mean()) if valid.size > 0 else np.nan

            results.append(
                {
                    "country": iso3,
                    "year": year,
                    "mean_light": mean_light,
                }
            )

    return pd.DataFrame(results)


def build_viirs_panel(first_year: int = 2012, last_year: int = 2024) -> Path:
    """
    Construct a multi-year country–year panel of VIIRS night-time lights intensity.

    The function iterates over the requested year range, computes country-level means for each year
    where the corresponding raster is available, concatenates annual outputs into a single panel,
    and writes the result to disk.

    Parameters
    ----------
    first_year : int, default 2012
        First year (inclusive) to attempt to process.
    last_year : int, default 2024
        Last year (inclusive) to attempt to process.

    Returns
    -------
    pathlib.Path
        Path to the processed CSV saved under `BASE_DIR/data/processed/viirs_panel.csv`.

    Output schema
    -------------
      - country : str
          Country identifier (from the shapefile field `SOV_A3`).
      - year : int
          Calendar year.
      - mean_light : float
          Mean VIIRS intensity for the country-year.

    Notes
    -----
    Missing raster years are skipped rather than causing the full pipeline to fail. This supports
    partial local availability of the VIIRS archive while maintaining a reproducible workflow.
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

    out_path = BASE_DIR / "data" / "processed" / "viirs_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"Saved VIIRS panel to: {out_path}")
    return out_path


if __name__ == "__main__":
    # Enable reproducible module execution (e.g., `python -m src.viirs_processing`).
    # The year range here should match the set of VIIRS rasters available locally.
    build_viirs_panel(first_year=2012, last_year=2021)
