# src/config.py
from pathlib import Path

# Root folder of your project (where main.py is)
BASE_DIR = Path(__file__).resolve().parent.parent

# All countries for which you want night-lights data (including Eritrea)
TARGET_COUNTRIES = [
    "ERI",  # Eritrea (no GDP, we'll PREDICT for it)
    "ETH",
    "DJI",
    "SOM",
    "SDN",
    "KEN",
    "UGA",
    "TZA",
    "RWA",
    "BDI",
    "TCD",
    "CAF",
    "MOZ",
]

# Countries that DO have GDP data (used for training)
TRAINING_COUNTRIES = [c for c in TARGET_COUNTRIES]

# Column name in your shapefile that contains these codes
ISO_COL = "ISO3"  # change if your shapefile uses a different name

# Years to process
YEARS = list(range(2012, 2023))  # 2012–2022 inclusive


def viirs_raster_path(year: int) -> Path:
    # Files are directly inside data/raw, e.g. data/raw/viirs_2012.tif
    return (
        BASE_DIR
        / "data"
        / "raw"
        / f"viirs_{year}.tif"
    )



def countries_shapefile_path() -> Path:
    """Path to the countries shapefile."""
    return BASE_DIR / "data" / "raw" / "boundaries" / "countries.shp"


def per_year_output_dir() -> Path:
    """Folder where per-year VIIRS CSVs will be stored."""
    return BASE_DIR / "data" / "processed" / "viirs_country_year"


def panel_output_path() -> Path:
    """Path to VIIRS panel CSV."""
    return BASE_DIR / "data" / "processed" / "viirs_country_panel.csv"


def gdp_panel_path() -> Path:
    """Path to cleaned GDP panel CSV (output of gdp_processing)."""
    return BASE_DIR / "data" / "processed" / "gdp_africa_panel.csv"
