# src/config.py
from pathlib import Path

# Root folder of your project (where main.py / src live)
BASE_DIR = Path(__file__).resolve().parent.parent

# All countries for which you want night-lights data (including Eritrea, our prediction target)
TARGET_COUNTRIES = [
    "ERI",  # Eritrea (prediction target)

    # --- Africa ---
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CAF", "TCD",
    "COM", "COD", "COG", "CIV", "DJI", "EGY", "GNQ", "SWZ", "ETH",
    "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY",
    "MDG", "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER",
    "NGA", "RWA", "SEN", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA",
    "TGO", "TUN", "UGA", "ZMB", "ZWE",

    # --- Europe ---
    "ALB", "ARM", "AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU",
    "NLD", "NOR", "POL", "PRT", "ROU", "ESP", "SWE", "CHE", "GBR",

    # --- Middle East ---
    "BHR", "IRN", "IRQ", "JOR", "KWT", "LBN", "OMN", "QAT", "SAU",
    "SYR", "TUR", "ARE", "YEM",

    # --- Asia (added) ---
    # East Asia
    "CHN", "JPN", "KOR", "MNG",
    # South Asia
    "IND", "PAK", "BGD", "LKA", "NPL", "AFG",
    # Southeast Asia
    "IDN", "VNM", "THA", "MYS", "PHL", "SGP", "KHM", "LAO", "MMR",
    # Central Asia
    "KAZ", "KGZ", "TJK", "TKM", "UZB",
]



# Countries that DO have GDP data (used for training)
# -> all TARGET_COUNTRIES except Eritrea
TRAINING_COUNTRIES = [c for c in TARGET_COUNTRIES if c != "ERI"]

# Column name in your shapefile that contains these codes
# If you're still using Natural Earth, this is likely "SOV_A3" or "ADM0_A3",
# not "ISO3". Adjust to match your shapefile.
ISO_COL = "ISO3"

# Years to process (adapt if you have more VIIRS years)
YEARS = list(range(2012, 2023))  # 2012–2022 inclusive


def viirs_raster_path(year: int) -> Path:
    """Path to VIIRS raster for a given year."""
    return BASE_DIR / "data" / "raw" / f"viirs_{year}.tif"


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
