# src/config.py
from pathlib import Path

# Repository root directory (i.e., the project’s top-level folder containing `main.py` and `src/`).
BASE_DIR = Path(__file__).resolve().parent.parent

# Universe of ISO3 country codes defining the geographic scope of the analysis.
# Eritrea (ERI) is included as a held-out prediction target.
TARGET_COUNTRIES = [
    "ERI",  # Eritrea (held-out prediction target)

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

    # --- Asia ---
    # East Asia
    "CHN", "JPN", "KOR", "MNG",
    # South Asia
    "IND", "PAK", "BGD", "LKA", "NPL", "AFG",
    # Southeast Asia
    "IDN", "VNM", "THA", "MYS", "PHL", "SGP", "KHM", "LAO", "MMR",
    # Central Asia
    "KAZ", "KGZ", "TJK", "TKM", "UZB",
]


# Definition of the training sample: all analysis countries excluding Eritrea, which is left out for estimation.
TRAINING_COUNTRIES = [c for c in TARGET_COUNTRIES if c != "ERI"]

# Administrative boundary dataset used to spatially aggregate raster values to the country level.
# `ISO_COL` identifies the attribute containing ISO3-equivalent codes within the shapefile.
SHAPE_PATH = BASE_DIR / "data" / "raw" / "ne_10m_admin_0_countries.shp"
ISO_COL = "SOV_A3"

# Temporal coverage of the analysis (inclusive).
YEARS = list(range(2012, 2023))  # 2012–2022 inclusive


