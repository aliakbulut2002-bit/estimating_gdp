from pathlib import Path
import pandas as pd

from src.config import TRAINING_COUNTRIES, BASE_DIR


def build_gdp_panel(
    raw_filename: str = "gdp_per_capita_raw.xlsx",
    first_year: int = 2012,
) -> Path:
    """
    Build clean GDP per capita panel:
    columns: country, year, gdp_pcap
    """
    raw_path = BASE_DIR / "data" / "raw" / "gdp" / raw_filename
    print(f"Reading raw GDP file from: {raw_path}")

    df_raw = pd.read_excel(
        raw_path,
        sheet_name="Data",
        skiprows=3,
    )

    year_cols = [
        c for c in df_raw.columns
        if isinstance(c, str) and c.isdigit() and int(c) >= first_year
    ]

    df = df_raw[["Country Name", "Country Code"] + year_cols].copy()
    df = df[df["Country Code"].isin(TRAINING_COUNTRIES)].copy()

    df_long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_pcap (US dollars)",
    )

    df_long["year"] = df_long["year"].astype(int)
    df_long.rename(columns={"Country Code": "country"}, inplace=True)
    df_long = df_long.dropna(subset=["gdp_pcap (US dollars)"])

    panel = df_long[["country", "year", "gdp_pcap (US dollars)"]].sort_values(["country", "year"])

    out_path = BASE_DIR / "data" / "processed" / "gdp_africa_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    print(f"Saved cleaned GDP panel to {out_path}")
    return out_path


if __name__ == "__main__":
    # so that `python -m src.gdp_processing` actually DOES something
    build_gdp_panel()
