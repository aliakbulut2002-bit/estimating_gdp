# notebooks/ols.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import BASE_DIR  # so paths are consistent

def main():
    csv_path = BASE_DIR / "data" / "processed" / "gdp_viirs_africa_panel.csv"
    print(f"Reading merged panel from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Example: restrict to 2012
    df_2012 = df[df["year"] == 2012].copy()

    # Scatter plot
    plt.scatter(df_2012["mean_light"], df_2012["gdp_pcap (US dollars)"])
    plt.xlabel("Mean VIIRS light")
    plt.ylabel("GDP per capita (USD)")
    plt.title("GDP vs VIIRS mean light, 2012")
    plt.show()

    # Simple OLS: GDP on mean_light
    X = sm.add_constant(df_2012["mean_light"])
    y = df_2012["gdp_pcap (US dollars)"]

    model = sm.OLS(y, X).fit()
    print(model.summary())


if __name__ == "__main__":
    main()
