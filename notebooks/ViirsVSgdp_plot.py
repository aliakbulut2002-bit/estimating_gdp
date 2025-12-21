import pandas as pd
import matplotlib.pyplot as plt
from src.config import BASE_DIR
import numpy as np

PANEL_PATH = BASE_DIR / "data" / "processed" / "model_panel.csv"
df = pd.read_csv(PANEL_PATH)
df = df.rename(columns={"gdp_pcap (US dollars)": "gdp_pcap"})
df = df.dropna(subset=["mean_light", "gdp_pcap"])

x_raw = df["mean_light"].values
y = np.log(df["gdp_pcap"].values)

# keep only x in [0,1] (zoom)
mask = (x_raw >= 0) & (x_raw <= 1)
x_raw = x_raw[mask]
y = y[mask]

# transform x for fitting (and plotting)
x = np.log1p(x_raw)

plt.figure(figsize=(8,5))
plt.scatter(x, y, alpha=0.35, s=18)

m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = m * x_line + b
plt.plot(x_line, y_line, color="red", linewidth=2)

plt.xlabel("log(1 + mean_light)  (zoom: mean_light 0–1)")
plt.ylabel("log(GDP per capita)")
plt.title("Night lights vs log GDP per capita (with linear trend)")
plt.tight_layout()
plt.show()
