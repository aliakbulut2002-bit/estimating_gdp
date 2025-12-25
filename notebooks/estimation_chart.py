import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path

# --- 1) Your estimates ---
df = pd.DataFrame(
    {
        "year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        "gdp_pcap_est_usd": [902, 905, 883, 862, 879, 881, 877, 870, 862, 836],
    }
).sort_values("year")

# --- 2) Robust output path: save PNG into ./notebooks/ ---
cwd = Path.cwd()
project_root = cwd if (cwd / "src").exists() else (cwd.parent if (cwd.parent / "src").exists() else cwd)
out_dir = project_root / "notebooks"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "eritrea_gdp_per_capita_estimates_statista_style.png"

# --- 3) Statista-like styling ---
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

years = df["year"].tolist()
vals = df["gdp_pcap_est_usd"].tolist()
x = list(range(len(years)))

# light alternating vertical bands
for i in range(len(x)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="black", alpha=0.03, zorder=0)

# bars
ax.bar(x, vals, width=0.78, color="#0B5ED7", edgecolor="white", linewidth=0.8, zorder=3)

# labels
ax.set_ylabel("GDP per capita (US$)")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")

# y-axis formatting
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))

# --- Make variations more visible (single plot, no inset) ---
y_min, y_max = min(vals), max(vals)
pad = 20  # adjust if you want a tighter/wider window
low = int((y_min - pad) // 10 * 10)
high = int(((y_max + pad + 9) // 10) * 10)
ax.set_ylim(low, high)

# more granular ticks for this narrower range
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))

# gridlines
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=1)
ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.4, alpha=0.25, zorder=1)
ax.xaxis.grid(False)

# clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# footer (optional)
footer = "Details: Eritrea; model-based estimates; 2012 to 2021"
ax.text(0.0, -0.16, footer, transform=ax.transAxes, va="top", ha="left", fontsize=9, alpha=0.75)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_path}")
