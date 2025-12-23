"""
notebooks/desc_stats.py

Purpose
-------
Compute descriptive statistics for numeric variables from the processed modeling
dataset and export the results as a static PNG table suitable for inclusion in
a written report.

Input
-----
- data/processed/model_panel.csv

Output
------
- notebooks/desc_stats.png

Scope
-----
- Numeric variables only.
- Statistics reported per variable: N (non-missing count), mean, standard
  deviation, minimum, median, and maximum.

Design choices
--------------
- A PNG artifact is produced to facilitate direct inclusion in the report.
- The implementation avoids pandas Styler to minimize optional dependencies.
- Formatting routines are included to improve readability for large-magnitude
  variables (e.g., population).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def repo_root() -> Path:
    """
    Return the repository root directory.

    The root is inferred from the script location:
    notebooks/desc_stats.py -> parent directory is treated as the repository root.
    """
    return Path(__file__).resolve().parents[1]


def prettify_variable_names(tab: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw variable identifiers with report-friendly labels.

    The GDP unit is indicated using a dollar sign in the label rather than
    spelling out the currency textually.
    """
    name_map = {
        "year": "Year",
        "mean_light": "Night-time lights (mean)",
        "viirs_mean": "Night-time lights (mean)",
        "ntl_mean": "Night-time lights (mean)",
        "gdp_pcap": "GDP per capita ($)",
        "gdp_per_capita": "GDP per capita ($)",
        "population": "Population",
        "land_area": "Land area (km^2)",
        "land_area_km2": "Land area (km^2)",
        "urban_pop_rate": "Urban population rate (%)",
        "urban_rate": "Urban population rate (%)",
    }

    tab2 = tab.copy()
    tab2 = tab2.rename(index=lambda s: name_map.get(s, s))
    tab2.index.name = "Variable"
    return tab2


def make_desc_table(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """
    Construct a descriptive statistics table for numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset as loaded from the processed modeling panel.
    exclude : list[str]
        Column names to be excluded from summarization (e.g., identifiers).

    Returns
    -------
    pd.DataFrame
        Table indexed by variable name and containing standard descriptive statistics.
    """
    # Exclude non-analytic columns when requested (e.g., country identifiers).
    df2 = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    # Restrict the summary to numeric variables for comparability across measures.
    num = df2.select_dtypes(include="number").replace([float("inf"), float("-inf")], pd.NA)

    if num.shape[1] == 0:
        raise ValueError("No numeric columns available for summarization after exclusions.")

    # Compute summary statistics using standard definitions.
    tab = pd.DataFrame(
        {
            "N": num.count(),
            "Mean": num.mean(),
            "Std": num.std(ddof=1),
            "Min": num.min(),
            "Median": num.median(),
            "Max": num.max(),
        }
    )

    # Apply report-oriented rounding to reduce visual clutter.
    tab = tab.round({"Mean": 2, "Std": 2, "Min": 2, "Median": 2, "Max": 2})
    tab.index.name = "Variable"

    # Replace internal variable identifiers with report-friendly labels.
    tab = prettify_variable_names(tab)

    return tab


def save_table_png(tab: pd.DataFrame, png_path: Path, title: str | None = None) -> None:
    """
    Render a descriptive statistics table as a static PNG using matplotlib.

    Parameters
    ----------
    tab : pd.DataFrame
        Descriptive statistics table indexed by variable name.
    png_path : Path
        Destination path for the PNG artifact.
    title : str | None
        Optional title rendered above the table.

    Notes
    -----
    The function is designed to produce a report-ready table:
    - variable labels are wrapped and left-aligned to avoid clipping;
    - numeric columns are right-aligned for readability;
    - large magnitudes are formatted with thousands separators to prevent
      digit crowding (e.g., population-related statistics);
    - column widths are allocated proportionally to content length.
    """
    import numpy as np

    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Move the index into a column to simplify table rendering.
    display_tab = tab.reset_index()
    first_col = display_tab.columns[0]

    # Matplotlib uses $ to delimit mathtext; escaping ensures literal rendering.
    display_tab[first_col] = display_tab[first_col].astype(str).str.replace("$", r"\$", regex=False)

    # Wrap variable labels to prevent spillover outside cell boundaries.
    display_tab[first_col] = display_tab[first_col].astype(str).apply(
        lambda s: "\n".join(textwrap.wrap(s, width=28))
    )

    def fmt_num(x, colname: str) -> str:
        """
        Format numeric values for compact, readable tabular display.

        - N is formatted as an integer.
        - Large magnitudes use thousands separators to enhance legibility.
        """
        if pd.isna(x):
            return ""
        try:
            x = float(x)
        except Exception:
            return str(x)

        if colname == "N":
            return f"{int(round(x))}"

        # Thousands separators are applied to large magnitudes to reduce visual crowding.
        if abs(x) >= 1_000_000:
            s = f"{x:,.2f}"
        else:
            s = f"{x:.2f}"

        # Remove trailing ".00" where appropriate to avoid redundant precision.
        if s.endswith(".00"):
            s = s[:-3]
        return s

    # Create a formatted copy for rendering (preserves numeric tab for any further use).
    formatted = display_tab.copy()
    for col in formatted.columns[1:]:
        formatted[col] = display_tab[col].apply(lambda v, c=col: fmt_num(v, c))

    # Convert all values to strings for stable matplotlib table rendering.
    cell_text = formatted.astype(str).values.tolist()
    col_labels = list(formatted.columns)

    nrows = len(formatted)
    ncols = len(col_labels)

    def max_line_len(s: str) -> int:
        """Return the maximum line length of a potentially multi-line string."""
        parts = str(s).splitlines()
        return max((len(p) for p in parts), default=0)

    # Allocate column widths based on the maximum visible content length.
    maxlens = []
    for j, col in enumerate(col_labels):
        col_vals = [col] + [row[j] for row in cell_text]
        if j == 0:
            maxlens.append(max(max_line_len(v) for v in col_vals) + 4)
        else:
            maxlens.append(max(len(str(v)) for v in col_vals) + 2)

    weights = np.array(maxlens, dtype=float)
    col_widths = (weights / weights.sum()).tolist()

    # Figure size is scaled to content to reduce overlap in dense columns.
    char_sum = float(weights.sum())
    fig_w = max(12.0, min(18.0, 0.12 * char_sum))
    fig_h = max(3.2, 0.50 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, pad=12)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    # Use a fixed font size to ensure consistent rendering across environments.
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)

    # Presentation conventions: labels left-aligned; numeric content right-aligned.
    for (r, c), cell in tbl.get_celld().items():
        cell.PAD = 0.08
        if c == 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_wrap(True)
        else:
            cell.get_text().set_ha("right")

    # Adjust row heights to accommodate wrapped labels without intersecting grid lines.
    base_h = tbl[(1, 0)].get_height() if (1, 0) in tbl.get_celld() else 0.05

    # Header row is slightly taller to improve legibility.
    for c in range(ncols):
        if (0, c) in tbl.get_celld():
            tbl[(0, c)].set_height(base_h * 1.2)

    # Data row heights scale with the number of wrapped lines in the first column.
    for r in range(1, nrows + 1):
        text = formatted.iloc[r - 1, 0]
        lines = str(text).count("\n") + 1
        row_h = base_h * max(1, lines) * 1.15
        for c in range(ncols):
            if (r, c) in tbl.get_celld():
                tbl[(r, c)].set_height(row_h)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Script entry point.

    The script reads the processed modeling panel, computes descriptive statistics,
    and exports a single PNG table to facilitate inclusion in the report.
    """
    root = repo_root()

    parser = argparse.ArgumentParser(description="Generate a PNG descriptive statistics table.")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Column names to exclude (e.g., iso3 country year).",
    )
    parser.add_argument(
        "--png",
        type=str,
        default=str(root / "notebooks" / "desc_stats.png"),
        help="Output path for the PNG table (default: notebooks/desc_stats.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Descriptive statistics (model panel)",
        help="Title displayed above the table.",
    )
    args = parser.parse_args()

    input_path = root / "data" / "processed" / "model_panel.csv"
    png_path = Path(args.png)

    df = pd.read_csv(input_path)
    tab = make_desc_table(df, exclude=args.exclude)
    save_table_png(tab, png_path, title=args.title)

    print(f"[OK] Wrote:  {png_path}")
    print("\nPreview:\n")
    print(tab.to_string())


if __name__ == "__main__":
    main()

