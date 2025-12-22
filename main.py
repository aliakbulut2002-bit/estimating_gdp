# main.py
"""
Main entry point for the project.

This script provides a single, reproducible execution pathway for the full workflow:
    raw data -> processed country-year tables -> merged modeling panel
             -> model training and comparative evaluation
             -> out-of-sample GDP-per-capita estimation for Eritrea

Intended usage (from the repository root):
    python main.py

Outputs:
    - Intermediate processed datasets: data/processed/
    - Final artifacts (tables/figures/estimates): results/
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DIR
from src.data_loader import load_and_split
from src.models import train_random_forest, train_knn, train_linear, train_gbdt

from src.evaluation import (
    evaluate_one_model,
    build_results_table,
    print_results_table,
)

RANDOM_STATE = 42


def run_module(module_name: str) -> None:
    """Execute a project module via `python -m <module>` using the current interpreter."""
    print(f"  Running {module_name} ...")
    subprocess.run([sys.executable, "-m", module_name], check=True)


def run_data_pipeline() -> None:
    """
    Execute preprocessing modules in a standard dependency order and construct the modeling panel.
    """
    print("\n0) Building dataset from raw files...")

    run_module("src.viirs_processing")
    run_module("src.population_processing")
    run_module("src.urbanpop_processing")
    run_module("src.land_area_processing")
    run_module("src.gdp_processing")

    # Final merge (produces model_panel.csv consumed by src/data_loader.py)
    run_module("src.merge_data")

    print("   ✓ Data pipeline finished")


def save_table_png(df: pd.DataFrame, out_path: Path, title: str | None = None) -> None:
    """
    Save a pandas DataFrame as a readable PNG table.
    """
    df_disp = df.copy()

    # Convert all cells to strings (avoid float artifacts like 2012.000)
    for col in df_disp.columns:
        if pd.api.types.is_numeric_dtype(df_disp[col]):
            df_disp[col] = df_disp[col].map(
                lambda x: ""
                if pd.isna(x)
                else str(int(x))
                if float(x).is_integer()
                else str(x)
            )
        else:
            df_disp[col] = df_disp[col].astype(str)

    cell_text = df_disp.values.tolist()
    col_labels = df_disp.columns.tolist()

    # Auto-size columns based on content length
    col_max_lens = []
    for j, col in enumerate(col_labels):
        max_len = len(str(col))
        for i in range(len(cell_text)):
            max_len = max(max_len, len(str(cell_text[i][j])))
        col_max_lens.append(max_len)

    total = sum(col_max_lens) if sum(col_max_lens) > 0 else 1
    col_widths = [m / total for m in col_max_lens]

    nrows, ncols = df_disp.shape
    fig_w = min(18.0, max(10.0, 14.0 * sum(col_widths)))
    fig_h = min(10.0, max(2.5, 0.60 * nrows + 1.8))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)

    # Bold headers
    for j in range(ncols):
        table[(0, j)].set_text_props(weight="bold")

    # Left-align first column (model names / country codes)
    for i in range(1, nrows + 1):
        table[(i, 0)].set_text_props(ha="left")

    if title is not None:
        ax.set_title(title, pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _slug(s: str) -> str:
    """Create a filename-safe slug."""
    return (
        s.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("²", "2")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def save_true_vs_pred_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str = "True log GDP per capita",
    ylabel: str = "Estimated log GDP per capita",
) -> None:
    """Save a True vs Estimated scatter plot with a 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_bars_png(
    results_df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    bar_width: float = 0.45,
    zoom: bool = True,
) -> None:
    """
    Save a bar chart comparing models on a given metric.

    - Thin bars via `bar_width`
    - Optional y-axis zoom to reduce “visual exaggeration”
    """
    df = results_df.copy()
    if metric_col.endswith("rmse"):
        df = df.sort_values(metric_col, ascending=True)
    else:
        df = df.sort_values(metric_col, ascending=False)

    labels = df["model"].tolist()
    vals = df[metric_col].astype(float).values
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, vals, width=bar_width)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if zoom and vmax > vmin:
        if metric_col.endswith("r2"):
            lo = max(0.0, vmin - 0.05)
            hi = min(1.0, vmax + 0.05)
        else:
            pad = 0.10 * (vmax - vmin)
            lo = max(0.0, vmin - pad)
            hi = vmax + pad
        ax.set_ylim(lo, hi)

    for b, val in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_nightlights_vs_log_gdp_zoom_png(
    results_dir: Path,
    panel_path: Path | None = None,
    x_col: str = "mean_light",
    gdp_col_candidates: tuple[str, ...] = ("gdp_pcap", "gdp_pcap (US dollars)"),
) -> None:
    """
    Save a diagnostic scatter plot of night-time lights vs log GDP per capita (zoomed to mean_light in [0, 1]),
    including a fitted linear trend line in the transformed x-scale.

    This reads the merged model panel produced by `src.merge_data`.
    """
    if panel_path is None:
        panel_path = BASE_DIR / "data" / "processed" / "model_panel.csv"

    try:
        df = pd.read_csv(panel_path)
    except FileNotFoundError:
        print(f"   ! Warning: model_panel.csv not found at {panel_path}; skipping night-lights plot.")
        return

    # Resolve GDP column name
    gdp_col = None
    for c in gdp_col_candidates:
        if c in df.columns:
            gdp_col = c
            break

    if x_col not in df.columns or gdp_col is None:
        print(
            "   ! Warning: required columns for night-lights plot not found. "
            f"Need '{x_col}' and one of {gdp_col_candidates}. Skipping."
        )
        return

    df_plot = df[[x_col, gdp_col]].copy()
    df_plot = df_plot.dropna(subset=[x_col, gdp_col])

    # Ensure numeric types
    df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors="coerce")
    df_plot[gdp_col] = pd.to_numeric(df_plot[gdp_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[x_col, gdp_col])

    if len(df_plot) == 0:
        print("   ! Warning: no valid rows for night-lights plot after dropping NAs; skipping.")
        return

    x_raw = df_plot[x_col].values
    y = np.log(df_plot[gdp_col].values)

    # Zoom: keep only mean_light in [0, 1]
    msk = (x_raw >= 0) & (x_raw <= 1)
    x_raw = x_raw[msk]
    y = y[msk]

    if x_raw.size < 2:
        print("   ! Warning: insufficient data in mean_light range [0, 1] for night-lights plot; skipping.")
        return

    # Transform x for fitting and plotting
    x = np.log1p(x_raw)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.35, s=18)

    # Linear trend in transformed space
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color="red", linewidth=2)

    ax.set_xlabel("log(1 + mean_light)  (zoom: mean_light 0–1)")
    ax.set_ylabel("log(GDP per capita)")
    ax.set_title("Night lights vs log GDP per capita (with linear trend)")

    fig.tight_layout()
    out_path = results_dir / "nightlights_vs_log_gdp_zoom_0_1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"   ✓ Saved: {out_path.name}")


def main() -> None:
    print("=" * 70)
    print("GDP per capita regression: End-to-end pipeline")
    print("=" * 70)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 0) Build processed data and merged modeling panel from raw inputs
    run_data_pipeline()

    # NEW: save the night-lights vs log GDP scatter (zoomed) from model_panel.csv
    print("\n0b) Saving exploratory plot (night lights vs log GDP)...")
    save_nightlights_vs_log_gdp_zoom_png(results_dir=results_dir)

    # 1) Load split (Eritrea is held out)
    print("\n1) Loading modeling panel and splitting data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_eritrea,
        eritrea_meta,
    ) = load_and_split(test_size=0.2, random_state=RANDOM_STATE)

    print(f"   Train size: {X_train.shape}")
    print(f"   Test size:  {X_test.shape}")
    if X_eritrea is not None:
        print(f"   Eritrea rows kept for estimation: {X_eritrea.shape[0]}")

    # 2) Train models
    print("\n2) Training candidate models...")
    lr_model = train_linear(X_train, y_train)
    knn_model = train_knn(X_train, y_train, n_neighbors=15)
    gbdt_model = train_gbdt(X_train, y_train, random_state=RANDOM_STATE)
    rf_model = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
    print("   ✓ All models trained")

    # 3) Evaluate
    print("\n3) Evaluating models (target = log GDP per capita)...")
    models = [
        ("Linear regression", lr_model),
        ("kNN", knn_model),
        ("Gradient Boosted Trees", gbdt_model),
        ("Random Forest", rf_model),
    ]

    results = [evaluate_one_model(m, X_train, y_train, X_test, y_test, name) for name, m in models]
    results_df = build_results_table(results, sort_by="test_rmse")
    print_results_table(results_df)

    # Model comparison table (PNG only)
    model_table_png = results_dir / "model_comparison.png"
    save_table_png(
        results_df.round(3),
        model_table_png,
        title="Model comparison (target: log GDP per capita)",
    )
    print(f"\nSaved model comparison table to: {model_table_png}")

    # 4) Save figures
    print("\n4) Saving figures to results/ ...")

    # 4a) True vs Estimated (4 plots)
    for name, model in models:
        y_pred = model.predict(X_test)
        out_path = results_dir / f"true_vs_est_{_slug(name)}.png"
        save_true_vs_pred_png(
            y_true=y_test,
            y_pred=y_pred,
            out_path=out_path,
            title=f"{name}: True vs Estimated (test set)",
            ylabel=f"Estimated log GDP per capita ({name})",
        )
        print(f"   ✓ Saved: {out_path.name}")

    # 4b) Metric charts (2 plots)
    r2_png = results_dir / "comparison_test_r2.png"
    save_metric_bars_png(
        results_df=results_df,
        metric_col="test_r2",
        out_path=r2_png,
        title="Model comparison: Test R²",
        ylabel="R² (test)",
        bar_width=0.45,
        zoom=True,
    )
    print(f"   ✓ Saved: {r2_png.name}")

    rmse_png = results_dir / "comparison_test_rmse.png"
    save_metric_bars_png(
        results_df=results_df,
        metric_col="test_rmse",
        out_path=rmse_png,
        title="Model comparison: Test RMSE",
        ylabel="RMSE (test, log GDP per capita)",
        bar_width=0.45,
        zoom=True,
    )
    print(f"   ✓ Saved: {rmse_png.name}")

    # 5) Choose best model
    winner_row = results_df.iloc[0]
    winner_name = str(winner_row["model"])
    best_model = dict(models)[winner_name]

    print("\n" + "=" * 70)
    print("Leaderboard (sorted by Test RMSE; lower is better):")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:24s}  RMSE={row['test_rmse']:.3f}  R²={row['test_r2']:.3f}")
    print("-" * 70)
    print(f"Best model: {winner_name} (Test RMSE = {winner_row['test_rmse']:.3f})")
    print("=" * 70)

    # 6) Run summary (PNG)
    summary_df = pd.DataFrame(
        {
            "Item": ["Best model (by test RMSE)", "Test RMSE", "Test R²"],
            "Value": [
                winner_name,
                f"{float(winner_row['test_rmse']):.6f}",
                f"{float(winner_row['test_r2']):.6f}",
            ],
        }
    )
    summary_png = results_dir / "run_summary.png"
    save_table_png(summary_df, summary_png, title="Run summary")
    print(f"Saved run summary table to: {summary_png}")

    # 7) Eritrea estimation table (PNG only) — ensure year is integer (no .000)
    if X_eritrea is not None and len(X_eritrea) > 0:
        print("\nEstimating GDP per capita for Eritrea...")
        y_log_eri = best_model.predict(X_eritrea)
        y_eri = np.exp(y_log_eri)

        est_df = eritrea_meta.copy()
        est_df["year"] = est_df["year"].astype(int)
        est_df["gdp_pcap_est_usd"] = y_eri

        est_for_png = est_df.copy()
        est_for_png["gdp_pcap_est_usd"] = est_for_png["gdp_pcap_est_usd"].map(lambda x: f"{x:,.0f}")

        eritrea_png = results_dir / "eritrea_gdp_estimation.png"
        save_table_png(
            est_for_png,
            eritrea_png,
            title="Eritrea: GDP per capita estimation (USD)",
        )

        print("\nEstimates for Eritrea (GDP per capita in USD):")
        for (country, year), gdp_pcap in zip(est_df[["country", "year"]].values, y_eri):
            print(f"{country} {int(year)}: {gdp_pcap:,.0f} USD")

        print(f"\nSaved Eritrea estimation table to: {eritrea_png}")
        print("=" * 70)
    else:
        print("\nNo Eritrea estimates: X_eritrea is None or empty.")


if __name__ == "__main__":
    main()
