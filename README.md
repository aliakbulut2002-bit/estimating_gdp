# Estimating GDP per Capita from Night-Time Lights: Model Comparison

## Research Question
Can we estimate GDP per capita in Eritrea, where official GDP data have been unavailable since 2011, using non-economic data—such as night-time lights intensity and auxiliary variables (population, urban population share, and land area)—and which machine learning approach provides the best results ?

## Method 
Models are trained on a multi-country, country–year panel and then used to generate out-of-sample GDP-per-capita estimates for Eritrea using night-time lights and auxiliary covariates, including population, land area and urban population share (urban population as a percentage of total population).



## Setup

The script was tested on several 64-bit Windows machines and on a Mac, and it ran consistently. Please follow these requirements; otherwise main.py may fail:

Python 3.11.x (3.11.9 recommended)

64-bit Windows, or macOS 11 (Big Sur) or newer

If the script does not run properly, this is most likely because you are using a different Python version, or because your macOS version is too old (typically older than ~6 years, i.e., pre-2020 / older than macOS 11).

# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
python main.py

# macOS/Linux 

python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
pip install --prefer-binary -r requirements.txt
python main.py


## Outputs
Running `python main.py` writes all figures to the `results/` directory:

- `comparison_test_r2.png` — Bar chart comparing test R² across models  
- `comparison_test_rmse.png` — Bar chart comparing test RMSE across models  
- `model_comparison.png` — Combined model comparison figure  
- `run_summary.png` — Summary of the best-performing model and key performance indicators  
- `nightlights_vs_log_gdp_zoom_0_1.png` — Relationship between night lights and log GDP per capita  
- `true_vs_est_random_forest.png` — True vs. predicted plot for Random Forest  
- `true_vs_est_gradient_boosted_trees.png` — True vs. predicted plot for Gradient Boosted Trees  
- `true_vs_est_knn.png` — True vs. predicted plot for KNN  
- `true_vs_est_linear_regression.png` — True vs. predicted plot for Linear Regression  
- `eritrea_gdp_estimation.png` — Eritrea GDP-per-capita model-based estimate for 2012–2021  


## Project Structure

ESTIMATING_GDP-1/
├── main.py                                 # Main entry point
├── src/                                    # Source code
│   ├── config.py                           # Paths, country list and year range.
│   ├── data_loader.py                      # Loads the merged panel, splits train/test.
│   ├── gdp_processing.py                   # Raw World Bank GDP data processing
│   ├── population_processing.py            # Raw World Bank population data processing
│   ├── urbanpop_processing.py              # Raw World Bank urban population processing
│   ├── land_area_processing.py             # Raw World Bank land area processing
│   ├── viirs_processing.py                 # Night-time lights (VIIRS data) processing,
│   ├── merge_data.py                       # Merges processed panels into a single dataset.
│   ├── models.py                           # Defines and trains the models (RF, GBDT, KNN, Linear).
│   └── evaluation.py                       # Computes RMSE/R², generates the plots.
├── data/
│   ├── raw/                                # Raw input datasets (Not present on the Repo,sent by DB)
│   └── processed/                          # Cleaned/intermediate datasets 
├── results/                                # Output plots and figures
├── notebooks/                              # Produces optional charts and tables 
├── requirements.txt                        # Dependencies (pinned)
└── README.md                               # Documentation

## Results
- Random Forest: Test R² = 0.964, Test RMSE = 0.281  
- Gradient Boosted Trees: Test R² = 0.931, Test RMSE = 0.389  
- KNN: Test R² = 0.856, Test RMSE = 0.562  
- Linear Regression: Test R² = 0.635, Test RMSE = 0.892  
- Winner: Random Forest (highest R² and lowest RMSE)

## Requirements
- Python 3.11.9 (recommended for `rasterio` compatibility)

All dependencies are pinned in `requirements.txt`. Core packages include:

- Geospatial: `rasterio==1.4.3`, `geopandas==0.14.4`, `fiona==1.10.1`, `shapely==2.1.2`, `pyproj==3.7.2`
- Machine learning / stats: `scikit-learn==1.7.2`, `scipy==1.16.3`, `statsmodels==0.14.5`
- Data / plotting: `pandas==2.2.2`, `numpy==1.26.4`, `matplotlib==3.10.7`


## Important Notes
- **Python version requirement:** This project must be run with **Python 3.11.9** (or an equivalent version that supports `rasterio`). Using an incompatible Python version may prevent `rasterio` from installing or running, and the pipeline will fail.
- **Raw data:** The data/raw/ folder (unprocessed source files) is not included in the repository. You can run main.py without it and it will give you all the results, but the preprocessing scripts require the raw data to be available locally. You can download the raw data file from this dropbox link https://www.dropbox.com/scl/fo/k7iofxn8yzlmncrqy1hns/AJTD-RV0YSCW2zGtMczPGdQ?rlkey=vy0j3ocu3efsabs4uf0rat32l&st=df8tc69p&dl=0 , unzip it and add it to the "data" folder under the name "raw" if you intend to run the preprocessing scripts.
- **Data availability constraint:** Because the locally available night-time lights data cover **2012–2021**, the Eritrea GDP-per-capita estimates are limited to this period and cannot be extended to **2024** within the current setup.

