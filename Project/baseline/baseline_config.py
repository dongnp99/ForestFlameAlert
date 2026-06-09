"""
baseline_config.py
Shared configuration for Logistic Regression and Random Forest baselines.
"""

import csv
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================
# PATHS
# ==============================

BASELINE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR  = BASELINE_DIR / "results"
DATA_PATH    = (BASELINE_DIR / "../data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet").resolve()

# ==============================
# TIME SPLIT
# ==============================

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"

# ==============================
# SETTINGS
# ==============================

RANDOM_STATE = 42

# sklearn LR và RF không dùng GPU — 4GB GPU chưa có tác dụng ở đây
LR_MAX_TRAIN_SAMPLES = 1_000_000
RF_MAX_TRAIN_SAMPLES = 800_000   # tăng từ 300K: 24GB RAM dư sức chứa

FEATURE_COLS = [
    # Weather
    "tmean", "rh", "wind", "rain", "vpd",
    # Rolling weather
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    # Fire history
    "fire_lag_1", "fire_lag_3",
    # Terrain
    "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    # Seasonality
    "sin_doy", "cos_doy",
    # Fire adjacency
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
    # Vegetation (Sentinel-2)
    "ndvi", "ndvi_std", "ndwi", "nbr", "ndii",
    "delta_ndvi_14d", "delta_nbr_7d", "has_s2",
    # Human activity
    "dist_road_km", "dist_settlement_km", "dist_forest_edge_km", "dist_powerline_km",
    "lulc_class", "cropland_frac_1km", "tree_cover_pct", "deforestation_lag_1y",
    "nightlight_mean", "pop_density",
    "fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y",
    "days_since_last_fire", "burn_season_flag", "days_since_harvest",
]

# ==============================
# DATA LOADING
# ==============================

def load_split(date_filters, max_samples=None):
    """Load a time-filtered split. Optionally random-sample max_samples rows.

    Dùng random sampling (không phải temporal) để tập train đại diện đều
    cho toàn bộ giai đoạn, tránh fire rate bị lệch khi lấy rows gần nhất.
    """
    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire", "date"],
        filters=date_filters,
        engine="pyarrow",
    )
    df["fire"] = df["fire"].astype("int8")
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if max_samples is not None and len(df) > max_samples:
        rng      = np.random.default_rng(RANDOM_STATE)
        keep_idx = rng.choice(len(df), size=max_samples, replace=False)
        keep_idx.sort()
        df = df.iloc[keep_idx].copy()

    return df.drop(columns=["date"])

# ==============================
# LOGGING
# ==============================

def setup_logging(model_name: str, run_id: str) -> logging.Logger:
    """Logger ghi ra stdout và file results/<model_name>_<run_id>.log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{model_name}_{run_id}.log"

    logger = logging.getLogger(f"{model_name}_{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Log: %s", log_path)
    return logger

# ==============================
# COMPARISON CSV
# ==============================

COMPARISON_CSV  = RESULTS_DIR / "comparison.csv"
_COMPARE_FIELDS = [
    "run_id", "model", "train_samples",
    "val_aucpr", "val_rocauc",
    "test_aucpr", "test_rocauc",
    "train_time_s", "notes",
]

def append_comparison(row: dict):
    """Thêm một dòng kết quả vào comparison.csv (tạo file + header nếu chưa có)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not COMPARISON_CSV.exists()
    with open(COMPARISON_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COMPARE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _COMPARE_FIELDS})