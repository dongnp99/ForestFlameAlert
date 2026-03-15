# config.py

from pathlib import Path

# =========================
# PATHS
# =========================

RAW_DATA_PATH = '../data/Daklak/final_inputs/daklak_fire_xgb_additional_features.parquet'
OUTPUT_DATA_PATH = '../data/Daklak/final_inputs/clstm_data'

# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [
    "grid_id",
    "date",

    "tmean", "rh", "wind", "rain", "vpd",

    "dem_mean", "slp_mean", "aspect_sin", "aspect_cos",

    "rain_14d_sum", "vpd_14d_mean",
    "rain_30d_sum", "vpd_30d_mean",

    "fire_lag_1", "fire_lag_3", "fire_lag_7",

    "neighbor_count",
    "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",

    "wind_vpd",
    "vpd_neighbor_1d",
    "vpd_fire_lag_1",

    "sin_doy", "cos_doy",

    "fire"
]

TARGET_COLUMN = "fire"

# =========================
# CONVLSTM SETTINGS
# =========================

TIME_STEPS = 7

TRAIN_YEARS = (2018, 2022)
VAL_YEAR = 2023
TEST_YEAR = 2024

# =========================
# GRID SETTINGS
# =========================

GRID_ID_COL = "grid_id"
DATE_COL = "date"