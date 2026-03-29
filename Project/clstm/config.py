# config.py

from pathlib import Path

# =========================
# PATHS
# =========================

RAW_DATA_PATH = '../data/Daklak/final_inputs/daklak_final_dataset.parquet'
OUTPUT_DATA_PATH = '../data/Daklak/final_inputs/clstm_data'
OUTPUT_MODEL_PATH = 'models'
# =========================
# FEATURES
# =========================

FEATURE_COLUMNS = [
    "grid_id",
    "date",

    # Weather
    "tmean", "rh", "wind", "rain", "vpd",

    # Rolling dryness (14d only — 30d redundant with sequence window)
    "rain_14d_sum", "vpd_14d_mean",

    # Fire history (lag 1 & 3 — lag 7 covered by seq window)
    "fire_lag_1", "fire_lag_3",

    # Spatial spread (1d & 3d — 7d covered by seq window)
    "neighbor_count",
    "neighbor_fire_1d", "neighbor_fire_3d",

    # Seasonality
    "sin_doy", "cos_doy",

    # Vegetation — key indices only (ConvLSTM learns delta through sequence)
    "ndvi", "nbr",

    "fire"
]

TARGET_COLUMN = "fire"

# =========================
# CONVLSTM SETTINGS
# =========================

TIME_STEPS = 7

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"

# =========================
# GRID SETTINGS
# =========================

GRID_ID_COL = "grid_id"
DATE_COL = "date"