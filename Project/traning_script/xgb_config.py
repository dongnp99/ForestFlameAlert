# =============================
# DATA PATH
# =============================

# DATA_PATH = "../data/Daklak/final_inputs/dataset_fire_final.csv"
DATA_PATH = "../data/Daklak/final_inputs/daklak_fire_xgb_additional_features.parquet"

# =============================
# TIME SPLIT
# =============================

TRAIN_END_DATE = "2020-12-31"
VAL_END_DATE   = "2022-12-31"
TEST_END_DATE  = "2023-12-31"

# =============================
# MODEL SETTINGS
# =============================

RANDOM_STATE = 42

FEATURE_COLS = [

    # ===== Weather hiện tại =====
    "tmean",
    "rh",
    "wind",
    "rain",
    "vpd",

    # ===== Rolling mạnh =====
    "rain_14d_sum",
    "rain_30d_sum",
    "vpd_14d_mean",
    "vpd_30d_mean",

    # ===== Dryness =====
    "dryness_14d",
    "dryness_30d",
    "consecutive_dry_days",

    # ===== Interaction =====
    "wind_vpd",

    # ===== Lag quan trọng =====
    "fire_lag_1",
    "fire_lag_3",

    # ===== Terrain =====
    "dem_mean",
    "dem_stdev",
    "slp_mean",
    "slp_stdev",
    "aspect_sin",
    "aspect_cos",

    # ===== Seasonality =====
    "sin_doy",
    "cos_doy",

    # ===== Fire Adjacency =====
    "neighbor_count",
    "neighbor_fire_1d",
    "neighbor_fire_3d",
    "neighbor_fire_7d",
    "neighbor_fire_ratio_1d",
    "neighbor_fire_ratio_3d",
    "neighbor_fire_ratio_7d"
]
