# =============================
# DATA PATH
# =============================

# DATA_PATH = "../data/Daklak/final_inputs/dataset_fire_final.csv"
DATA_PATH = "../data/Daklak/final_inputs/daklak_final_dataset.parquet"

# =============================
# TIME SPLIT
# =============================

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
TEST_END_DATE  = "2024-12-31"

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

    # ===== Rolling =====
    "rain_14d_sum",
    "rain_30d_sum",
    "vpd_14d_mean",
    "vpd_30d_mean",

    # ===== Lag quan trọng =====
    "fire_lag_1",
    "fire_lag_3",

    # ===== Terrain =====
    "dem_mean",
    "dem_stdev",
    "slp_mean",
    "slp_stdev",

    # ===== Seasonality =====
    "sin_doy",
    "cos_doy",

    # ===== Fire Adjacency =====
    "neighbor_count",
    "neighbor_fire_1d",
    "neighbor_fire_3d",
    "neighbor_fire_7d",
    "vpd_neighbor_1d",
    "vpd_fire_lag_1",

    # ===== Vegetation (Sentinel-2) =====
    "ndvi",
    "ndvi_std",
    "ndwi",
    "nbr",
    "ndii",
    "delta_ndvi_14d",
    "delta_nbr_7d",
    "has_s2",
]
