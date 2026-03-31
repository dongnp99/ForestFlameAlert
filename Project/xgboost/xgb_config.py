import numpy as np

# =============================
# DATA PATH
# =============================

# DATA_PATH = "../data/Daklak/final_inputs/dataset_fire_final.csv"
DATA_PATH = "../data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet"

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

# =============================
# HUMAN FIRE SAMPLE WEIGHTING
# =============================

# Graduated weights — see compute_sample_weights() for the full schedule.
# Kept for logging in train scripts (reports how many samples get weight > 1).
HUMAN_FIRE_MULTIPLIER = 1.5   # minimum non-trivial weight (1-pathway match)


def compute_sample_weights(df):
    """Return a float32 weight array for training rows in *df*.

    V4 pathway logic — graduated weights based on how many independent
    human-fire pathways a sample matches. Non-fire rows always get 1.0.

    Pathways (OR across, AND within):
      P1 — post-harvest agricultural burning:
           burn_season_flag == 1  AND  days_since_harvest < 30
      P2 — active deforestation with fire history:
           deforestation_lag_1y > 1.5  AND  fire_count_prev_year > 0
      P3 — repeat ignition during burn season:
           fire_count_prev_year > 1  AND  burn_season_flag == 1

    Weight schedule (score = number of pathways matched):
      0 pathways → 1.0x   (weather-driven or unclassified)
      1 pathway  → 1.5x
      2 pathways → 2.5x
      3 pathways → 3.5x

    Coverage on training data: ~44% of fire=1 samples get weight > 1.0.

    *df* must still contain the "fire" column when this is called.
    """
    fire = (df["fire"] == 1).values

    p1 = (
        (df["burn_season_flag"] == 1)
        & (df["days_since_harvest"] < 30)
    ).values

    p2 = (
        (df["deforestation_lag_1y"] > 1.5)
        & (df["fire_count_prev_year"] > 0)
    ).values

    p3 = (
        (df["fire_count_prev_year"] > 1)
        & (df["burn_season_flag"] == 1)
    ).values

    score = (p1.astype(np.int8) + p2.astype(np.int8) + p3.astype(np.int8))

    weights_map = np.array([1.0, 1.5, 2.5, 3.5], dtype="float32")
    weights = np.ones(len(df), dtype="float32")
    weights[fire] = weights_map[score[fire]]
    return weights

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

    # ===== Human Activity =====
    "dist_road_km",
    "dist_settlement_km",
    "dist_forest_edge_km",
    "dist_powerline_km",
    "lulc_class",
    "cropland_frac_1km",
    "tree_cover_pct",
    "deforestation_lag_1y",
    "nightlight_mean",
    "pop_density",
    "fire_count_prev_year",
    "fire_count_prev_3y",
    "fire_freq_5y",
    "days_since_last_fire",
    "burn_season_flag",
    "days_since_harvest",
]
