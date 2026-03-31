"""
precompute_predictions.py
=========================
Run XGBoost inference on the 2023–2024 test period and save results to
a single parquet file that app.py reads for display.

Columns: date, grid_id, lat, lon, fire_prob, fire
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_PATH  = BASE / "Project/xgboost/models/v3/xgb_human_features_tuned_v4_pathways.json"
DATA_PATH   = BASE / "Project/data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet"
COORDS_PATH = BASE / "Project/data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"
OUT_PATH    = BASE / "app_predictions.parquet"

FEATURE_COLS = [
    "tmean", "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    "fire_lag_1", "fire_lag_3",
    "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    "sin_doy", "cos_doy",
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
    "ndvi", "ndvi_std", "ndwi", "nbr", "ndii", "delta_ndvi_14d", "delta_nbr_7d",
    "has_s2",
    "dist_road_km", "dist_settlement_km", "dist_forest_edge_km", "dist_powerline_km",
    "lulc_class", "cropland_frac_1km", "tree_cover_pct", "deforestation_lag_1y",
    "nightlight_mean", "pop_density",
    "fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y",
    "days_since_last_fire", "burn_season_flag", "days_since_harvest",
]

DATE_CHUNK = 30  # days per chunk — reduce if OOM

# ── Load model ─────────────────────────────────────────────────────────────────
logging.info("Loading model from %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
logging.info("Model loaded.")

# ── Load grid coordinates ──────────────────────────────────────────────────────
coords = pd.read_csv(COORDS_PATH)
logging.info("Loaded %d grid coordinate rows.", len(coords))

# ── Get unique dates in test period (2023–2024) ────────────────────────────────
logging.info("Reading unique dates for 2023–2024 …")
date_df = pd.read_parquet(
    DATA_PATH,
    columns=["date"],
    filters=[
        ("date", ">=", pd.Timestamp("2023-01-01")),
        ("date", "<=", pd.Timestamp("2024-12-31")),
    ],
    engine="pyarrow",
)
all_dates = np.sort(date_df["date"].unique())
del date_df
gc.collect()
logging.info("Total dates: %d", len(all_dates))

# ── Chunked inference ──────────────────────────────────────────────────────────
results = []
n_chunks = int(np.ceil(len(all_dates) / DATE_CHUNK))

for i in range(n_chunks):
    chunk_dates = all_dates[i * DATE_CHUNK : (i + 1) * DATE_CHUNK]
    chunk_df = pd.read_parquet(
        DATA_PATH,
        columns=["date", "grid_id", "fire"] + FEATURE_COLS,
        filters=[
            ("date", ">=", pd.Timestamp(chunk_dates[0])),
            ("date", "<=", pd.Timestamp(chunk_dates[-1])),
        ],
        engine="pyarrow",
    )

    for col in FEATURE_COLS:
        if col in chunk_df.columns:
            chunk_df[col] = chunk_df[col].astype("float32")

    dmat = xgb.DMatrix(chunk_df[FEATURE_COLS])
    chunk_df["fire_prob"] = model.predict(dmat).astype("float32")

    results.append(chunk_df[["date", "grid_id", "fire", "fire_prob"]])

    del chunk_df, dmat
    gc.collect()

    if (i + 1) % 5 == 0 or i == n_chunks - 1:
        logging.info("  Chunks done: %d / %d", i + 1, n_chunks)

# ── Combine, join coordinates, save ───────────────────────────────────────────
logging.info("Concatenating chunks …")
pred_df = pd.concat(results, ignore_index=True)
del results
gc.collect()

pred_df = pred_df.merge(coords, on="grid_id", how="left")
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df.sort_values(["date", "grid_id"], inplace=True)
pred_df.reset_index(drop=True, inplace=True)

logging.info("Saving %d rows to %s …", len(pred_df), OUT_PATH)
pred_df.to_parquet(OUT_PATH, index=False)
logging.info("Done. Output: %s", OUT_PATH)