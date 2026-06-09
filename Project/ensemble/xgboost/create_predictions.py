from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, ".")
from Project.xgboost.xgb_config import (
    compute_dominant_factor,
    classify_fire_type,
    SHAP_GROUPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROJECT        = Path(__file__).parent.parent.parent
MODEL_PATH  = PROJECT / "xgboost/models/v5/xgb_human_features_tuned_v5_pathways.json"
DATA_PATH   = PROJECT / "data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"
COORDS_PATH = PROJECT / "data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"
MAP_PATH    = "test_predictions_map.parquet"

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2024-12-31")

DATE_CHUNK = 30

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

#  Feature columns (must match model) 
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
    # NEW pathway features
    "is_pathway_p1", "is_pathway_p2", "is_pathway_p3",
    "human_pathway_score", "human_signal_strength",
]

SHAP_COLS = list(SHAP_GROUPS.keys())

#  SHAP group index mapping 
_feat_index = {f: i for i, f in enumerate(FEATURE_COLS)}
GROUP_INDICES = {
    g: [_feat_index[f] for f in feats if f in _feat_index]
    for g, feats in SHAP_GROUPS.items()
}

#  Load model ─
log.info("Loading model: %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
model.set_param("nthread", str(-1))

#  Load coordinates 
coords = pd.read_csv(COORDS_PATH)
log.info("Loaded %d grid coordinates", len(coords))

#  Scan dates ─
log.info("Scanning dates from %s to %s", START_DATE, END_DATE)
date_df = pd.read_parquet(
    DATA_PATH, columns=["date"],
    filters=[
        ("date", ">=", START_DATE),
        ("date", "<=", END_DATE),
    ],
    engine="pyarrow",
)
all_dates = np.sort(date_df["date"].unique())
del date_df; gc.collect()
log.info("Total dates: %d", len(all_dates))

#  Output columns ─
MAP_COLS = ["date", "grid_id", "fire_prob", "fire"]

#  Writers 
_map_writer:    Optional[pq.ParquetWriter] = None

#  Chunked inference 
n_chunks = int(np.ceil(len(all_dates) / DATE_CHUNK))

for i in range(n_chunks):
    chunk_dates = all_dates[i * DATE_CHUNK : (i + 1) * DATE_CHUNK]
    log.info("Chunk %d/%d  (%s → %s)",
             i + 1, n_chunks,
             pd.Timestamp(chunk_dates[0]).date(),
             pd.Timestamp(chunk_dates[-1]).date())

    #  Load ─
    read_cols = ["date", "grid_id", "fire"] + FEATURE_COLS
    chunk_df = pd.read_parquet(
        DATA_PATH, columns=read_cols,
        filters=[
            ("date", ">=", pd.Timestamp(chunk_dates[0])),
            ("date", "<=", pd.Timestamp(chunk_dates[-1])),
        ],
        engine="pyarrow",
    )
    chunk_df[FEATURE_COLS] = chunk_df[FEATURE_COLS].astype("float32")
    X = chunk_df[FEATURE_COLS].values

    #  Predict + SHAP 
    dmat     = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    contribs = model.predict(dmat, pred_contribs=True, approx_contribs=True)
    raw_score = contribs.sum(axis=1)
    chunk_df["fire_prob"] = _sigmoid(raw_score).astype("float32")

    shap_vals = contribs[:, :-1]
    np.abs(shap_vals, out=shap_vals)

    for group_name, indices in GROUP_INDICES.items():
        chunk_df[group_name] = shap_vals[:, indices].sum(axis=1).astype("float32")

    group_total = chunk_df[SHAP_COLS].sum(axis=1).clip(lower=1e-8)
    for col in SHAP_COLS:
        chunk_df[col] = (chunk_df[col] / group_total * 100).round(1).astype("float32")

    #  Build map rows 
    map_chunk = chunk_df[MAP_COLS].merge(coords, on="grid_id", how="left")
    map_chunk["date"]      = pd.to_datetime(map_chunk["date"])
    map_chunk["fire_prob"] = map_chunk["fire_prob"].astype("float32")
    map_chunk["fire"]      = map_chunk["fire"].astype("int8")
    map_chunk.sort_values(["date", "grid_id"], inplace=True)

    #  Write ─
    map_table    = pa.Table.from_pandas(map_chunk,    preserve_index=False)

    if _map_writer is None:
        _map_writer = pq.ParquetWriter(MAP_PATH, map_table.schema, compression="zstd")

    _map_writer.write_table(map_table)

    del chunk_df, X, dmat, contribs, shap_vals, raw_score
    del map_chunk, map_table
    gc.collect()

#  Finalise ─
if _map_writer:    _map_writer.close()


log.info("Done.")
log.info("  Map file    : %s", MAP_PATH)
