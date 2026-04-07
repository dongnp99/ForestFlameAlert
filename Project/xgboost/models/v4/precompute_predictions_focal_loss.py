"""
precompute_predictions.py  (v2)
================================
Run XGBoost inference + SHAP attribution on the 2023–2024 test period.

Produces TWO parquet files optimised for different access patterns:

  app_predictions_map.parquet
      Purpose : feed the Leaflet heatmap (one row per grid per day)
      Columns : date, grid_id, lat, lon, fire_prob, fire, dominant_factor
      Size    : ~7 columns × 13.7 M rows  → compact, fast to index by date

  app_predictions_detail.parquet
      Purpose : populate sidebar tabs when a grid cell is clicked
      Columns : date, grid_id, fire_prob, fire
                + 5 weather display fields
                + 5 human-signal display fields
                + 5 SHAP group % contributions
                + dominant_factor
      Size    : ~20 columns × 13.7 M rows → indexed by grid_id in backend
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent / "Project" / "xgboost"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from xgb_config import compute_dominant_factor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
MODEL_PATH  = BASE / "Project/xgboost/models/v4/xgb_focal_tuned_v1.json"
DATA_PATH   = BASE / "Project/data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet"
COORDS_PATH = BASE / "Project/data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"
MAP_PATH    = BASE / "app_predictions_map_upgrade.parquet"
DETAIL_PATH = BASE / "app_predictions_detail_upgrade.parquet"

DATE_CHUNK = 30   # larger chunks → fewer DMatrix constructions → faster overall

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid for log-odds → probability."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

# ── Feature columns (must match model training order) ─────────────────────────
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

# ── SHAP feature groups (2-group: human vs natural) ───────────────────────────
# Every feature in FEATURE_COLS belongs to exactly one group.
# Human: features that reflect direct human activity and land management.
# Natural: weather, fire spread dynamics, vegetation state, terrain.
SHAP_GROUPS: dict[str, list[str]] = {
    "shap_human": [
        "dist_road_km", "dist_settlement_km", "dist_powerline_km",
        "lulc_class", "cropland_frac_1km", "tree_cover_pct", "deforestation_lag_1y",
        "nightlight_mean", "pop_density",
        "fire_freq_5y", "burn_season_flag", "days_since_harvest",
    ],
    "shap_natural": [
        # Weather
        "tmean", "rh", "wind", "rain", "vpd",
        "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
        "vpd_neighbor_1d", "sin_doy", "cos_doy",
        # Fire history / spread
        "fire_lag_1", "fire_lag_3",
        "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
        "vpd_fire_lag_1", "days_since_last_fire",
        "fire_count_prev_year", "fire_count_prev_3y",
        # Vegetation
        "ndvi", "ndvi_std", "ndwi", "nbr", "ndii",
        "delta_ndvi_14d", "delta_nbr_7d", "has_s2", "dist_forest_edge_km",
        # Terrain
        "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    ],
}
SHAP_COLS = list(SHAP_GROUPS.keys())   # ["shap_human", "shap_natural"]

# ── Columns saved to each output file ─────────────────────────────────────────
# Map file: lean — only what the heatmap renderer needs
MAP_COLS = ["date", "grid_id", "fire_prob", "fire", "dominant_factor"]
# (lat, lon joined from coords after concat)

# Detail file: display values for sidebar tabs
DISPLAY_WEATHER = ["tmean", "rh", "vpd", "rain_14d_sum", "vpd_14d_mean"]
DISPLAY_HUMAN   = ["burn_season_flag", "fire_freq_5y", "dist_road_km",
                   "lulc_class", "dist_settlement_km"]
DETAIL_COLS = (
    ["date", "grid_id", "fire_prob", "fire"]
    + DISPLAY_WEATHER
    + DISPLAY_HUMAN
    + SHAP_COLS
    + ["dominant_factor"]
)

# ── Pre-build group → column-index mapping for fast SHAP aggregation ──────────
_feat_index = {f: i for i, f in enumerate(FEATURE_COLS)}
GROUP_INDICES: dict[str, list[int]] = {
    g: [_feat_index[f] for f in feats if f in _feat_index]
    for g, feats in SHAP_GROUPS.items()
}

# ── Load model ─────────────────────────────────────────────────────────────────
log.info("Loading XGBoost model from %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
model.set_param("nthread", str(-1))   # use all physical cores
log.info("Model loaded.")

# ── Load grid coordinates ──────────────────────────────────────────────────────
coords = pd.read_csv(COORDS_PATH)
log.info("Loaded %d grid coordinate rows.", len(coords))

# ── Collect unique dates in test period ───────────────────────────────────────
log.info("Scanning unique dates for 2023–2024 …")
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
log.info("Total dates to process: %d", len(all_dates))

# ── PyArrow schemas for incremental writers ────────────────────────────────────
# Schemas are inferred from the first chunk; writers are opened lazily.
_map_writer:    Optional[pq.ParquetWriter] = None
_detail_writer: Optional[pq.ParquetWriter] = None

# ── Chunked inference + SHAP ───────────────────────────────────────────────────
n_chunks = int(np.ceil(len(all_dates) / DATE_CHUNK))

for i in range(n_chunks):
    chunk_dates = all_dates[i * DATE_CHUNK : (i + 1) * DATE_CHUNK]
    log.info("Chunk %d/%d  (%s → %s) …",
             i + 1, n_chunks,
             pd.Timestamp(chunk_dates[0]).date(),
             pd.Timestamp(chunk_dates[-1]).date())

    # ── Load feature data ──────────────────────────────────────────────────
    read_cols = ["date", "grid_id", "fire"] + FEATURE_COLS
    chunk_df = pd.read_parquet(
        DATA_PATH,
        columns=read_cols,
        filters=[
            ("date", ">=", pd.Timestamp(chunk_dates[0])),
            ("date", "<=", pd.Timestamp(chunk_dates[-1])),
        ],
        engine="pyarrow",
    )
    log.info("Done read parquet...")
    # Batch-cast all feature columns to float32 in one operation
    chunk_df[FEATURE_COLS] = chunk_df[FEATURE_COLS].astype("float32")

    X = chunk_df[FEATURE_COLS].values   # numpy (n, 47), float32

    log.info("Done casting type...")
    # ── Single XGBoost pass: contribs covers both SHAP and fire_prob ───────
    # pred_contribs=True returns shape (n_rows, n_features + 1)
    # Last column is the bias/intercept (log-odds). Sum all columns → raw score.
    # Apply sigmoid to convert log-odds score to probability.
    dmat     = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    contribs = model.predict(dmat, pred_contribs=True,
                             approx_contribs=True)             # path-based approx: ~5–10× faster

    raw_score = contribs.sum(axis=1)                          # (n,) log-odds
    chunk_df["fire_prob"] = _sigmoid(raw_score).astype("float32")
    log.info("Done predicting...")

    shap_vals = contribs[:, :-1]                              # (n, 47) — drop bias
    np.abs(shap_vals, out=shap_vals)                          # in-place absolute values

    log.info("Done shapping...")
    # Aggregate into group sums, then normalise to 100 %
    for group_name, indices in GROUP_INDICES.items():
        chunk_df[group_name] = shap_vals[:, indices].sum(axis=1).astype("float32")

    group_total = chunk_df[SHAP_COLS].sum(axis=1).clip(lower=1e-8)
    for col in SHAP_COLS:
        chunk_df[col] = (chunk_df[col] / group_total).round(1).astype("float32")

    # Dominant factor: rule-based (same pathways as sample weights) — not from SHAP
    chunk_df["dominant_factor"] = compute_dominant_factor(chunk_df)

    log.info("Done computing dominant_factor...")
    # ── Build map rows (merge coords inline) ───────────────────────────────
    map_chunk = chunk_df[MAP_COLS].merge(coords, on="grid_id", how="left")
    map_chunk["date"]      = pd.to_datetime(map_chunk["date"])
    map_chunk["fire_prob"] = map_chunk["fire_prob"].astype("float32")
    map_chunk["fire"]      = map_chunk["fire"].astype("int8")
    map_chunk.sort_values(["date", "grid_id"], inplace=True)

    log.info("Done building map...")
    # ── Build detail rows ──────────────────────────────────────────────────
    detail_chunk = chunk_df[DETAIL_COLS].copy()
    detail_chunk["date"]      = pd.to_datetime(detail_chunk["date"])
    detail_chunk["fire_prob"] = detail_chunk["fire_prob"].astype("float32")
    detail_chunk["fire"]      = detail_chunk["fire"].astype("int8")
    for col in DISPLAY_WEATHER:
        detail_chunk[col] = detail_chunk[col].astype("float32")
    detail_chunk["burn_season_flag"]   = detail_chunk["burn_season_flag"].astype("int8")
    detail_chunk["fire_freq_5y"]       = detail_chunk["fire_freq_5y"].astype("float32")
    detail_chunk["dist_road_km"]       = detail_chunk["dist_road_km"].astype("float32")
    detail_chunk["dist_settlement_km"] = detail_chunk["dist_settlement_km"].astype("float32")
    detail_chunk["lulc_class"]         = detail_chunk["lulc_class"].astype("int8")
    detail_chunk.sort_values(["grid_id", "date"], inplace=True)

    log.info("Done building detail...")
    # ── Write chunks directly to parquet (no list accumulation) ───────────
    map_table    = pa.Table.from_pandas(map_chunk,    preserve_index=False)
    detail_table = pa.Table.from_pandas(detail_chunk, preserve_index=False)

    if _map_writer is None:
        _map_writer = pq.ParquetWriter(MAP_PATH, map_table.schema,
                                       compression="zstd")
    if _detail_writer is None:
        _detail_writer = pq.ParquetWriter(DETAIL_PATH, detail_table.schema,
                                          compression="zstd")

    _map_writer.write_table(map_table)
    _detail_writer.write_table(detail_table)

    del chunk_df, X, dmat, contribs, shap_vals, raw_score
    del map_chunk, detail_chunk, map_table, detail_table
    gc.collect()

# ── Finalise parquet files ─────────────────────────────────────────────────────
if _map_writer:
    _map_writer.close()
    log.info("Map file written (raw) → %s", MAP_PATH)
if _detail_writer:
    _detail_writer.close()
    log.info("Detail file written (raw) → %s", DETAIL_PATH)

# ── Normalise fire_prob → percentage [0, 100] (min-max across full dataset) ────
# Both map and detail files share the same fire_prob values so we compute
# global min/max once from the map file (smaller read) and apply to both.
log.info("Computing global min/max of fire_prob for normalisation …")
prob_stats = pd.read_parquet(MAP_PATH, columns=["fire_prob"])["fire_prob"]
prob_min   = float(prob_stats.min())
prob_max   = float(prob_stats.max())
del prob_stats
gc.collect()
log.info("fire_prob  min=%.6f  max=%.6f", prob_min, prob_max)

def _normalise_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale fire_prob to [0, 100] percentage using global range."""
    df["fire_prob"] = (
        (df["fire_prob"] - prob_min) / (prob_max - prob_min)
    ).round(2).astype("float32")
    return df

for src_path, label in [(MAP_PATH, "map"), (DETAIL_PATH, "detail")]:
    log.info("Normalising %s file …", label)
    tmp_path = src_path.with_suffix(".tmp.parquet")
    src_pf   = pq.ParquetFile(src_path)
    writer   = None
    for batch in src_pf.iter_batches(batch_size=500_000):
        chunk = _normalise_prob(batch.to_pandas())
        tbl   = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        del chunk, tbl
        gc.collect()
    if writer:
        writer.close()
    del src_pf            # release file handle before rename (required on Windows)
    gc.collect()
    src_path.unlink()     # delete original (Windows requires this before replace)
    tmp_path.rename(src_path)
    log.info("  → %s normalised and written.", src_path)

log.info("Done.")
log.info("  Map file    : %s", MAP_PATH)
log.info("  Detail file : %s", DETAIL_PATH)
