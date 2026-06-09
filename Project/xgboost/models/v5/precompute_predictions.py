"""
precompute_predictions_v5.py — Option C inference
==================================================
Run XGBoost inference + SHAP + fire type classification.

Produces THREE files:

  app_predictions_map.parquet
      Heatmap data: date, grid_id, lat, lon, fire_prob, fire, dominant_factor
      + fire_type, fire_subtype, confidence_score

  app_predictions_detail.parquet
      Sidebar data: weather + human display fields + SHAP groups
      + fire_type, fire_subtype, confidence_score, matched_pathways

  app_fire_type_summary.csv
      Aggregate: daily counts by fire_type and fire_subtype (for dashboard)

Yêu cầu: dataset v3_pathways + model v5.
"""

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

BASE        = Path(__file__).parent
MODEL_PATH  = BASE / "xgb_human_features_tuned_v5_pathways.json"
DATA_PATH   = BASE / "../../..//data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"
COORDS_PATH = BASE / "../../../data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"
MAP_PATH    = BASE / "app_predictions_map.parquet"
DETAIL_PATH = BASE / "app_predictions_detail.parquet"
SUMMARY_PATH = BASE / "app_fire_type_summary.csv"

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

# Display columns for detail file
DISPLAY_WEATHER = ["tmean", "rh", "vpd", "rain_14d_sum", "vpd_14d_mean"]
DISPLAY_HUMAN   = ["burn_season_flag", "fire_freq_5y", "dist_road_km",
                   "lulc_class", "dist_settlement_km"]

#  SHAP group index mapping 
_feat_index = {f: i for i, f in enumerate(FEATURE_COLS)}
GROUP_INDICES = {
    g: [_feat_index[f] for f in feats if f in _feat_index]
    for g, feats in SHAP_GROUPS.items()
}

# Số feature mỗi nhóm — dùng để chuẩn hóa khi so sánh
N_FEAT = {g: len(indices) for g, indices in GROUP_INDICES.items()}
# shap_human: 17 features, shap_natural: 35 features

#  Load model ─
log.info("Loading model: %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
model.set_param("nthread", str(-1))

#  Load coordinates 
coords = pd.read_csv(COORDS_PATH)
log.info("Loaded %d grid coordinates", len(coords))

#  Scan dates ─
log.info("Scanning dates for 2023–2024...")
date_df = pd.read_parquet(
    DATA_PATH, columns=["date"],
    filters=[
        ("date", ">=", pd.Timestamp("2023-01-01")),
        ("date", "<=", pd.Timestamp("2024-12-31")),
    ],
    engine="pyarrow",
)
all_dates = np.sort(date_df["date"].unique())
del date_df; gc.collect()
log.info("Total dates: %d", len(all_dates))

#  Output columns ─
MAP_COLS = ["date", "grid_id", "fire_prob", "fire", "dominant_factor",
            "fire_type", "fire_subtype", "confidence_score"]

DETAIL_COLS = (
    ["date", "grid_id", "fire_prob", "fire"]
    + DISPLAY_WEATHER + DISPLAY_HUMAN
    + SHAP_COLS
    + ["dominant_factor", "fire_type", "fire_subtype",
       "confidence_score", "matched_pathways"]
)

#  Writers 
_map_writer:    Optional[pq.ParquetWriter] = None
_detail_writer: Optional[pq.ParquetWriter] = None
_summary_rows = []

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

    # Giữ nguyên dấu để phân biệt đóng góp tăng/giảm xác suất cháy
    shap_signed = contribs[:, :-1]   # shape (n_rows, n_features), có dấu +/-

    #  Chỉ lấy SHAP dương (features tích cực đẩy prob cháy lên) 
    # SHAP âm = feature đang giảm rủi ro → không phải nguyên nhân gây cháy
    shap_pos = np.maximum(shap_signed, 0).astype("float32")

    # Cộng theo nhóm rồi chia số feature → so sánh công bằng (per-feature avg)
    h_idx = GROUP_INDICES["shap_human"]
    n_idx = GROUP_INDICES["shap_natural"]
    pos_h_sum = shap_pos[:, h_idx].sum(axis=1)
    pos_n_sum = shap_pos[:, n_idx].sum(axis=1)

    shap_h_avg = (pos_h_sum / N_FEAT["shap_human"]).astype("float32")   # avg SHAP+ / feature
    shap_n_avg = (pos_n_sum / N_FEAT["shap_natural"]).astype("float32")
    avg_total  = (shap_h_avg + shap_n_avg).clip(min=1e-8)

    # Lưu dưới dạng normalized % (shap_human + shap_natural = 100%)
    # → nếu shap_human > 50% thì fire_type = "Con người" (luôn nhất quán)
    chunk_df["shap_human"]   = (shap_h_avg / avg_total * 100).round(1).astype("float32")
    chunk_df["shap_natural"] = (shap_n_avg / avg_total * 100).round(1).astype("float32")

    #  Dominant factor (rule-based) 
    chunk_df["dominant_factor"] = compute_dominant_factor(chunk_df)

    #  Fire type classification ─
    fire_type_shap = np.where(shap_h_avg >= shap_n_avg, "Con người", "Tự nhiên")

    fire_type_df = classify_fire_type(chunk_df)

    subtype = fire_type_df["fire_subtype"].to_numpy(dtype=object).copy()
    switched = (fire_type_shap == "Tự nhiên") & (fire_type_df["fire_type"] == "Con người").values
    subtype[switched] = "Thời tiết / Tự nhiên"

    # Confidence = normalized % của nhóm chiếm ưu thế (luôn > 50% nếu phân loại rõ ràng)
    confidence = np.where(
        fire_type_shap == "Con người",
        shap_h_avg / avg_total,
        shap_n_avg / avg_total,
    ).astype("float32")

    chunk_df["fire_type"]        = pd.Categorical(fire_type_shap)
    chunk_df["fire_subtype"]     = pd.Categorical(subtype)
    chunk_df["confidence_score"] = confidence
    chunk_df["matched_pathways"] = fire_type_df["matched_pathways"]

    #  Daily summary for dashboard ─
    daily_summary = (
        chunk_df[chunk_df["fire"] == 1]
        .groupby(["date", "fire_type", "fire_subtype"])
        .size()
        .reset_index(name="count")
    )
    _summary_rows.append(daily_summary)

    #  Build map rows 
    map_chunk = chunk_df[MAP_COLS].merge(coords, on="grid_id", how="left")
    map_chunk["date"]      = pd.to_datetime(map_chunk["date"])
    map_chunk["fire_prob"] = map_chunk["fire_prob"].astype("float32")
    map_chunk["fire"]      = map_chunk["fire"].astype("int8")
    map_chunk.sort_values(["date", "grid_id"], inplace=True)

    #  Build detail rows ─
    detail_chunk = chunk_df[DETAIL_COLS].copy()
    detail_chunk["date"]      = pd.to_datetime(detail_chunk["date"])
    detail_chunk["fire_prob"] = detail_chunk["fire_prob"].astype("float32")
    detail_chunk["fire"]      = detail_chunk["fire"].astype("int8")
    detail_chunk.sort_values(["grid_id", "date"], inplace=True)

    #  Write ─
    map_table    = pa.Table.from_pandas(map_chunk,    preserve_index=False)
    detail_table = pa.Table.from_pandas(detail_chunk, preserve_index=False)

    if _map_writer is None:
        _map_writer = pq.ParquetWriter(MAP_PATH, map_table.schema, compression="zstd")
    if _detail_writer is None:
        _detail_writer = pq.ParquetWriter(DETAIL_PATH, detail_table.schema, compression="zstd")

    _map_writer.write_table(map_table)
    _detail_writer.write_table(detail_table)

    del chunk_df, X, dmat, contribs, shap_signed, shap_pos, raw_score
    del fire_type_df, map_chunk, detail_chunk, map_table, detail_table
    gc.collect()

#  Finalise ─
if _map_writer:    _map_writer.close()
if _detail_writer: _detail_writer.close()

# Save summary
if _summary_rows:
    summary_df = pd.concat(_summary_rows, ignore_index=True)
    summary_df = summary_df.groupby(["date", "fire_type", "fire_subtype"])["count"].sum().reset_index()
    summary_df.to_csv(SUMMARY_PATH, index=False)
    log.info("Summary written → %s", SUMMARY_PATH)

log.info("Done.")
log.info("  Map file    : %s", MAP_PATH)
log.info("  Detail file : %s", DETAIL_PATH)
log.info("  Summary     : %s", SUMMARY_PATH)