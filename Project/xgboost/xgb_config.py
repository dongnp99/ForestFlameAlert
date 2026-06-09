"""
xgb_config.py — V5 (Option C: Pathway-aware features)
======================================================
Changes from V4:
  - DATA_PATH → v3_pathways dataset (has 5 new pathway features)
  - FEATURE_COLS → thêm 5 pathway-aware features
  - compute_sample_weights() → V4 graduated weighting (unchanged)
  - compute_dominant_factor() → V4 pathway logic (unchanged)
  - classify_fire_type() → NEW: detailed post-hoc fire type classification
  - SHAP_GROUPS updated → pathway features thuộc nhóm "shap_human"
"""

import numpy as np

# =============================
# DATA PATH
# =============================

DATA_PATH = "D:/Master/ForestFlameAlert/Project/data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"

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
# HUMAN FIRE SAMPLE WEIGHTING (V4 — unchanged)
# =============================

HUMAN_FIRE_MULTIPLIER = 1.5   # minimum non-trivial weight (1-pathway match)


def compute_pathways(df):
    """Compute V4 pathway boolean masks. Reused by weights, dominant_factor, classify.

    Returns dict of {name: boolean_array}.
    """
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

    return {"p1": p1, "p2": p2, "p3": p3}


def compute_sample_weights(df):
    """Return a float32 weight array for training rows.

    V4 pathway logic — graduated weights:
      0 pathways → 1.0x
      1 pathway  → 1.5x
      2 pathways → 2.5x
      3 pathways → 3.5x

    *df* must still contain the "fire" column.
    """
    fire = (df["fire"] == 1).values
    pw = compute_pathways(df)

    score = (
        pw["p1"].astype(np.int8)
        + pw["p2"].astype(np.int8)
        + pw["p3"].astype(np.int8)
    )

    weights_map = np.array([1.0, 1.5, 2.5, 3.5], dtype="float32")
    weights = np.ones(len(df), dtype="float32")
    weights[fire] = weights_map[score[fire]]
    return weights


def compute_dominant_factor(df) -> "pd.Series":
    """Return 'Con người' or 'Tự nhiên' per row.

    A row is 'Con người' if it matches ANY V4 pathway.
    """
    pw = compute_pathways(df)
    human = pw["p1"] | pw["p2"] | pw["p3"]
    import pandas as pd
    return pd.Series(
        np.where(human, "Con người", "Tự nhiên"),
        index=df.index,
        dtype="category",
    )


def classify_fire_type(df) -> "pd.DataFrame":
    """Post-hoc fire type classification — Tầng 2 của Option C.

    Gán nhãn chi tiết cho mỗi row dựa trên V4 pathways.

    Returns DataFrame with columns:
      - fire_type        : "Con người" | "Tự nhiên"
      - fire_subtype     : chi tiết hơn (e.g., "Đốt nương rẫy", "Phá rừng", ...)
      - confidence_score : 0.0–1.0, cao = chắc chắn hơn
      - matched_pathways : e.g., "P1,P3" hoặc "" nếu tự nhiên
    """
    import pandas as pd

    pw = compute_pathways(df)
    score = (
        pw["p1"].astype(np.int8)
        + pw["p2"].astype(np.int8)
        + pw["p3"].astype(np.int8)
    )

    #  Fire type 
    human = pw["p1"] | pw["p2"] | pw["p3"]
    fire_type = np.where(human, "Con người", "Tự nhiên")

    #  Fire subtype (ưu tiên pathway có selectivity cao nhất) ─
    # P3 (7.76x) > P2 (2.41x) > P1 (1.74x)
    subtype = np.full(len(df), "Thời tiết / Tự nhiên", dtype=object)
    # Gán theo thứ tự ưu tiên thấp → cao (high priority ghi đè)
    subtype[pw["p1"]] = "Đốt nương rẫy sau thu hoạch"
    subtype[pw["p2"]] = "Phá rừng / Khai hoang"
    subtype[pw["p3"]] = "Đốt lặp lại theo mùa"

    # Nếu match nhiều pathway, gán subtype = "Đa yếu tố nhân tạo"
    subtype[score >= 2] = "Đa yếu tố nhân tạo"

    #  Confidence score ─
    # Dựa trên pathway score + human_signal_strength (nếu có)
    confidence = np.zeros(len(df), dtype="float32")

    # Base confidence từ pathway score
    conf_map = np.array([0.2, 0.5, 0.75, 0.95], dtype="float32")
    confidence = conf_map[np.clip(score, 0, 3)]

    # Boost bằng human_signal_strength nếu có
    if "human_signal_strength" in df.columns:
        hss = df["human_signal_strength"].values.astype("float32")
        # Blend: 70% pathway score + 30% soft signal
        confidence = np.clip(0.7 * confidence + 0.3 * hss, 0.0, 1.0)

    # Tự nhiên: confidence = 1 - confidence (chắc chắn tự nhiên)
    confidence[~human] = 1.0 - confidence[~human]

    #  Matched pathways string 
    pw_labels = []
    for i in range(len(df)):
        parts = []
        if pw["p1"][i]: parts.append("P1")
        if pw["p2"][i]: parts.append("P2")
        if pw["p3"][i]: parts.append("P3")
        pw_labels.append(",".join(parts))

    return pd.DataFrame({
        "fire_type":        pd.Categorical(fire_type),
        "fire_subtype":     pd.Categorical(subtype),
        "confidence_score": confidence,
        "matched_pathways": pw_labels,
    }, index=df.index)


# =============================
# FEATURE COLUMNS
# =============================

FEATURE_COLS = [
    # v1
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

    #v2
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

    # ===== NEW: Pathway-aware features (Option C) =====
    "is_pathway_p1",            # binary: post-harvest burning context
    "is_pathway_p2",            # binary: deforestation context
    "is_pathway_p3",            # binary: repeat ignition context
    "human_pathway_score",      # 0-3: số pathway matched
    "human_signal_strength",    # 0-1: soft continuous human signal
]


# =============================
# SHAP FEATURE GROUPS
# =============================

SHAP_GROUPS = {
    "shap_human": [
        "dist_road_km", "dist_settlement_km", "dist_powerline_km",
        "lulc_class", "cropland_frac_1km", "tree_cover_pct",
        "deforestation_lag_1y", "nightlight_mean", "pop_density",
        "fire_freq_5y", "burn_season_flag", "days_since_harvest",
        # NEW pathway features → nhóm human
        "is_pathway_p1", "is_pathway_p2", "is_pathway_p3",
        "human_pathway_score", "human_signal_strength",
    ],
    "shap_natural": [
        "tmean", "rh", "wind", "rain", "vpd",
        "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
        "vpd_neighbor_1d", "sin_doy", "cos_doy",
        "fire_lag_1", "fire_lag_3",
        "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d",
        "neighbor_fire_7d", "vpd_fire_lag_1", "days_since_last_fire",
        "fire_count_prev_year", "fire_count_prev_3y",
        "ndvi", "ndvi_std", "ndwi", "nbr", "ndii",
        "delta_ndvi_14d", "delta_nbr_7d", "has_s2",
        "dist_forest_edge_km",
        "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    ],
}