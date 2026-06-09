"""
train_random_forest.py — Optimized Random Forest trên dataset v3_pathways
=========================================================================
Thiết kế cho Kaggle CPU (29GB RAM, 4 cores).

Pattern tránh OOM cho train set (~48M rows):
  - Dùng pyarrow.read_table thay vì pandas cho train
    → Arrow overhead ~200MB vs pandas ~2-3GB
  - Pre-allocate numpy X_train, fill từng cột từ arrow (không tạo DataFrame trung gian)
  - del arrow table ngay sau khi fill xong

Memory budget (full ~48M rows):
  - Arrow table + numpy X_train đồng thời: ~9.2 + 9.0 = ~18.2GB
  - Sau del table: ~9.3GB (X + y + w)
  - RF training (shared fork): ~12–17GB
  - OS + Python + libs: ~5GB
  → Tổng peak: ~23GB — an toàn trong 29GB

Val/Test vẫn dùng pandas (6.8M / 13.7M rows — đủ nhỏ).

Thời gian dự kiến: 2–4 giờ. Nếu timeout, giảm n_estimators=500.
"""

import gc
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ==============================
# SETTINGS
# ==============================

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v3_pathways.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

# Bootstrap size per tree — 5M từ pool ~48M (10% per tree → diversity cao hơn)
# 15M/tree gây OOM sau ~7h do worker memory tích lũy; 5M fix triệt để
MAX_SAMPLES_PER_TREE = 5_000_000

# ==============================
# FEATURES (47 — không bao gồm pathway-derived features)
# ==============================

FEATURE_COLS = [
    # Weather
    "tmean", "rh", "wind", "rain", "vpd",
    # Rolling weather
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    # Fire history
    "fire_lag_1", "fire_lag_3",
    # Terrain
    "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    # Seasonality
    "sin_doy", "cos_doy",
    # Fire adjacency
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
    # Vegetation (Sentinel-2)
    "ndvi", "ndvi_std", "ndwi", "nbr", "ndii",
    "delta_ndvi_14d", "delta_nbr_7d", "has_s2",
    # Human activity
    "dist_road_km", "dist_settlement_km", "dist_forest_edge_km", "dist_powerline_km",
    "lulc_class", "cropland_frac_1km", "tree_cover_pct", "deforestation_lag_1y",
    "nightlight_mean", "pop_density",
    "fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y",
    "days_since_last_fire", "burn_season_flag", "days_since_harvest",
]

# Pre-compute indices cho pathway (dùng với numpy arrays)
_BSF_I = FEATURE_COLS.index("burn_season_flag")
_DSH_I = FEATURE_COLS.index("days_since_harvest")
_DEF_I = FEATURE_COLS.index("deforestation_lag_1y")
_FPY_I = FEATURE_COLS.index("fire_count_prev_year")

# ==============================
# HYPERPARAMETERS
# ==============================

RF_PARAMS = {
    "n_estimators":     500,     # giảm từ 1000: tránh OOM tích lũy sau nhiều giờ
    "max_depth":        None,    # fully grown — RF averaging tự chống overfit
    "min_samples_leaf": 2,
    "max_features":     "sqrt",
    "bootstrap":        True,
    "max_samples":      MAX_SAMPLES_PER_TREE,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
}

# ==============================
# HELPERS — numpy variants (cho train set)
# ==============================

def load_train_numpy(date_filters):
    """Load train set qua pyarrow trực tiếp vào numpy — không tạo pandas DataFrame.

    Peak memory: arrow table (~9.2GB) + numpy X (~9.0GB) = ~18.2GB đồng thời.
    Sau khi hàm trả về: chỉ còn X (~9GB) và y (~48MB).
    """
    logging.info("Reading parquet via pyarrow (bypassing pandas)...")
    table = pq.read_table(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire"],
        filters=date_filters,
    )
    n = len(table)
    logging.info("Arrow table loaded: %s rows", f"{n:,}")

    # Extract y
    y = table.column("fire").to_numpy(zero_copy_only=False).astype(np.int8)
    logging.info("Fire rate: %.6f", y.mean())

    # Pre-allocate X — peak tại đây (arrow ~9.2GB + numpy ~9.0GB)
    X = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        col_arr = table.column(col)
        raw = col_arr.to_numpy(zero_copy_only=False).astype(np.float32)
        if col_arr.null_count > 0:
            null_mask = col_arr.is_null().to_numpy(zero_copy_only=False)
            raw[null_mask] = np.nan   # đánh dấu để fill median sau
        X[:, i] = raw

    del table, raw
    gc.collect()
    return X, y


def compute_pathways_np(X):
    """Pathway masks từ numpy X matrix."""
    p1 = (X[:, _BSF_I] == 1) & (X[:, _DSH_I] < 30)
    p2 = (X[:, _DEF_I] > 1.5) & (X[:, _FPY_I] > 0)
    p3 = (X[:, _FPY_I] > 1) & (X[:, _BSF_I] == 1)
    return {"p1": p1, "p2": p2, "p3": p3}


def compute_sample_weights_np(X, y):
    """V4 pathway weights × class balance, từ numpy arrays."""
    fire  = (y == 1)
    base  = int((~fire).sum()) / int(fire.sum())
    pw    = compute_pathways_np(X)
    score = pw["p1"].astype(np.int8) + pw["p2"].astype(np.int8) + pw["p3"].astype(np.int8)
    mult  = np.array([1.0, 1.5, 2.5, 3.5], dtype="float32")
    w     = np.ones(len(y), dtype="float32")
    w[fire] = base * mult[score[fire]]
    return w


# ==============================
# HELPERS — pandas variants (cho val/test)
# ==============================

def load_split_df(date_filters):
    """Load val/test qua pandas — đủ nhỏ để không gây OOM."""
    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire"],
        filters=date_filters,
        engine="pyarrow",
    )
    df["fire"] = df["fire"].astype("int8")
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


def compute_pathways_df(df):
    """Pathway masks từ pandas DataFrame."""
    p1 = ((df["burn_season_flag"] == 1) & (df["days_since_harvest"] < 30)).values
    p2 = ((df["deforestation_lag_1y"] > 1.5) & (df["fire_count_prev_year"] > 0)).values
    p3 = ((df["fire_count_prev_year"] > 1) & (df["burn_season_flag"] == 1)).values
    return {"p1": p1, "p2": p2, "p3": p3}


def eval_group(y_true, y_pred, pathway_scores, group_name, min_positive=10):
    for label, mask in [
        ("Human (score>0)", pathway_scores > 0),
        ("Natural (score=0)", pathway_scores == 0),
    ]:
        y_t, y_p = y_true[mask], y_pred[mask]
        n_pos = int(y_t.sum())
        if n_pos < min_positive:
            logging.info("  %s %-20s: skipped (%d positive)", group_name, label, n_pos)
            continue
        pr  = average_precision_score(y_t, y_p)
        roc = roc_auc_score(y_t, y_p)
        logging.info(
            "  %s %-20s  n=%8d  fire=%5d  AUC-PR=%.6f  ROC-AUC=%.6f",
            group_name, label, len(y_t), n_pos, pr, roc,
        )

# ==============================
# LOAD TRAIN (pyarrow → numpy)
# ==============================

logging.info("Loading train set (≤ %s)...", TRAIN_END_DATE)
X_train, y_train = load_train_numpy([("date", "<=", pd.Timestamp(TRAIN_END_DATE))])
logging.info("Train size: %s", f"{len(y_train):,}")

# Fill NaN in-place với column medians (trên strided numpy views — không copy)
logging.info("Filling NaN in-place (nanmedian per column)...")
col_medians = {}
for i, col in enumerate(FEATURE_COLS):
    col_view = X_train[:, i]
    nan_mask = np.isnan(col_view)
    if nan_mask.any():
        m = float(np.nanmedian(col_view))
        col_view[nan_mask] = m
        col_medians[col] = m
    else:
        col_medians[col] = float(col_view[0])

# Pathway breakdown (logging)
pw        = compute_pathways_np(X_train)
fire_mask = (y_train == 1)
score_arr = pw["p1"].astype(int) + pw["p2"].astype(int) + pw["p3"].astype(int)
for s in range(4):
    cnt = int(((score_arr == s) & fire_mask).sum())
    logging.info("  Pathway score=%d: %d fire (%.2f%%)", s, cnt, 100 * cnt / max(fire_mask.sum(), 1))
del pw, score_arr, fire_mask
gc.collect()

# Sample weights
w_train = compute_sample_weights_np(X_train, y_train)
logging.info("X_train shape: %s  |  weights ready", X_train.shape)

# ==============================
# TRAIN
# ==============================

logging.info("=" * 60)
logging.info("Training Random Forest")
logging.info("  n_estimators     : %d", RF_PARAMS["n_estimators"])
logging.info("  max_depth        : %s", RF_PARAMS["max_depth"])
logging.info("  min_samples_leaf : %d", RF_PARAMS["min_samples_leaf"])
logging.info("  max_features     : %s", RF_PARAMS["max_features"])
logging.info("  max_samples/tree : %s", f"{MAX_SAMPLES_PER_TREE:,}")
logging.info("  sample_weight    : V4 pathway × class balance")
logging.info("=" * 60)

t0    = time.time()
model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_train, y_train, sample_weight=w_train)
train_time = time.time() - t0
logging.info("Training complete in %.1f s (%.1f min)", train_time, train_time / 60)
del X_train, y_train, w_train
gc.collect()

# ==============================
# LOAD VAL (pandas)
# ==============================

logging.info("Loading val set (%s – %s)...", TRAIN_END_DATE, VAL_END_DATE)
val_df = load_split_df([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
logging.info("Val size: %s  |  fire rate: %.6f", f"{len(val_df):,}", val_df["fire"].mean())

val_pw            = compute_pathways_df(val_df)
val_pathway_score = val_pw["p1"].astype(int) + val_pw["p2"].astype(int) + val_pw["p3"].astype(int)
y_val   = val_df.pop("fire").values
X_val   = val_df[FEATURE_COLS].values
del val_df, val_pw
gc.collect()

for i, col in enumerate(FEATURE_COLS):
    v = X_val[:, i]; m = np.isnan(v)
    if m.any(): v[m] = col_medians[col]

# ==============================
# LOAD TEST (pandas)
# ==============================

logging.info("Loading test set (> %s)...", VAL_END_DATE)
test_df = load_split_df([("date", ">", pd.Timestamp(VAL_END_DATE))])
logging.info("Test size: %s  |  fire rate: %.6f", f"{len(test_df):,}", test_df["fire"].mean())

test_pw            = compute_pathways_df(test_df)
test_pathway_score = test_pw["p1"].astype(int) + test_pw["p2"].astype(int) + test_pw["p3"].astype(int)
y_test  = test_df.pop("fire").values
X_test  = test_df[FEATURE_COLS].values
del test_df, test_pw
gc.collect()

for i, col in enumerate(FEATURE_COLS):
    v = X_test[:, i]; m = np.isnan(v)
    if m.any(): v[m] = col_medians[col]

# ==============================
# EVALUATE — Overall
# ==============================

logging.info("Predicting on val and test...")
y_val_prob  = model.predict_proba(X_val)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

val_aucpr   = average_precision_score(y_val,  y_val_prob)
val_rocauc  = roc_auc_score(y_val,  y_val_prob)
test_aucpr  = average_precision_score(y_test, y_test_prob)
test_rocauc = roc_auc_score(y_test, y_test_prob)

logging.info("=" * 60)
logging.info("OVERALL METRICS")
logging.info("=" * 60)
logging.info("Validation  AUC-PR : %.6f", val_aucpr)
logging.info("Validation  ROC-AUC: %.6f", val_rocauc)
logging.info("Test        AUC-PR : %.6f", test_aucpr)
logging.info("Test        ROC-AUC: %.6f", test_rocauc)

# ==============================
# EVALUATE — Per-group (Human vs Natural)
# ==============================

logging.info("=" * 60)
logging.info("PER-GROUP METRICS (Human vs Natural fire)")
logging.info("=" * 60)
eval_group(y_val,  y_val_prob,  val_pathway_score,  "Val ")
eval_group(y_test, y_test_prob, test_pathway_score,  "Test")

# ==============================
# FEATURE IMPORTANCE — Top 20
# ==============================

logging.info("=" * 60)
logging.info("TOP 20 FEATURE IMPORTANCE (mean decrease in impurity)")
logging.info("=" * 60)
importances = model.feature_importances_
top_idx     = np.argsort(importances)[::-1][:20]
for rank, idx in enumerate(top_idx, 1):
    logging.info("  %2d. %-30s  importance=%.6f", rank, FEATURE_COLS[idx], importances[idx])

# ==============================
# SAVE
# ==============================

model_path   = os.path.join(OUTPUT_DIR, "rf_model_v1.joblib")
metrics_path = os.path.join(OUTPUT_DIR, "rf_metrics_v1.json")

joblib.dump(model, model_path, compress=3)
logging.info("Model saved: %s", model_path)

metrics = {
    "train_time_s":         round(train_time, 1),
    "val_aucpr":            round(float(val_aucpr), 6),
    "val_rocauc":           round(float(val_rocauc), 6),
    "test_aucpr":           round(float(test_aucpr), 6),
    "test_rocauc":          round(float(test_rocauc), 6),
    "n_estimators":         RF_PARAMS["n_estimators"],
    "max_depth":            RF_PARAMS["max_depth"],
    "min_samples_leaf":     RF_PARAMS["min_samples_leaf"],
    "max_samples_per_tree": MAX_SAMPLES_PER_TREE,
    "n_features":           len(FEATURE_COLS),
    "col_medians":          col_medians,
    "feature_importance_top20": {
        FEATURE_COLS[i]: round(float(importances[i]), 8)
        for i in top_idx
    },
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
logging.info("Metrics saved: %s", metrics_path)
logging.info("Done.")