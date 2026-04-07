"""
xgb_retrain_v5.py — Option C: Pathway-aware features
=====================================================
Train XGBoost with:
  - V4 graduated weighting (unchanged)
  - 5 new pathway-aware features (is_pathway_p1/p2/p3, score, signal_strength)
  - Extended evaluation: overall + per-group (human vs natural) metrics

Yêu cầu: đã chạy add_pathway_features.py để tạo dataset v3_pathways.
"""

import os
import gc
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# =============================
# LOAD DATA
# =============================

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v3_pathways.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

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
    "is_pathway_p1",
    "is_pathway_p2",
    "is_pathway_p3",
    "human_pathway_score",
    "human_signal_strength",
]

def load_split(filters):
    """Load feature cols + fire + pathway deps for weight computation."""
    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire"],
        filters=filters,
        engine="pyarrow",
    )
    df["fire"] = df["fire"].astype("int8")
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df

# ── Helpers ──────────────────────────────────────────────────────────────────────

def compute_pathways(df):
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

# ── Train ──────────────────────────────────────────────────────────────────────
logging.info("Loading train set...")
train_df = load_split([("date", "<=", pd.Timestamp(TRAIN_END_DATE))])
logging.info("Train size: %s  |  fire rate: %.6f", f"{len(train_df):,}", train_df["fire"].mean())

# Log pathway feature stats
for col in ["is_pathway_p1", "is_pathway_p2", "is_pathway_p3",
            "human_pathway_score", "human_signal_strength"]:
    if col in train_df.columns:
        fire_mean = train_df.loc[train_df["fire"] == 1, col].mean()
        all_mean  = train_df[col].mean()
        logging.info("  %-25s  all_mean=%.4f  fire_mean=%.4f", col, all_mean, fire_mean)

# V4 weighting
pw = compute_pathways(train_df)
human_count = int((pw["p1"] | pw["p2"] | pw["p3"]).sum() & (train_df["fire"] == 1).values.sum())
score = pw["p1"].astype(int) + pw["p2"].astype(int) + pw["p3"].astype(int)
fire_mask = (train_df["fire"] == 1).values
for s in range(4):
    cnt = ((score == s) & fire_mask).sum()
    logging.info("  Pathway score=%d: %d fire samples (%.2f%%)", s, cnt, 100 * cnt / fire_mask.sum())

logging.info("Creating dtrain QuantileDMatrix...")
w_train = compute_sample_weights(train_df)
y_train = train_df.pop("fire").values
dtrain  = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del train_df, y_train, w_train
gc.collect()

# ── Val ────────────────────────────────────────────────────────────────────────
logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
logging.info("Val size: %s", f"{len(val_df):,}")

# Save val pathway scores for per-group evaluation later
val_pathway_score = (
    val_df["is_pathway_p1"].values.astype(int)
    + val_df["is_pathway_p2"].values.astype(int)
    + val_df["is_pathway_p3"].values.astype(int)
) if "is_pathway_p1" in val_df.columns else np.zeros(len(val_df), dtype=int)

y_val = val_df.pop("fire").values
dval  = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df
gc.collect()

# ── Test ───────────────────────────────────────────────────────────────────────
logging.info("Loading test set...")
test_df = load_split([("date", ">", pd.Timestamp(VAL_END_DATE))])
logging.info("Test size: %s", f"{len(test_df):,}")

test_pathway_score = (
    test_df["is_pathway_p1"].values.astype(int)
    + test_df["is_pathway_p2"].values.astype(int)
    + test_df["is_pathway_p3"].values.astype(int)
) if "is_pathway_p1" in test_df.columns else np.zeros(len(test_df), dtype=int)

y_test = test_df.pop("fire").values
dtest  = xgb.QuantileDMatrix(test_df, y_test, ref=dtrain)
del test_df
gc.collect()

# =============================
# PARAMS (from Optuna V4 — retune if needed)
# =============================

best_params = {
    "max_depth":        7,
    "min_child_weight": 9,
    "learning_rate":    0.07663559850345975,
    "subsample":        0.8556253267078845,
    "colsample_bytree": 0.5009065181704361,
    "gamma":            1.481068503057549,
    "reg_lambda":       3.657104262783736,
    "reg_alpha":        7.980994519859137,
    "scale_pos_weight": 662.4909167188719,
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "device":           "cuda",
    "random_state":     RANDOM_STATE,
    "max_bin":          256,
    "grow_policy":      "lossguide",
}

# =============================
# TRAIN
# =============================

logging.info("Training V5 model (Option C: pathway-aware features)...")

model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=100,
)

# =============================
# EVALUATE — Overall
# =============================

logging.info("Evaluating...")

y_val_pred  = model.predict(dval)
y_test_pred = model.predict(dtest)

val_pr   = average_precision_score(y_val,  y_val_pred)
test_pr  = average_precision_score(y_test, y_test_pred)
val_roc  = roc_auc_score(y_val,  y_val_pred)
test_roc = roc_auc_score(y_test, y_test_pred)

logging.info("=" * 55)
logging.info("OVERALL METRICS")
logging.info("=" * 55)
logging.info("Best iteration    : %d", model.best_iteration)
logging.info("Validation AUC-PR : %.6f", val_pr)
logging.info("Test AUC-PR       : %.6f", test_pr)
logging.info("Validation ROC-AUC: %.6f", val_roc)
logging.info("Test ROC-AUC      : %.6f", test_roc)

# =============================
# EVALUATE — Per-group (Human vs Natural)
# =============================

logging.info("=" * 55)
logging.info("PER-GROUP METRICS (Human fire vs Natural fire)")
logging.info("=" * 55)

def eval_group(y_true, y_pred, pathway_scores, group_name, min_positive=10):
    """Evaluate metrics for human (score>0) and natural (score==0) fire subgroups."""
    human_mask   = pathway_scores > 0
    natural_mask = pathway_scores == 0

    for label, mask in [("Human (score>0)", human_mask), ("Natural (score=0)", natural_mask)]:
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n_pos = y_t.sum()
        n_total = len(y_t)

        if n_pos < min_positive:
            logging.info("  %s %s: skipped (only %d positive samples)", group_name, label, n_pos)
            continue

        pr  = average_precision_score(y_t, y_p)
        roc = roc_auc_score(y_t, y_p)
        logging.info("  %s %-20s  n=%8d  fire=%5d  AUC-PR=%.6f  ROC-AUC=%.6f",
                     group_name, label, n_total, n_pos, pr, roc)

eval_group(y_val,  y_val_pred,  val_pathway_score,  "Val ")
eval_group(y_test, y_test_pred, test_pathway_score,  "Test")

# =============================
# FEATURE IMPORTANCE — Top 15
# =============================

logging.info("=" * 55)
logging.info("TOP 15 FEATURE IMPORTANCE (gain)")
logging.info("=" * 55)

importance = model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
for rank, (feat, gain) in enumerate(sorted_imp, 1):
    # Flag pathway features
    marker = " ← PATHWAY" if feat.startswith(("is_pathway", "human_pathway", "human_signal")) else ""
    logging.info("  %2d. %-30s  gain=%.1f%s", rank, feat, gain, marker)

# =============================
# SAVE
# =============================

out_path = os.path.join(OUTPUT_DIR, "xgb_human_features_tuned_v5_pathways.json")
model.save_model(out_path)
logging.info("Model saved to %s", out_path)