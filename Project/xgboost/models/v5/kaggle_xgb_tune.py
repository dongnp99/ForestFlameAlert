"""
xgb_tune_v5.py — Optuna tuning for V5 (Option C: pathway-aware features)
=========================================================================
Tune hyperparameters cho dataset v3_pathways (52 features).
Search space đã điều chỉnh cho:
  - Feature space lớn hơn (52 vs 47)
  - Có correlated features (pathway score = sum của p1+p2+p3)
  - Graduated weighting V4 đã thay đổi effective class balance

Chạy trên Kaggle GPU.
"""

import gc
import os
import logging

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v3_pathways.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

N_TRIALS          = 40
TRAIN_SAMPLE_SIZE = 15_000_000   # most-recent N rows by date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

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

TRAIN_END = pd.to_datetime(TRAIN_END_DATE)
VAL_END   = pd.to_datetime(VAL_END_DATE)

# ============================================================
# HELPERS
# ============================================================

def load_split(date_filters):
    return pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire", "date"],
        filters=date_filters,
        engine="pyarrow",
    )


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

# ============================================================
# TRAIN — load → sample → DMatrix → free
# ============================================================

logging.info("Loading train set...")
train_df = load_split([("date", "<=", TRAIN_END)])
logging.info("Full train size: %s", f"{len(train_df):,}")

# Temporal sample: keep the most-recent TRAIN_SAMPLE_SIZE rows
if len(train_df) > TRAIN_SAMPLE_SIZE:
    keep_idx = np.argpartition(train_df["date"].values, -TRAIN_SAMPLE_SIZE)[-TRAIN_SAMPLE_SIZE:]
    keep_idx.sort()
    train_df = train_df.iloc[keep_idx].copy()
    del keep_idx
    gc.collect()
    logging.info("After sampling: %s", f"{len(train_df):,}")

del train_df["date"]

neg = (train_df["fire"] == 0).sum()
pos = (train_df["fire"] == 1).sum()
base_spw = neg / pos
logging.info("Train fire rate     : %.6f", pos / (neg + pos))
logging.info("Base scale_pos_weight: %.2f", base_spw)

# Pathway breakdown
pw = compute_pathways(train_df)
score = pw["p1"].astype(int) + pw["p2"].astype(int) + pw["p3"].astype(int)
fire_mask = (train_df["fire"] == 1).values
for s in range(4):
    cnt = ((score == s) & fire_mask).sum()
    logging.info("  Pathway score=%d: %d fire (%.2f%%)", s, cnt, 100 * cnt / fire_mask.sum())

logging.info("Creating dtrain QuantileDMatrix...")
w_train = compute_sample_weights(train_df)
y_train = train_df.pop("fire").values
dtrain  = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del train_df, y_train, w_train
gc.collect()

# ============================================================
# VAL
# ============================================================

logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  TRAIN_END),
    ("date", "<=", VAL_END),
])
logging.info("Val size: %s", f"{len(val_df):,}")

del val_df["date"]
y_val = val_df.pop("fire").values
logging.info("Creating dval QuantileDMatrix...")
dval = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df, y_val
gc.collect()

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):
    params = {
        #  Tree structure 
        "max_depth":         trial.suggest_int("max_depth", 6, 11),
        "min_child_weight":  trial.suggest_int("min_child_weight", 2, 15),

        #  Learning 
        "learning_rate":     trial.suggest_float("learning_rate", 0.015, 0.08),

        #  Sampling 
        "subsample":         trial.suggest_float("subsample", 0.6, 0.95),
        # colsample_bytree: mở rộng xuống 0.5 vì nhiều correlated features
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 0.85),

        #  Regularization 
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        # reg_alpha/lambda: tăng lower bound vì pathway features correlated
        "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 12.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.5, 8.0),

        #  Class balance 
        # Mở rộng range vì graduated weighting đã thay đổi effective balance
        "scale_pos_weight":  trial.suggest_float(
            "scale_pos_weight", base_spw * 0.6, base_spw * 1.3
        ),

        #  Fixed 
        "objective":   "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device":      "cuda",
        "max_bin":     256,
        "grow_policy": "lossguide",
        "random_state": RANDOM_STATE,
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-aucpr"
    )

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        early_stopping_rounds=120,
        callbacks=[pruning_callback],
        verbose_eval=False,
    )
    return booster.best_score

# ============================================================
# RUN STUDY
# ============================================================

logging.info("Starting Optuna study  (n_trials=%d)...", N_TRIALS)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=150),
)
study.optimize(objective, n_trials=N_TRIALS)

# ============================================================
# SAVE RESULTS
# ============================================================

trials_path = os.path.join(OUTPUT_DIR, "xgb_v5_optuna_trials.csv")
study.trials_dataframe().to_csv(trials_path, index=False)

logging.info("=" * 55)
logging.info("BEST AUC-PR : %.6f", study.best_value)
logging.info("BEST PARAMS :")
for k, v in study.best_params.items():
    logging.info("  %-20s = %s", k, v)
logging.info("=" * 55)
logging.info("Trials saved to: %s", trials_path)

# Print params in copy-paste format for retrain script
print("\n" + "=" * 55)
print("COPY-PASTE INTO xgb_retrain_v5.py best_params:")
print("=" * 55)
print("best_params = {")
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f'    "{k}": {v},')
    else:
        print(f'    "{k}": {v},')
print('    "objective":    "binary:logistic",')
print('    "eval_metric":  "aucpr",')
print('    "tree_method":  "hist",')
print('    "device":       "cuda",')
print('    "max_bin":      256,')
print('    "grow_policy":  "lossguide",')
print(f'    "random_state": {RANDOM_STATE},')
print("}")