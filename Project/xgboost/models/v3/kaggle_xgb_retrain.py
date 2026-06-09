"""
kaggle_xgb_retrain.py
=====================
Retrain final model with best Optuna params — self-contained for Kaggle Notebooks.

Setup on Kaggle:
  1. Upload your dataset and set DATASET_SLUG below.
  2. Paste best_params from kaggle_xgb_tune.py output into BEST_PARAMS below.
  3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100).
  4. Run as script:  !python /kaggle/working/kaggle_xgb_retrain.py
     OR paste cell-by-cell into a notebook using the # %% markers.

Outputs written to /kaggle/working/:
  - xgb_human_features_tuned_v4_pathways.json
"""

# %% [markdown]
# ## 0. Imports

# %%
import gc
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# %% [markdown]
# ## 1. Paths & constants

# 

DATA_PATH  = f"/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v2_human.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

# Graduated weights — see compute_sample_weights() below.
# Score = number of V4 pathways matched; weight schedule: 0→1.0, 1→1.5, 2→2.5, 3→3.5
HUMAN_FIRE_WEIGHTS = [1.0, 1.5, 2.5, 3.5]

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

#  Paste best_params from Optuna study output here ─

BEST_PARAMS = {
    "max_depth":        9,
    "min_child_weight": 10,
    "learning_rate":    0.01801927682679985,
    "subsample":        0.8126406981655034,
    "colsample_bytree": 0.6596834432905521,
    "gamma":            0.3252579649263976,
    "reg_lambda":       9.514412603906665,
    "reg_alpha":        4.828160165372797,
    "scale_pos_weight": 1175.1539980614325,
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "device":           "cuda",
    "random_state":     RANDOM_STATE,
    "max_bin":          256,
    "grow_policy":      "lossguide",
}
# 

# %% [markdown]
# ## 2. Helpers

# %%
def load_split(date_filters):
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


def compute_sample_weights(df):
    """V4 graduated weighting — score = number of human-fire pathways matched.
    P1: burn_season_flag==1 & days_since_harvest<30
    P2: deforestation_lag_1y>1.5 & fire_count_prev_year>0
    P3: fire_count_prev_year>1 & burn_season_flag==1
    Weight schedule: 0→1.0x, 1→1.5x, 2→2.5x, 3→3.5x
    """
    fire = (df["fire"] == 1).values
    p1 = ((df["burn_season_flag"] == 1) & (df["days_since_harvest"] < 30)).values
    p2 = ((df["deforestation_lag_1y"] > 1.5) & (df["fire_count_prev_year"] > 0)).values
    p3 = ((df["fire_count_prev_year"] > 1) & (df["burn_season_flag"] == 1)).values
    score = p1.astype(np.int8) + p2.astype(np.int8) + p3.astype(np.int8)
    weights_map = np.array(HUMAN_FIRE_WEIGHTS, dtype="float32")
    weights = np.ones(len(df), dtype="float32")
    weights[fire] = weights_map[score[fire]]
    return weights

# %% [markdown]
# ## 3. Load data

# %%
logging.info("Loading train set...")
train_df = load_split([("date", "<=", pd.Timestamp(TRAIN_END_DATE))])
logging.info("Train size: %d  |  fire rate: %.6f", len(train_df), train_df["fire"].mean())

_p1 = (train_df["burn_season_flag"] == 1) & (train_df["days_since_harvest"] < 30)
_p2 = (train_df["deforestation_lag_1y"] > 1.5) & (train_df["fire_count_prev_year"] > 0)
_p3 = (train_df["fire_count_prev_year"] > 1) & (train_df["burn_season_flag"] == 1)
human_count = int(((train_df["fire"] == 1) & (_p1 | _p2 | _p3)).sum())
logging.info("Human-fire samples upweighted: %d  (~44%% expected)", human_count)

logging.info("Creating dtrain QuantileDMatrix...")
w_train = compute_sample_weights(train_df)
y_train = train_df.pop("fire").values
dtrain  = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del train_df, y_train, w_train
gc.collect()

# %%
logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
logging.info("Val size: %d", len(val_df))
y_val = val_df.pop("fire").values
logging.info("Creating dval QuantileDMatrix...")
dval = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df
gc.collect()

# %%
logging.info("Loading test set...")
test_df = load_split([("date", ">", pd.Timestamp(VAL_END_DATE))])
logging.info("Test size: %d", len(test_df))
y_test = test_df.pop("fire").values
logging.info("Creating dtest QuantileDMatrix...")
dtest = xgb.QuantileDMatrix(test_df, y_test, ref=dtrain)
del test_df
gc.collect()

# %% [markdown]
# ## 4. Train

# %%
logging.info("Training final tuned model...")
model = xgb.train(
    BEST_PARAMS,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=100,
)

# %% [markdown]
# ## 5. Evaluate

# %%
logging.info("Evaluating...")
y_val_pred  = model.predict(dval)
y_test_pred = model.predict(dtest)

val_pr   = average_precision_score(y_val,  y_val_pred)
test_pr  = average_precision_score(y_test, y_test_pred)
val_roc  = roc_auc_score(y_val,  y_val_pred)
test_roc = roc_auc_score(y_test, y_test_pred)

logging.info("===================================")
logging.info("Best iteration    : %d",   model.best_iteration)
logging.info("Validation AUC-PR : %.6f", val_pr)
logging.info("Test AUC-PR       : %.6f", test_pr)
logging.info("Validation ROC-AUC: %.6f", val_roc)
logging.info("Test ROC-AUC      : %.6f", test_roc)
logging.info("===================================")

# %% [markdown]
# ## 6. Save model

# %%
import os
out_path = os.path.join(OUTPUT_DIR, "xgb_human_features_tuned_v4_pathways.json")
model.save_model(out_path)
logging.info("Model saved to %s", out_path)