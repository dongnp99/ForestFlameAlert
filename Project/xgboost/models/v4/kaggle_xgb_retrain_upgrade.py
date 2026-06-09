"""
kaggle_xgb_retrain_upgrade.py
==============================
Retrain final model with best Focal Loss params from Optuna — self-contained for Kaggle Notebooks.

Setup on Kaggle:
  1. Run kaggle_xgb_tune_upgrade.py first and copy the printed best_params block.
  2. Paste FOCAL_GAMMA, FOCAL_ALPHA, and BEST_TREE_PARAMS below.
  3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100).
  4. Run as script:  !python /kaggle/working/kaggle_xgb_retrain_upgrade.py
     OR paste cell-by-cell into a notebook using the # %% markers.

Outputs written to /kaggle/working/:
  - xgb_focal_tuned_v1.json            (model — load with Booster.load_model)
  - xgb_focal_val_probs.npy            (calibrated val probabilities)
  - xgb_focal_test_probs.npy           (calibrated test probabilities)
  - xgb_focal_val_labels.npy           (val ground truth)
  - xgb_focal_test_labels.npy          (test ground truth)

IMPORTANT — loading the saved model for inference:
  Because a custom objective is used, XGBoost does NOT store the sigmoid
  transform inside the model file. When loading later, always do:

      booster = xgb.Booster()
      booster.load_model("xgb_focal_tuned_v1.json")
      raw = booster.predict(dmatrix, output_margin=True)   # raw logits
      probs = 1.0 / (1.0 + np.exp(-raw))                  # sigmoid → probabilities
"""

# %% [markdown]
# ## 0. Imports

# %%
import gc
import logging
import os

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

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v2_human.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

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

#  Paste from kaggle_xgb_tune_upgrade.py output here 

FOCAL_GAMMA = 1.2466160831748727    # <-- replace with best trial value
FOCAL_ALPHA = 0.4995059822923817   # <-- replace with best trial value

BEST_TREE_PARAMS = {
    "max_depth":        9,      # <-- replace with best trial value
    "min_child_weight": 5,      # <-- replace with best trial value
    "learning_rate":    0.024611543338188106,   # <-- replace with best trial value
    "subsample":        0.6765125600300572,   # <-- replace with best trial value
    "colsample_bytree": 0.7611258402402383,   # <-- replace with best trial value
    "gamma":            4.111739565312337,    # <-- replace with best trial value (min_split_loss)
    "reg_lambda":       4.076202262616299,    # <-- replace with best trial value
    "reg_alpha":        3.3730604019224977,    # <-- replace with best trial value
    # Fixed settings — do not change
    "tree_method":  "hist",
    "device":       "cuda",
    "random_state": RANDOM_STATE,
    "max_bin":      256,
    "grow_policy":  "lossguide",
    # No objective / eval_metric / scale_pos_weight — handled by focal loss
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
    """V4 graduated weighting — score = number of human-fire pathways matched."""
    fire = (df["fire"] == 1).values
    p1 = ((df["burn_season_flag"] == 1) & (df["days_since_harvest"] < 30)).values
    p2 = ((df["deforestation_lag_1y"] > 1.5) & (df["fire_count_prev_year"] > 0)).values
    p3 = ((df["fire_count_prev_year"] > 1) & (df["burn_season_flag"] == 1)).values
    score = p1.astype(np.int8) + p2.astype(np.int8) + p3.astype(np.int8)
    weights_map = np.array(HUMAN_FIRE_WEIGHTS, dtype="float32")
    weights = np.ones(len(df), dtype="float32")
    weights[fire] = weights_map[score[fire]]
    return weights


def make_focal_obj(focal_gamma: float, focal_alpha: float):
    """Focal loss custom objective for xgb.train(obj=...).

    Gradients w.r.t. raw score s (where p = sigmoid(s)):
        g_+ = alpha   * (1-p)^gamma * [gamma * p * log(p) - (1-p)]   [y=1]
        g_- = (1-alpha) * p^gamma   * [p - (1-p) * gamma * log(1-p)] [y=0]

    Hessian (positive-definite approximation):
        h_+ = alpha   * (1-p)^gamma * p*(1-p)
        h_- = (1-alpha) * p^gamma   * p*(1-p)

    XGBoost multiplies returned (grad, hess) by DMatrix sample weights
    automatically — do NOT apply weights manually inside this function.
    """
    def focal_obj(predt: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-predt))
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)

        g_pos = focal_alpha * (1.0 - p) ** focal_gamma * (
            focal_gamma * p * np.log(p) - (1.0 - p)
        )
        g_neg = (1.0 - focal_alpha) * p ** focal_gamma * (
            p - (1.0 - p) * focal_gamma * np.log(1.0 - p)
        )
        grad = np.where(y == 1, g_pos, g_neg)

        h_pos = focal_alpha * (1.0 - p) ** focal_gamma * p * (1.0 - p)
        h_neg = (1.0 - focal_alpha) * p ** focal_gamma * p * (1.0 - p)
        hess = np.where(y == 1, h_pos, h_neg)

        hess = np.clip(hess, 1e-6, None)

        return grad.astype(np.float32), hess.astype(np.float32)

    return focal_obj


def aucpr_metric(predt: np.ndarray, dtrain: xgb.DMatrix):
    """Custom eval metric: AUC-PR from raw margin predictions."""
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    score = average_precision_score(y, p)
    return "aucpr", score


def raw_to_probs(raw: np.ndarray) -> np.ndarray:
    """Apply sigmoid to convert raw margins to probabilities."""
    return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

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
logging.info("focal_gamma=%.4f  focal_alpha=%.4f", FOCAL_GAMMA, FOCAL_ALPHA)
logging.info("Training final focal-loss model...")

focal_obj = make_focal_obj(FOCAL_GAMMA, FOCAL_ALPHA)

early_stop = xgb.callback.EarlyStopping(
    rounds=100,
    metric_name="aucpr",
    data_name="val",
    maximize=True,
    save_best=True,
)

model = xgb.train(
    BEST_TREE_PARAMS,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    obj=focal_obj,
    custom_metric=aucpr_metric,
    callbacks=[early_stop],
    verbose_eval=100,
)

# %% [markdown]
# ## 5. Evaluate

# %%
logging.info("Evaluating (applying sigmoid to raw margins)...")

# output_margin=True: explicitly get raw logits; apply sigmoid manually.
# Required because a custom objective does not store a link function.
y_val_raw  = model.predict(dval,  output_margin=True)
y_test_raw = model.predict(dtest, output_margin=True)

y_val_prob  = raw_to_probs(y_val_raw)
y_test_prob = raw_to_probs(y_test_raw)

val_pr   = average_precision_score(y_val,  y_val_prob)
test_pr  = average_precision_score(y_test, y_test_prob)
val_roc  = roc_auc_score(y_val,  y_val_prob)
test_roc = roc_auc_score(y_test, y_test_prob)

logging.info("===================================")
logging.info("Best iteration    : %d",   model.best_iteration)
logging.info("Validation AUC-PR : %.6f", val_pr)
logging.info("Test AUC-PR       : %.6f", test_pr)
logging.info("Validation ROC-AUC: %.6f", val_roc)
logging.info("Test ROC-AUC      : %.6f", test_roc)
logging.info("===================================")

# %% [markdown]
# ## 6. Save model & predictions

# %%
out_model = os.path.join(OUTPUT_DIR, "xgb_focal_tuned_v1.json")
model.save_model(out_model)
logging.info("Model saved to %s", out_model)

# Save probability arrays for downstream calibration / threshold analysis
np.save(os.path.join(OUTPUT_DIR, "xgb_focal_val_probs.npy"),   y_val_prob)
np.save(os.path.join(OUTPUT_DIR, "xgb_focal_test_probs.npy"),  y_test_prob)
np.save(os.path.join(OUTPUT_DIR, "xgb_focal_val_labels.npy"),  y_val)
np.save(os.path.join(OUTPUT_DIR, "xgb_focal_test_labels.npy"), y_test)
logging.info("Probability arrays saved to %s", OUTPUT_DIR)
logging.info("")
logging.info(" Reminder: load for inference with ")
logging.info("    booster = xgb.Booster()")
logging.info("    booster.load_model('xgb_focal_tuned_v1.json')")
logging.info("    raw   = booster.predict(dmatrix, output_margin=True)")
logging.info("    probs = 1.0 / (1.0 + np.exp(-raw))")
logging.info("")
