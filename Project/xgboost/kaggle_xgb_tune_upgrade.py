"""
kaggle_xgb_tune_upgrade.py
==========================
Optuna hyperparameter search with **Focal Loss** objective — self-contained for Kaggle Notebooks.

Key differences from kaggle_xgb_tune.py:
  - Replaces binary:logistic + scale_pos_weight  with a custom Focal Loss objective.
  - Focal gamma (focusing) and focal alpha (class weighting) are tuned by Optuna
    alongside the usual tree hyperparameters.
  - AUC-PR is computed via a custom eval metric (sigmoid applied to raw margins).
  - Human-fire pathway sample weights (V4) are still applied via DMatrix(weight=...).

Setup on Kaggle:
  1. Upload your dataset; keep DATA_PATH below pointing to the parquet file.
  2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100).
  3. Add a Code cell before this script:  !pip install -q optuna
  4. Run as script:  !python /kaggle/working/kaggle_xgb_tune_upgrade.py
     OR paste cell-by-cell into a notebook using the # %% markers.

Outputs written to /kaggle/working/:
  - xgb_focal_optuna_trials_v1.csv   (all trial results)

After the study finishes, paste the printed best_params block into
kaggle_xgb_retrain_upgrade.py and run that script to train the final model.
"""

# %% [markdown]
# ## 0. Install & imports

# %%
# !pip install -q optuna   # uncomment if running as a notebook cell

import gc
import logging
import os
from functools import partial

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# %% [markdown]
# ## 1. Paths & constants

# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v2_human.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

# Kaggle has ~16 GB RAM — keep sample size conservative
TRAIN_SAMPLE_SIZE = 12_000_000

# Fewer trials than local; each trial takes ~10–15 min on T4 with focal loss
N_TRIALS = 20

# Human-fire V4 pathway weights (same as original)
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

# %% [markdown]
# ## 2. Helpers

# %%
def load_split(date_filters):
    return pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire", "date"],
        filters=date_filters,
        engine="pyarrow",
    )


def compute_sample_weights(df):
    """V4 graduated weighting — score = number of human-fire pathways matched.
    P1: burn_season_flag==1 & days_since_harvest<30
    P2: deforestation_lag_1y>1.5 & fire_count_prev_year>0
    P3: fire_count_prev_year>1 & burn_season_flag==1
    Weight schedule: 0→1.0x, 1→1.5x, 2→2.5x, 3→3.5x
    These weights are multiplied INTO the focal loss gradients by XGBoost
    automatically — do NOT multiply inside the custom objective.
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


def make_focal_obj(focal_gamma: float, focal_alpha: float):
    """Return a closure implementing focal loss as an XGBoost custom objective.

    Focal Loss per sample:
        FL_+ = -alpha   * (1-p)^focal_gamma * log(p)       [y=1]
        FL_- = -(1-alpha) * p^focal_gamma  * log(1-p)      [y=0]

    Gradients w.r.t. raw score s (where p = sigmoid(s)):
        g_+ = alpha   * (1-p)^focal_gamma * [focal_gamma * p * log(p) - (1-p)]
        g_- = (1-alpha) * p^focal_gamma   * [p - (1-p) * focal_gamma * log(1-p)]

    Hessian: positive-definite approximation (standard practice for focal loss):
        h_+ = alpha   * (1-p)^focal_gamma * p * (1-p)
        h_- = (1-alpha) * p^focal_gamma   * p * (1-p)

    XGBoost multiplies the returned (grad, hess) by DMatrix sample weights
    automatically, so the human-fire pathway weights are applied on top without
    any manual multiplication here.

    Args:
        focal_gamma: focusing exponent. 0 = standard BCE. Typical range: [0.5, 3].
        focal_alpha: class weight for positives. 0.5 = neutral. Typical: [0.1, 0.5].
    """
    def focal_obj(predt: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-predt))
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)

        # --- Gradient ---
        g_pos = focal_alpha * (1.0 - p) ** focal_gamma * (
            focal_gamma * p * np.log(p) - (1.0 - p)
        )
        g_neg = (1.0 - focal_alpha) * p ** focal_gamma * (
            p - (1.0 - p) * focal_gamma * np.log(1.0 - p)
        )
        grad = np.where(y == 1, g_pos, g_neg)

        # --- Hessian (positive-definite approximation) ---
        h_pos = focal_alpha * (1.0 - p) ** focal_gamma * p * (1.0 - p)
        h_neg = (1.0 - focal_alpha) * p ** focal_gamma * p * (1.0 - p)
        hess = np.where(y == 1, h_pos, h_neg)

        # Clamp hessian away from 0 to keep XGBoost stable
        hess = np.clip(hess, 1e-6, None)

        return grad.astype(np.float32), hess.astype(np.float32)

    return focal_obj


def aucpr_metric(predt: np.ndarray, dtrain: xgb.DMatrix):
    """Custom eval metric: AUC-PR computed from raw margin predictions.

    With a custom objective, XGBoost does not apply any link function,
    so predt contains raw scores (logits). We apply sigmoid here before
    computing AUC-PR.

    Returns ("aucpr", score) — higher is better, monitored by EarlyStopping
    with maximize=True and by Optuna's XGBoostPruningCallback.
    """
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    score = average_precision_score(y, p)
    return "aucpr", score

# %% [markdown]
# ## 3. Load & prepare training data

# %%
TRAIN_END = pd.to_datetime(TRAIN_END_DATE)
VAL_END   = pd.to_datetime(VAL_END_DATE)

logging.info("Loading train set...")
train_df = load_split([("date", "<=", TRAIN_END)])
logging.info("Full train size: %d", len(train_df))

# Temporal sample: keep most-recent rows to fit in RAM
if len(train_df) > TRAIN_SAMPLE_SIZE:
    keep_idx = np.argpartition(train_df["date"].values, -TRAIN_SAMPLE_SIZE)[-TRAIN_SAMPLE_SIZE:]
    keep_idx.sort()
    train_df = train_df.iloc[keep_idx].copy()
    del keep_idx
    gc.collect()
    logging.info("After sampling: %d", len(train_df))

del train_df["date"]

neg = (train_df["fire"] == 0).sum()
pos = (train_df["fire"] == 1).sum()
logging.info("Train fire rate: %.6f  (neg=%d, pos=%d)", pos / (neg + pos), neg, pos)

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

# %% [markdown]
# ## 4. Load validation data

# %%
logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  TRAIN_END),
    ("date", "<=", VAL_END),
])
logging.info("Val size: %d", len(val_df))

del val_df["date"]
y_val = val_df.pop("fire").values
logging.info("Creating dval QuantileDMatrix...")
dval = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df, y_val
gc.collect()

# %% [markdown]
# ## 5. Optuna objective

# %%
def objective(trial):
    # ── Focal loss hyperparameters ─────────────────────────────────────────────
    # focal_gamma: 0 = standard BCE; higher = more focus on hard/misclassified
    #              examples; anomalous human fires are typically hard examples
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)

    # focal_alpha: weight for the positive class (fires).
    #              Note: this is NOT scale_pos_weight — it operates multiplicatively
    #              inside the loss, not as a resampling correction.
    #              Range [0.1, 0.5]: even 0.1 strongly upweights positives given
    #              the ~1:638 imbalance.
    focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.5)

    # ── Tree structure hyperparameters (same search space as original) ─────────
    tree_params = {
        "max_depth":        trial.suggest_int("max_depth", 6, 11),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
        "learning_rate":    trial.suggest_float("learning_rate", 0.015, 0.08),
        "subsample":        trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),   # min_split_loss
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        # No objective / eval_metric / scale_pos_weight — handled by focal loss
        "tree_method":  "hist",
        "device":       "cuda",
        "max_bin":      256,
        "grow_policy":  "lossguide",
    }

    focal_obj = make_focal_obj(focal_gamma, focal_alpha)

    # EarlyStopping must know to maximize AUC-PR (not minimize)
    early_stop = xgb.callback.EarlyStopping(
        rounds=120,
        metric_name="aucpr",
        data_name="validation",
        maximize=True,
        save_best=True,
    )

    # Optuna pruning: monitors "validation-aucpr"
    # Format: "{evals set name}-{metric name returned by aucpr_metric}"
    pruning_cb = optuna.integration.XGBoostPruningCallback(trial, "validation-aucpr")

    booster = xgb.train(
        tree_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        obj=focal_obj,
        custom_metric=aucpr_metric,
        callbacks=[early_stop, pruning_cb],
        verbose_eval=False,
    )

    # best_score is set by EarlyStopping(save_best=True)
    return booster.best_score

# %% [markdown]
# ## 6. Run study

# %%
logging.info("Starting Optuna study  (n_trials=%d)...", N_TRIALS)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=150),
)
study.optimize(objective, n_trials=N_TRIALS)

# %% [markdown]
# ## 7. Save results

# %%
out_csv = os.path.join(OUTPUT_DIR, "xgb_focal_optuna_trials_v1.csv")
study.trials_dataframe().to_csv(out_csv, index=False)

logging.info("====================================")
logging.info("Best AUC-PR  : %.6f", study.best_value)
logging.info("Best Params  : %s",   study.best_params)
logging.info("Trials saved : %s",   out_csv)
logging.info("====================================")
logging.info("")
logging.info("── Paste the block below into kaggle_xgb_retrain_upgrade.py ──")
logging.info("FOCAL_GAMMA = %s", study.best_params.get("focal_gamma"))
logging.info("FOCAL_ALPHA = %s", study.best_params.get("focal_alpha"))
tree_only = {k: v for k, v in study.best_params.items()
             if k not in ("focal_gamma", "focal_alpha")}
logging.info("BEST_TREE_PARAMS = %s", tree_only)
