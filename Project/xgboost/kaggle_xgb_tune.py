"""
kaggle_xgb_tune.py
==================
Optuna hyperparameter search — self-contained for Kaggle Notebooks.

Setup on Kaggle:
  1. Upload your dataset and set DATASET_SLUG below (the folder name under /kaggle/input/).
  2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100).
  3. Add a Code cell before this script: !pip install -q optuna
  4. Run as script:  !python /kaggle/working/kaggle_xgb_tune.py
     OR paste cell-by-cell into a notebook using the # %% markers.

Outputs written to /kaggle/working/:
  - xgb_human_features_optuna_trials_v2.csv
"""

# %% [markdown]
# ## 0. Install & imports

# %%
# !pip install -q optuna   # uncomment if running as a notebook cell

import gc
import logging
import os
import sys

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# %% [markdown]
# ## 1. Paths & constants

# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH  = f"/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v2_human.parquet"
OUTPUT_DIR = "/kaggle/working"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"
RANDOM_STATE   = 42

# Kaggle has ~16 GB RAM — keep sample size conservative
TRAIN_SAMPLE_SIZE = 12_000_000

# Fewer trials than local runs; each trial takes ~10 min on T4
N_TRIALS = 20

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

neg     = (train_df["fire"] == 0).sum()
pos     = (train_df["fire"] == 1).sum()
base_spw = neg / pos
logging.info("Train fire rate      : %.6f", pos / (neg + pos))
logging.info("Base scale_pos_weight: %.2f", base_spw)

_p1 = (train_df["burn_season_flag"] == 1) & (train_df["days_since_harvest"] < 30)
_p2 = (train_df["deforestation_lag_1y"] > 1.5) & (train_df["fire_count_prev_year"] > 0)
_p3 = (train_df["fire_count_prev_year"] > 1) & (train_df["burn_season_flag"] == 1)
human_count = int(((train_df["fire"] == 1) & (_p1 | _p2 | _p3)).sum())
logging.info("Human-fire samples upweighted: %d  (~44%% expected)", human_count)

logging.info("Creating dtrain QuantileDMatrix...")
w_train  = compute_sample_weights(train_df)
y_train  = train_df.pop("fire").values
dtrain   = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
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
    params = {
        "max_depth":        trial.suggest_int("max_depth", 6, 11),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
        "learning_rate":    trial.suggest_float("learning_rate", 0.015, 0.08),
        "subsample":        trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", base_spw * 0.8, base_spw * 1.3
        ),
        "objective":    "binary:logistic",
        "eval_metric":  "aucpr",
        "tree_method":  "hist",
        "device":       "cuda",
        "max_bin":      256,
        "grow_policy":  "lossguide",
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
out_csv = os.path.join(OUTPUT_DIR, "xgb_human_features_optuna_trials_v4_pathways.csv")
study.trials_dataframe().to_csv(out_csv, index=False)

logging.info("====================================")
logging.info("Best AUC-PR : %.6f", study.best_value)
logging.info("Best Params : %s",   study.best_params)
logging.info("Trials saved: %s",   out_csv)
logging.info("====================================")