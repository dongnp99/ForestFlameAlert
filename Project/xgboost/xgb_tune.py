import gc
import os
import logging
from datetime import datetime
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import xgb_config

# ============================================================
# SETTINGS
# ============================================================

N_TRIALS         = 40
TRAIN_SAMPLE_SIZE = 15_000_000   # most-recent N rows by date

os.makedirs("logs", exist_ok=True)
_log_file = f"logs/xgb_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)

TRAIN_END = pd.to_datetime(xgb_config.TRAIN_END_DATE)
VAL_END   = pd.to_datetime(xgb_config.VAL_END_DATE)

# ============================================================
# HELPERS
# ============================================================

def load_split(date_filters):
    """Load FEATURE_COLS + fire + date (date needed for temporal sampling)."""
    return pd.read_parquet(
        xgb_config.DATA_PATH,
        columns=xgb_config.FEATURE_COLS + ["fire", "date"],
        filters=date_filters,
        engine="pyarrow",
    )

# ============================================================
# TRAIN  — load → sample → DMatrix → free
# ============================================================

logging.info("Loading train set...")
train_df = load_split([("date", "<=", TRAIN_END)])
logging.info("Full train size: %s", len(train_df))

# Temporal sample: keep the most-recent TRAIN_SAMPLE_SIZE rows
if len(train_df) > TRAIN_SAMPLE_SIZE:
    # argsort on a single int64 Series is cheap (~120 MB for 15M rows)
    # tail() returns a view, .copy() owns the memory so in-place ops are safe
    keep_idx = np.argpartition(train_df["date"].values, -TRAIN_SAMPLE_SIZE)[-TRAIN_SAMPLE_SIZE:]
    keep_idx.sort()                                      # restore temporal order
    train_df = train_df.iloc[keep_idx].copy()
    del keep_idx
    gc.collect()
    logging.info("After sampling: %s", len(train_df))

# Drop date (not a feature); compute weights before pop("fire")
del train_df["date"]

neg = (train_df["fire"] == 0).sum()
pos = (train_df["fire"] == 1).sum()
base_spw = neg / pos
logging.info("Train fire rate   : %.6f", pos / (neg + pos))
logging.info("Base scale_pos_weight: %.2f", base_spw)

_p1 = (train_df["burn_season_flag"] == 1) & (train_df["days_since_harvest"] < 30)
_p2 = (train_df["deforestation_lag_1y"] > 1.5) & (train_df["fire_count_prev_year"] > 0)
_p3 = (train_df["fire_count_prev_year"] > 1) & (train_df["burn_season_flag"] == 1)
human_fire_count = int(((train_df["fire"] == 1) & (_p1 | _p2 | _p3)).sum())
logging.info("Human-fire samples upweighted: %d  (~44%% expected)", human_fire_count)

logging.info("Creating dtrain QuantileDMatrix...")
w_train  = xgb_config.compute_sample_weights(train_df)
y_train  = train_df.pop("fire").values                  # train_df now == FEATURE_COLS only
dtrain = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del train_df, y_train, w_train
gc.collect()

# ============================================================
# VAL  — load → DMatrix → free
# ============================================================

logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  TRAIN_END),
    ("date", "<=", VAL_END),
])
logging.info("Val size: %s", len(val_df))

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
        "max_depth":         trial.suggest_int("max_depth", 6, 11),
        "min_child_weight":  trial.suggest_int("min_child_weight", 2, 15),
        "learning_rate":     trial.suggest_float("learning_rate", 0.015, 0.08),
        "subsample":         trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 10.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
        "scale_pos_weight":  trial.suggest_float(
            "scale_pos_weight", base_spw * 0.8, base_spw * 1.3
        ),
        "objective":         "binary:logistic",
        "eval_metric":       "aucpr",
        "tree_method":       "hist",
        "device":            "cuda",
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
    sampler=optuna.samplers.TPESampler(seed=xgb_config.RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=150),
)
study.optimize(objective, n_trials=N_TRIALS)

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("models", exist_ok=True)
study.trials_dataframe().to_csv("models/xgb_human_features_optuna_trials_v4_pathways.csv", index=False)

logging.info("====================================")
logging.info("Best AUC-PR : %.6f", study.best_value)
logging.info("Best Params : %s",   study.best_params)
logging.info("====================================")
logging.info("Log written to: %s", _log_file)
