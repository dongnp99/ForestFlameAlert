import os
import gc
import logging
import optuna
import pandas as pd
import xgboost as xgb
import xgb_config
from datetime import datetime
# =====================================
# SETTINGS
# =====================================

N_TRIALS = 40
TRAIN_SAMPLE_SIZE = 15_000_000   # temporal sample
USE_GPU = True

# =====================================
# LOGGING
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Loading parquet data...")

# =====================================
# LOAD PARQUET SPLIT BY TIME
# =====================================
TRAIN_END = pd.to_datetime(xgb_config.TRAIN_END_DATE)
VAL_END   = pd.to_datetime(xgb_config.VAL_END_DATE)
def load_split(date_filters):
    df = pd.read_parquet(
        xgb_config.DATA_PATH,
        columns=xgb_config.FEATURE_COLS + ["fire", "date"],
        filters=date_filters
    )
    return df

train_df = load_split([
    ("date", "<=", TRAIN_END)
])

val_df = load_split([
    ("date", ">", TRAIN_END),
    ("date", "<=", VAL_END)
])

# =====================================
# TEMPORAL SAMPLING (GIỮ STRUCTURE)
# =====================================

if len(train_df) > TRAIN_SAMPLE_SIZE:
    train_df = (
        train_df
        .sort_values("date")
        .iloc[-TRAIN_SAMPLE_SIZE:]
    )

X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]

X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
base_spw = neg / pos

logging.info(f"Base scale_pos_weight: {base_spw:.2f}")

# =====================================
# BUILD QUANTILE DMATRIX
# =====================================

dtrain = xgb.QuantileDMatrix(X_train, y_train)
dval   = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)

del train_df, val_df
gc.collect()

# =====================================
# OPTUNA OBJECTIVE
# =====================================

def objective(trial):

    params = {
        "max_depth": trial.suggest_int("max_depth", 6, 11),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            base_spw * 0.8,
            base_spw * 1.3
        ),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda"
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial,
        "validation-aucpr"
    )

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        early_stopping_rounds=120,
        callbacks=[pruning_callback],
        verbose_eval=False
    )

    return booster.best_score


# =====================================
# RUN STUDY
# =====================================

logging.info("Starting Optuna study...")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=150),
)

study.optimize(objective, n_trials=N_TRIALS)

# =====================================
# SAVE RESULTS
# =====================================

os.makedirs("models", exist_ok=True)

study.trials_dataframe().to_csv(
    "models/optuna_gpu_trials.csv",
    index=False
)

logging.info("====================================")
logging.info(f"Best AUC-PR: {study.best_value:.6f}")
logging.info(f"Best Params: {study.best_params}")
logging.info("====================================")