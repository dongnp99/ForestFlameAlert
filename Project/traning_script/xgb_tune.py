import os
import logging
import gc
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
import xgb_config

# =============================
# SETTINGS
# =============================

N_TRIALS = 30                # Không cần 80
SAMPLE_SIZE = 6_000_000      # Giảm để nhanh hơn
USE_GPU = True

# =============================
# LOGGING
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("Loading data...")

# =============================
# LOAD DATA
# =============================

df = pd.read_csv(
    xgb_config.DATA_PATH,
    parse_dates=["date"],
    dtype={col: "float32" for col in xgb_config.FEATURE_COLS}
)

df["fire"] = df["fire"].astype("int8")

train_df = df[df["date"] <= xgb_config.TRAIN_END_DATE]
val_df   = df[(df["date"] > xgb_config.TRAIN_END_DATE) &
              (df["date"] <= xgb_config.VAL_END_DATE)]

if len(train_df) > SAMPLE_SIZE:
    train_df = train_df.sample(SAMPLE_SIZE, random_state=42)

X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]

X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
base_spw = neg / pos

logging.info(f"Base scale_pos_weight: {base_spw:.2f}")

# =============================
# BUILD DMATRIX ONCE
# =============================

dtrain = xgb.QuantileDMatrix(X_train, y_train)
dval   = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)

del df, train_df
gc.collect()

# =============================
# OBJECTIVE
# =============================

def objective(trial):

    params = {
        # 🔥 Depth tập trung vùng tốt
        "max_depth": trial.suggest_int("max_depth", 6, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),

        # 🔥 Learning rate hẹp lại
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12),

        # 🔥 Sampling tối ưu cho imbalance
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),

        # 🔥 Regularization
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),

        # 🔥 Weight tuning (rất quan trọng)
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            base_spw * 0.6,
            base_spw * 1.0
        ),

        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda"
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[(dval, "validation")],
        early_stopping_rounds=80,
        verbose_eval=False
    )

    preds = booster.predict(dval)
    auc_pr = average_precision_score(y_val, preds)

    return auc_pr


# =============================
# RUN STUDY
# =============================

logging.info("Starting Optuna study...")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS)

# =============================
# SAVE RESULTS
# =============================

os.makedirs("models", exist_ok=True)

study.trials_dataframe().to_csv("models/optuna_gpu_trials.csv", index=False)

logging.info("====================================")
logging.info(f"Best AUC-PR: {study.best_value:.6f}")
logging.info(f"Best Params: {study.best_params}")
logging.info("====================================")