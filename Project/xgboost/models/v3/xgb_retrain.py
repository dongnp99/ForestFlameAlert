import os
import gc
import logging
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import xgb_config

# =============================
# LOGGING
# =============================

os.makedirs("../../logs", exist_ok=True)
_log_file = f"logs/xgb_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)

logging.info("Loading data...")

# =============================
# LOAD DATA
# =============================
def load_split(filters):
    # Only load feature cols + fire — skip unused columns to save RAM
    df = pd.read_parquet(
        xgb_config.DATA_PATH,
        columns=xgb_config.FEATURE_COLS + ["fire"],
        filters=filters,
        engine="pyarrow"
    )
    df["fire"] = df["fire"].astype("int8")
    for col in xgb_config.FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df

# Load one split at a time, create DMatrix immediately, then delete before loading the next
logging.info("Loading train set...")
train_df = load_split([("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))])
logging.info("Train size: %s  |  fire rate: %.6f", len(train_df), train_df["fire"].mean())

_p1 = (train_df["burn_season_flag"] == 1) & (train_df["days_since_harvest"] < 30)
_p2 = (train_df["deforestation_lag_1y"] > 1.5) & (train_df["fire_count_prev_year"] > 0)
_p3 = (train_df["fire_count_prev_year"] > 1) & (train_df["burn_season_flag"] == 1)
human_fire_count = int(((train_df["fire"] == 1) & (_p1 | _p2 | _p3)).sum())
logging.info("Human-fire samples upweighted: %d  (~44%% expected)", human_fire_count)

logging.info("Creating dtrain QuantileDMatrix...")
# Compute weights before pop("fire"); train_df is now FEATURE_COLS only after pop
w_train = xgb_config.compute_sample_weights(train_df)
y_train = train_df.pop("fire").values
dtrain  = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del train_df, y_train, w_train
gc.collect()

logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE)),
])
logging.info("Val size: %s", len(val_df))
logging.info("Creating dval QuantileDMatrix...")
y_val = val_df.pop("fire").values
dval  = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df
gc.collect()

logging.info("Loading test set...")
test_df = load_split([("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))])
logging.info("Test size: %s", len(test_df))
logging.info("Creating dtest QuantileDMatrix...")
y_test = test_df.pop("fire").values
dtest  = xgb.QuantileDMatrix(test_df, y_test, ref=dtrain)
del test_df
gc.collect()

# =============================
# BEST PARAMS FROM OPTUNA
# =============================

best_params = {
    "max_depth": 8,
    "min_child_weight": 2,
    "learning_rate": 0.05741824143332552,
    "subsample": 0.7705102323506257,
    "colsample_bytree": 0.6733292800810863,
    "gamma": 4.661230074006847,
    "reg_lambda": 4.471351286348217,
    "reg_alpha": 4.631961421397106,
    "scale_pos_weight": 638.4840070541702,
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "device":           "cuda",
    "random_state":     42,
    "max_bin":          256,
    "grow_policy":      "lossguide"
}

# =============================
# TRAIN FINAL MODEL
# =============================

logging.info("Training final tuned model...")

model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=100
)

# =============================
# EVALUATE
# =============================

logging.info("Evaluating...")

y_val_pred = model.predict(dval)
y_test_pred = model.predict(dtest)

val_pr = average_precision_score(y_val, y_val_pred)
test_pr = average_precision_score(y_test, y_test_pred)

val_roc = roc_auc_score(y_val, y_val_pred)
test_roc = roc_auc_score(y_test, y_test_pred)

logging.info("===================================")
logging.info(f"Best iteration: {model.best_iteration}")
logging.info(f"Validation AUC-PR: {val_pr:.6f}")
logging.info(f"Test AUC-PR: {test_pr:.6f}")
logging.info(f"Validation ROC-AUC: {val_roc:.6f}")
logging.info(f"Test ROC-AUC: {test_roc:.6f}")
logging.info("===================================")

# =============================
# SAVE MODEL
# =============================

model.save_model("models/xgb_human_features_tuned_v4_pathways.json")
logging.info("Model saved to models/xgb_human_features_tuned_v4_pathways.json")
logging.info("Log written to: %s", _log_file)