import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import xgb_config
import gc
import logging

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

# Load one split at a time, extract X/y, then delete before loading the next
logging.info("Loading train set...")
train_df = load_split([("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))])
logging.info("Train size: %s", len(train_df))
X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]
logging.info("Train fire rate: %.6f", y_train.mean())
del train_df
gc.collect()

logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE))
])
logging.info("Val size: %s", len(val_df))
X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]
del val_df
gc.collect()

logging.info("Loading test set...")
test_df = load_split([("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))])
logging.info("Test size: %s", len(test_df))
X_test = test_df[xgb_config.FEATURE_COLS]
y_test = test_df["fire"]
del test_df
gc.collect()

# =============================
# BEST PARAMS FROM OPTUNA
# =============================

best_params = {
  "max_depth":        8,
  "min_child_weight": 15,
  "learning_rate":    0.06257960621774133,
  "subsample":        0.8095304694689628,
  "colsample_bytree": 0.6546065241548528,
  "gamma":            0.7799726016810132,
  "reg_lambda":       8.0,        # adjusted from 1.05 — TPE consensus
  "reg_alpha":        4.330880728874676,
  "scale_pos_weight": 828.4237170569176,
  "objective":        "binary:logistic",
  "eval_metric":      "aucpr",
  "tree_method":      "hist",
  "device":           "cuda",
  "random_state":     42,
  "max_bin":          256,
  "grow_policy":      "lossguide"
}


# =============================
# CREATE DMATRIX
# =============================

logging.info("Creating QuantileDMatrix...")

dtrain = xgb.QuantileDMatrix(X_train, y_train)
dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)
dtest = xgb.QuantileDMatrix(X_test, y_test, ref=dtrain)

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

model.save_model("models/xgb_fire_after_tuned.json")
logging.info("Model saved to models/xgb_fire_after_tuned.json")