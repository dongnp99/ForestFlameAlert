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

df = pd.read_csv(xgb_config.DATA_PATH, parse_dates=["date"])
df[xgb_config.FEATURE_COLS] = df[xgb_config.FEATURE_COLS].astype("float32")

train_df = df[df["date"] <= xgb_config.TRAIN_END_DATE].copy()
val_df = df[
    (df["date"] > xgb_config.TRAIN_END_DATE) &
    (df["date"] <= xgb_config.VAL_END_DATE)
].copy()
test_df = df[
    (df["date"] > xgb_config.VAL_END_DATE) &
    (df["date"] <= xgb_config.TEST_END_DATE)
].copy()

X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]

X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]

X_test = test_df[xgb_config.FEATURE_COLS]
y_test = test_df["fire"]

logging.info(f"Train size: {len(X_train)}")
logging.info(f"Val size: {len(X_val)}")
logging.info(f"Test size: {len(X_test)}")

del df
gc.collect()

# =============================
# BEST PARAMS FROM OPTUNA
# =============================

best_params = {
    'max_depth': 7,
    'min_child_weight': 3,
    'learning_rate': 0.024112764664411382,
    'subsample': 0.9266855235616202,
    'colsample_bytree': 0.9731718307245405,
    'gamma': 0.038276295122049345,
    'reg_lambda': 3.1196855460633235,
    'reg_alpha': 2.0083989794095136,
    'scale_pos_weight': 703.5395883034547,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": xgb_config.RANDOM_STATE
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

model.save_model("models/xgb_fire_tuned.json")
logging.info("Model saved to models/xgb_fire_tuned.json")