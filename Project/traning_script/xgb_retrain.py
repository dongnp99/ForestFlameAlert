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

    df = pd.read_parquet(
        xgb_config.DATA_PATH,
        filters=filters,
        engine="pyarrow"
    )

    # đảm bảo dtype nhỏ
    df["fire"] = df["fire"].astype("int8")
    df["neighbor_count"] = df["neighbor_count"].astype("int8")

    for col in xgb_config.FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    return df

logging.info("Loading train set...")
train_df = load_split([
    ("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))
])

logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE))
])

logging.info("Loading test set...")
test_df = load_split([
    ("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))
])

logging.info("Train size: %s", len(train_df))
logging.info("Val size: %s", len(val_df))
logging.info("Test size: %s", len(test_df))
gc.collect()


X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]

X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]

X_test = test_df[xgb_config.FEATURE_COLS]
y_test = test_df["fire"]

logging.info("Train fire rate: %.6f", y_train.mean())

del train_df, val_df, test_df
gc.collect()

# =============================
# BEST PARAMS FROM OPTUNA
# =============================

best_params = {
    "max_depth": 6,
    "min_child_weight": 14,
    "learning_rate": 0.05492840731823069,
    "subsample": 0.7380574174836383,
    "colsample_bytree": 0.845377362927354,
    "gamma": 1.1268230706741285,
    "reg_lambda": 8.89203532155487,
    "reg_alpha": 3.770634003396035,
    "scale_pos_weight": 672.4936647367721,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
    "max_bin": 256,
    "grow_policy": "lossguide"
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