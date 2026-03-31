import gc
import logging
import os
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import xgb_config

os.makedirs("logs", exist_ok=True)
_log_file = f"logs/xgb_baseline_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)

logging.info("Loading parquet data...")

# ============================================================
# HELPERS
# ============================================================

def load_split(filters):
    """Load only FEATURE_COLS + fire — no date, grid_id, or other columns."""
    df = pd.read_parquet(
        xgb_config.DATA_PATH,
        columns=xgb_config.FEATURE_COLS + ["fire"],
        filters=filters,
        engine="pyarrow",
    )
    df["fire"] = df["fire"].astype("int8")
    for col in xgb_config.FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


# ============================================================
# SCALE_POS_WEIGHT  (from train labels only, cheap)
# ============================================================

logging.info("Loading train set...")
train_df = load_split([("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))])
logging.info("Train size : %s  |  fire rate : %.6f", f"{len(train_df):,}", train_df["fire"].mean())

neg = (train_df["fire"] == 0).sum()
pos = (train_df["fire"] == 1).sum()
scale_pos_weight = neg / pos
logging.info("scale_pos_weight : %.2f", scale_pos_weight)

_p1 = (train_df["burn_season_flag"] == 1) & (train_df["days_since_harvest"] < 30)
_p2 = (train_df["deforestation_lag_1y"] > 1.5) & (train_df["fire_count_prev_year"] > 0)
_p3 = (train_df["fire_count_prev_year"] > 1) & (train_df["burn_season_flag"] == 1)
human_fire_count = int(((train_df["fire"] == 1) & (_p1 | _p2 | _p3)).sum())
logging.info("Human-fire samples upweighted : %s  (~44%% expected)", f"{human_fire_count:,}")

# ============================================================
# CREATE DMATRICES — one at a time, pop() to avoid column copy
# ============================================================

# Train: compute weights before pop("fire"); train_df now == FEATURE_COLS only
logging.info("Creating dtrain QuantileDMatrix...")
w_train = xgb_config.compute_sample_weights(train_df)
y_train = train_df.pop("fire").values
dtrain  = xgb.QuantileDMatrix(train_df, y_train, weight=w_train)
del w_train
del train_df, y_train
gc.collect()

# Val
logging.info("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE)),
])
logging.info("Val size : %s", f"{len(val_df):,}")
logging.info("Creating dval QuantileDMatrix...")
y_val  = val_df.pop("fire").values
dval   = xgb.QuantileDMatrix(val_df, y_val, ref=dtrain)
del val_df
gc.collect()

# Test
logging.info("Loading test set...")
test_df = load_split([("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))])
logging.info("Test size : %s", f"{len(test_df):,}")
logging.info("Creating dtest QuantileDMatrix...")
y_test = test_df.pop("fire").values
dtest  = xgb.QuantileDMatrix(test_df, y_test, ref=dtrain)
del test_df
gc.collect()

# ============================================================
# PARAMS
# ============================================================

params = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist",
    "device":           "cuda",
    "random_state":     xgb_config.RANDOM_STATE,
    "scale_pos_weight": scale_pos_weight,
}

# ============================================================
# TRAIN
# ============================================================

logging.info("Training...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=100,
)

# ============================================================
# EVALUATE
# ============================================================

logging.info("Evaluating...")
y_val_pred  = model.predict(dval)
y_test_pred = model.predict(dtest)

val_pr   = average_precision_score(y_val,  y_val_pred)
test_pr  = average_precision_score(y_test, y_test_pred)
val_roc  = roc_auc_score(y_val,  y_val_pred)
test_roc = roc_auc_score(y_test, y_test_pred)

logging.info("=========================================")
logging.info("Best iteration   : %d", model.best_iteration)
logging.info("Validation AUC-PR: %.6f", val_pr)
logging.info("Test AUC-PR      : %.6f", test_pr)
logging.info("Validation ROC-AUC: %.6f", val_roc)
logging.info("Test ROC-AUC      : %.6f", test_roc)
logging.info("=========================================")

model.save_model("models/xgb_human_features_baseline.json")
logging.info("Model saved.")
logging.info("Log written to: %s", _log_file)
