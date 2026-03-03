import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import xgb_config
import gc

print("Loading parquet data...")

# ==========================================================
# 1️⃣ LOAD TRAIN / VAL / TEST DIRECTLY FROM PARQUET
# ==========================================================

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


print("Loading train set...")
train_df = load_split([
    ("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))
])

print("Loading val set...")
val_df = load_split([
    ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE))
])

print("Loading test set...")
test_df = load_split([
    ("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))
])

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# ==========================================================
# 2️⃣ SAFE NEIGHBOR RATIO (NO NaN, NO float64)
# ==========================================================

def add_neighbor_ratio(df):

    for w in [1, 3, 7]:
        col = f"neighbor_fire_{w}d"

        if col in df.columns:

            num = df[col].values.astype(np.float32)
            den = df["neighbor_count"].values.astype(np.float32)

            ratio = np.divide(
                num,
                den,
                out=np.zeros_like(num, dtype=np.float32),
                where=den != 0
            )

            df[f"neighbor_fire_ratio_{w}d"] = ratio.astype("float32")

    return df


print("Creating neighbor_fire_ratio...")
train_df = add_neighbor_ratio(train_df)
val_df   = add_neighbor_ratio(val_df)
test_df  = add_neighbor_ratio(test_df)

gc.collect()

# ==========================================================
# 3️⃣ CREATE MATRICES (NO EXTRA COPIES)
# ==========================================================

X_train = train_df[xgb_config.FEATURE_COLS]
y_train = train_df["fire"]

X_val = val_df[xgb_config.FEATURE_COLS]
y_val = val_df["fire"]

X_test = test_df[xgb_config.FEATURE_COLS]
y_test = test_df["fire"]

print("Train fire rate:", y_train.mean())

del train_df, val_df, test_df
gc.collect()

# ==========================================================
# 4️⃣ SCALE POS WEIGHT
# ==========================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)

# ==========================================================
# 5️⃣ MODEL PARAMS
# ==========================================================

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": xgb_config.RANDOM_STATE,
    "scale_pos_weight": scale_pos_weight,
}

# ==========================================================
# 6️⃣ QUANTILE DMATRIX (GPU SAFE)
# ==========================================================

print("Creating QuantileDMatrix...")

dtrain = xgb.QuantileDMatrix(X_train, y_train)
dval   = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)
dtest  = xgb.QuantileDMatrix(X_test, y_test, ref=dtrain)

del X_train, X_val, X_test
gc.collect()

# ==========================================================
# 7️⃣ TRAIN
# ==========================================================

print("Training...")

model = xgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=100
)

# ==========================================================
# 8️⃣ EVALUATE
# ==========================================================

print("Evaluating...")

y_val_pred = model.predict(dval)
y_test_pred = model.predict(dtest)

val_pr = average_precision_score(y_val, y_val_pred)
test_pr = average_precision_score(y_test, y_test_pred)

val_roc = roc_auc_score(y_val, y_val_pred)
test_roc = roc_auc_score(y_test, y_test_pred)

print("===================================")
print("Best iteration:", model.best_iteration)
print("Validation AUC-PR:", val_pr)
print("Test AUC-PR:", test_pr)
print("Validation ROC-AUC:", val_roc)
print("Test ROC-AUC:", test_roc)
print("===================================")

model.save_model("models/xgb_fire_full_gpu_af.json")
print("Model saved.")