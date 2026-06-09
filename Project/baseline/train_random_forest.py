"""
train_random_forest.py
Random Forest baseline trên dataset v3_pathways.

Chạy từ Project/baseline/:
    python train_random_forest.py
"""

import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline_config import (
    FEATURE_COLS, RESULTS_DIR,
    TRAIN_END_DATE, VAL_END_DATE,
    RANDOM_STATE, RF_MAX_TRAIN_SAMPLES,
    load_split, setup_logging, append_comparison,
)

# ==============================
# HYPERPARAMETERS
# ==============================

RF_PARAMS = {
    "n_estimators":     500,        # tăng từ 300: ổn định hơn với nhiều data hơn
    "max_depth":        20,
    "min_samples_leaf": 5,          # giảm từ 10: tinh chỉnh hơn khi train set lớn hơn
    "max_features":     "sqrt",
    "class_weight":     "balanced_subsample",  # tính balance riêng cho từng cây
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
}

# ==============================
# SETUP
# ==============================

RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "random_forest"
log        = setup_logging(MODEL_NAME, RUN_ID)

log.info("=" * 60)
log.info("Random Forest Baseline")
log.info("=" * 60)
log.info("Params          : %s", RF_PARAMS)
log.info("Max train samples: %s", f"{RF_MAX_TRAIN_SAMPLES:,}")
log.info("Num features    : %d", len(FEATURE_COLS))

# ==============================
# LOAD DATA
# ==============================

log.info("Loading train set (≤ %s)...", TRAIN_END_DATE)
train_df = load_split(
    [("date", "<=", pd.Timestamp(TRAIN_END_DATE))],
    max_samples=RF_MAX_TRAIN_SAMPLES,
)
log.info("Train size: %s  |  fire rate: %.6f", f"{len(train_df):,}", train_df["fire"].mean())

y_train = train_df.pop("fire").values
X_train = train_df[FEATURE_COLS].values

log.info("Loading val set (%s – %s)...", TRAIN_END_DATE, VAL_END_DATE)
val_df = load_split([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
log.info("Val size: %s  |  fire rate: %.6f", f"{len(val_df):,}", val_df["fire"].mean())
y_val = val_df.pop("fire").values
X_val = val_df[FEATURE_COLS].values

log.info("Loading test set (> %s)...", VAL_END_DATE)
test_df = load_split([("date", ">", pd.Timestamp(VAL_END_DATE))])
log.info("Test size: %s  |  fire rate: %.6f", f"{len(test_df):,}", test_df["fire"].mean())
y_test = test_df.pop("fire").values
X_test = test_df[FEATURE_COLS].values

# ==============================
# PREPROCESSING
# ==============================

log.info("Fitting imputer on train...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

# ==============================
# TRAIN
# ==============================

log.info("Training Random Forest (n_estimators=%d, max_depth=%d)...",
         RF_PARAMS["n_estimators"], RF_PARAMS["max_depth"])
t0    = time.time()
model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_train, y_train)
train_time = time.time() - t0
log.info("Done in %.1f s", train_time)

# ==============================
# EVALUATE
# ==============================

log.info("=" * 60)
log.info("METRICS")
log.info("=" * 60)

y_val_prob  = model.predict_proba(X_val)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]

val_aucpr   = average_precision_score(y_val,  y_val_prob)
val_rocauc  = roc_auc_score(y_val,  y_val_prob)
test_aucpr  = average_precision_score(y_test, y_test_prob)
test_rocauc = roc_auc_score(y_test, y_test_prob)

log.info("Validation  AUC-PR : %.6f", val_aucpr)
log.info("Validation  ROC-AUC: %.6f", val_rocauc)
log.info("Test        AUC-PR : %.6f", test_aucpr)
log.info("Test        ROC-AUC: %.6f", test_rocauc)

# ==============================
# FEATURE IMPORTANCE — Top 15
# ==============================

log.info("=" * 60)
log.info("TOP 15 FEATURE IMPORTANCE (mean decrease in impurity)")
log.info("=" * 60)
importances = model.feature_importances_
top_idx     = np.argsort(importances)[::-1][:15]
for rank, idx in enumerate(top_idx, 1):
    log.info("  %2d. %-30s  importance=%.6f", rank, FEATURE_COLS[idx], importances[idx])

# ==============================
# SAVE
# ==============================

joblib.dump(model,   RESULTS_DIR / f"{MODEL_NAME}_{RUN_ID}.joblib")
joblib.dump(imputer, RESULTS_DIR / f"{MODEL_NAME}_imputer_{RUN_ID}.joblib")
log.info("Artifacts saved to: %s", RESULTS_DIR)

append_comparison({
    "run_id":        RUN_ID,
    "model":         MODEL_NAME,
    "train_samples": len(y_train),
    "val_aucpr":     round(val_aucpr, 6),
    "val_rocauc":    round(val_rocauc, 6),
    "test_aucpr":    round(test_aucpr, 6),
    "test_rocauc":   round(test_rocauc, 6),
    "train_time_s":  round(train_time, 1),
    "notes":         f"n_est={RF_PARAMS['n_estimators']}, max_depth={RF_PARAMS['max_depth']}, min_leaf={RF_PARAMS['min_samples_leaf']}, max_samples={RF_MAX_TRAIN_SAMPLES}",
})
log.info("Results appended to comparison.csv")
log.info("Done.")