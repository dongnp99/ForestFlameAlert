"""
train_logistic_regression.py
Logistic Regression baseline trên dataset v3_pathways.

Chạy từ Project/baseline/:
    python train_logistic_regression.py
"""

import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline_config import (
    FEATURE_COLS, RESULTS_DIR,
    TRAIN_END_DATE, VAL_END_DATE,
    RANDOM_STATE, LR_MAX_TRAIN_SAMPLES,
    load_split, setup_logging, append_comparison,
)

# ==============================
# HYPERPARAMETERS
# ==============================

LR_PARAMS = {
    "C":            1.0,
    "solver":       "saga",        # nhanh nhất cho large n, hỗ trợ L1/L2
    "penalty":      "l2",
    "max_iter":     4000,          # tăng từ 1000: lần trước chưa converge
    "tol":          1e-3,          # nới lỏng từ 1e-4: đủ chính xác cho baseline
    "class_weight": "balanced",    # xử lý class imbalance tự động
    "n_jobs":       -1,
    "random_state": RANDOM_STATE,
}

# ==============================
# SETUP
# ==============================

RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = "logistic_regression"
log        = setup_logging(MODEL_NAME, RUN_ID)

log.info("=" * 60)
log.info("Logistic Regression Baseline")
log.info("=" * 60)
log.info("Params          : %s", LR_PARAMS)
log.info("Max train samples: %s", f"{LR_MAX_TRAIN_SAMPLES:,}")
log.info("Num features    : %d", len(FEATURE_COLS))

# ==============================
# LOAD DATA
# ==============================

log.info("Loading train set (≤ %s)...", TRAIN_END_DATE)
train_df = load_split(
    [("date", "<=", pd.Timestamp(TRAIN_END_DATE))],
    max_samples=LR_MAX_TRAIN_SAMPLES,
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

log.info("Fitting imputer + scaler on train...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ==============================
# TRAIN
# ==============================

log.info("Training Logistic Regression...")
t0    = time.time()
model = LogisticRegression(**LR_PARAMS)
model.fit(X_train, y_train)
train_time = time.time() - t0
log.info("Done in %.1f s  |  iterations: %d", train_time, model.n_iter_[0])

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
# TOP FEATURES BY |COEFFICIENT|
# ==============================

log.info("=" * 60)
log.info("TOP 15 FEATURES BY |COEFFICIENT|")
log.info("=" * 60)
coef    = np.abs(model.coef_[0])
top_idx = np.argsort(coef)[::-1][:15]
for rank, idx in enumerate(top_idx, 1):
    log.info("  %2d. %-30s  |coef|=%.6f", rank, FEATURE_COLS[idx], coef[idx])

# ==============================
# SAVE
# ==============================

joblib.dump(model,   RESULTS_DIR / f"{MODEL_NAME}_{RUN_ID}.joblib")
joblib.dump(imputer, RESULTS_DIR / f"{MODEL_NAME}_imputer_{RUN_ID}.joblib")
joblib.dump(scaler,  RESULTS_DIR / f"{MODEL_NAME}_scaler_{RUN_ID}.joblib")
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
    "notes":         f"C={LR_PARAMS['C']}, solver={LR_PARAMS['solver']}, penalty={LR_PARAMS['penalty']}, max_samples={LR_MAX_TRAIN_SAMPLES}",
})
log.info("Results appended to comparison.csv")
log.info("Done.")