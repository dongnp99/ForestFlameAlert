"""
evaluate_local.py — Tính metrics cho XGBoost v5 từ model .json local
=====================================================================
Chạy từ Project/xgboost/models/v5/:
    python evaluate_local.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[2]))
from xgb_config import (
    DATA_PATH, FEATURE_COLS,
    TRAIN_END_DATE, VAL_END_DATE,
    compute_pathways,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

MODEL_PATH = Path(__file__).parent / "xgb_human_features_tuned_v5_pathways.json"

# ==============================
# LOAD MODEL
# ==============================

logging.info("Loading model: %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
logging.info("Best iteration: %s", getattr(model, "best_iteration", "N/A"))

# ==============================
# HELPERS
# ==============================

def load_split(date_filters):
    df = pd.read_parquet(
        DATA_PATH,
        columns=FEATURE_COLS + ["fire"],
        filters=date_filters,
        engine="pyarrow",
    )
    df["fire"] = df["fire"].astype("int8")
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    return df


def pathway_score(df):
    pw = compute_pathways(df)
    return pw["p1"].astype(int) + pw["p2"].astype(int) + pw["p3"].astype(int)


def predict(df):
    X = df[FEATURE_COLS].values
    dmat = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    return model.predict(dmat)


def eval_split(name, df):
    scores = pathway_score(df)
    y      = df["fire"].values
    prob   = predict(df)

    aucpr  = average_precision_score(y, prob)
    rocauc = roc_auc_score(y, prob)

    logging.info("=" * 55)
    logging.info("%s — OVERALL  (n=%s, fire_rate=%.5f)",
                 name, f"{len(y):,}", y.mean())
    logging.info("  AUC-PR : %.6f", aucpr)
    logging.info("  ROC-AUC: %.6f", rocauc)

    logging.info("%s — PER-GROUP", name)
    for label, mask in [("Human (score>0)", scores > 0), ("Natural (score=0)", scores == 0)]:
        y_t, y_p = y[mask], prob[mask]
        if y_t.sum() < 10:
            logging.info("  %-20s: skipped (%d positives)", label, y_t.sum())
            continue
        pr  = average_precision_score(y_t, y_p)
        roc = roc_auc_score(y_t, y_p)
        logging.info("  %-20s  n=%8d  fire=%5d  AUC-PR=%.6f  ROC-AUC=%.6f",
                     label, len(y_t), int(y_t.sum()), pr, roc)

    return {"aucpr": round(float(aucpr), 6), "rocauc": round(float(rocauc), 6)}

# ==============================
# EVALUATE VAL
# ==============================

logging.info("Loading val set (%s – %s)...", TRAIN_END_DATE, VAL_END_DATE)
val_df = load_split([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
logging.info("Val size: %s", f"{len(val_df):,}")
val_metrics = eval_split("Val ", val_df)
del val_df

# ==============================
# EVALUATE TEST
# ==============================

logging.info("Loading test set (> %s)...", VAL_END_DATE)
test_df = load_split([("date", ">", pd.Timestamp(VAL_END_DATE))])
logging.info("Test size: %s", f"{len(test_df):,}")
test_metrics = eval_split("Test", test_df)
del test_df

# ==============================
# SUMMARY
# ==============================

logging.info("=" * 55)
logging.info("SUMMARY — XGBoost v5 (pathway-aware features)")
logging.info("  Val  AUC-PR : %.6f  |  ROC-AUC: %.6f",
             val_metrics["aucpr"], val_metrics["rocauc"])
logging.info("  Test AUC-PR : %.6f  |  ROC-AUC: %.6f",
             test_metrics["aucpr"], test_metrics["rocauc"])
logging.info("=" * 55)