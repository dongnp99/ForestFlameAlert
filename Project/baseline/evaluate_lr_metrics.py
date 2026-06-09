"""
evaluate_lr_metrics.py — Đánh giá đầy đủ Logistic Regression baseline
======================================================================
Tính: Precision, Recall, F2, AUC-PR, Recall@P_min, FRES, PEI
Threshold tối ưu tìm trên val set theo F2, sau đó áp dụng trên test set.

Tự động dùng model .joblib mới nhất trong results/.
Pipeline preprocessing: imputer (median) → scaler (StandardScaler).

Chạy từ thư mục này:
    python evaluate_lr_metrics.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)

from baseline_config import (
    FEATURE_COLS, RESULTS_DIR,
    TRAIN_END_DATE, VAL_END_DATE,
    load_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

#  Cấu hình domain metrics ─
P_MIN      = 0.05
K_PCT      = 0.05
FRES_ALPHA = 0.7


# ================================================================
# SECTION 1 — METRIC FUNCTIONS (giống RF)
# ================================================================

def _best_threshold_f2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    beta2 = 4.0
    f2 = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-9)
    best_idx = int(np.argmax(f2[:-1]))
    return float(thresholds[best_idx])


def _recall_at_pmin(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    valid = prec >= P_MIN
    if not valid.any():
        return {"recall_at_pmin": 0.0, "precision": 0.0, "threshold": None, "achievable": False}
    idx = int(np.argmax(rec[valid]))
    valid_idx = np.where(valid)[0][idx]
    return {
        "recall_at_pmin": float(rec[valid][idx]),
        "precision":      float(prec[valid][idx]),
        "threshold":      float(thresholds[min(valid_idx, len(thresholds) - 1)]),
        "achievable":     True,
    }


def _pei(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    n = len(y_true)
    k = max(1, int(n * K_PCT))
    top_k = np.argsort(y_prob)[::-1][:k]
    total_fires = float(y_true.sum())
    caught = float(y_true[top_k].sum())
    recall_at_k = caught / total_fires if total_fires > 0 else 0.0
    return {
        "pei":          recall_at_k / K_PCT if K_PCT > 0 else 0.0,
        "recall_at_k":  recall_at_k,
        "fires_caught": int(caught),
        "fires_total":  int(total_fires),
    }


def _fres(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    n = len(y_true)
    k = max(1, int(n * K_PCT))
    top_k = np.argsort(y_prob)[::-1][:k]
    total_fires = float(y_true.sum())
    caught = float(y_true[top_k].sum())
    recall_at_k    = caught / total_fires if total_fires > 0 else 0.0
    precision_at_k = caught / k
    return {
        "fres":           FRES_ALPHA * recall_at_k + (1 - FRES_ALPHA) * precision_at_k,
        "recall_at_k":    recall_at_k,
        "precision_at_k": precision_at_k,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    y_bin = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    beta2     = 4.0
    f2        = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)

    rap  = _recall_at_pmin(y_true, y_prob)
    pei  = _pei(y_true, y_prob)
    fres = _fres(y_true, y_prob)

    return {
        "threshold": threshold,
        "auc_pr":    auc_pr,
        "auc_roc":   auc_roc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "f2":        f2,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall_at_pmin":      rap["recall_at_pmin"],
        "pmin_precision":      rap["precision"],
        "pmin_threshold":      rap["threshold"],
        "pmin_achievable":     rap["achievable"],
        "pei":                 pei["pei"],
        "pei_recall_at_k":     pei["recall_at_k"],
        "pei_fires_caught":    pei["fires_caught"],
        "pei_fires_total":     pei["fires_total"],
        "fres":                fres["fres"],
        "fres_recall_at_k":    fres["recall_at_k"],
        "fres_precision_at_k": fres["precision_at_k"],
    }


def print_metrics(m: dict, model_name: str):
    logging.info("=" * 60)
    logging.info("TEST METRICS — %s", model_name)
    logging.info("  Threshold (F2-optimal on val): %.4f", m["threshold"])
    logging.info("")
    logging.info("  [Standard]")
    logging.info("    AUC-PR    = %.4f", m["auc_pr"])
    logging.info("    AUC-ROC   = %.4f", m["auc_roc"])
    logging.info("    Precision = %.4f", m["precision"])
    logging.info("    Recall    = %.4f", m["recall"])
    logging.info("    F1        = %.4f", m["f1"])
    logging.info("    F2        = %.4f", m["f2"])
    logging.info("    TP=%d  FP=%d  FN=%d  TN=%d",
                 m["tp"], m["fp"], m["fn"], m["tn"])
    logging.info("")
    logging.info("  [Domain — K=%.0f%%]", K_PCT * 100)
    if m["pmin_achievable"]:
        logging.info("    Recall@P_min(%.0f%%) = %.4f  (prec=%.4f, thresh=%.4f)",
                     P_MIN * 100, m["recall_at_pmin"], m["pmin_precision"], m["pmin_threshold"])
    else:
        logging.info("    Recall@P_min(%.0f%%) = N/A (precision %.0f%% không đạt được)",
                     P_MIN * 100, P_MIN * 100)
    logging.info("    PEI@%.0f%%  = %.2f  (recall_at_k=%.4f, fires=%d/%d)",
                 K_PCT * 100, m["pei"], m["pei_recall_at_k"],
                 m["pei_fires_caught"], m["pei_fires_total"])
    logging.info("    FRES@%.0f%% = %.4f  (recall=%.4f, prec=%.4f, alpha=%.1f)",
                 K_PCT * 100, m["fres"], m["fres_recall_at_k"],
                 m["fres_precision_at_k"], FRES_ALPHA)


# ================================================================
# SECTION 2 — LOAD MODEL (auto-detect latest)
# ================================================================

def _find_latest_model() -> tuple[Path, Path, Path]:
    """Tìm bộ (model, imputer, scaler) .joblib mới nhất trong results/."""
    # Lọc ra model chính (không phải imputer/scaler)
    candidates = sorted([
        p for p in RESULTS_DIR.glob("logistic_regression_[0-9]*.joblib")
        if "imputer" not in p.stem and "scaler" not in p.stem
    ])
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy logistic_regression_*.joblib trong {RESULTS_DIR}")
    model_path = candidates[-1]
    run_id     = model_path.stem.replace("logistic_regression_", "")

    imputer_path = RESULTS_DIR / f"logistic_regression_imputer_{run_id}.joblib"
    scaler_path  = RESULTS_DIR / f"logistic_regression_scaler_{run_id}.joblib"

    for p in (imputer_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy artifact: {p}")

    return model_path, imputer_path, scaler_path


model_path, imputer_path, scaler_path = _find_latest_model()
logging.info("Loading model  : %s", model_path.name)
logging.info("Loading imputer: %s", imputer_path.name)
logging.info("Loading scaler : %s", scaler_path.name)

model   = joblib.load(model_path)
imputer = joblib.load(imputer_path)
scaler  = joblib.load(scaler_path)


# ================================================================
# SECTION 3 — LOAD DATA & PREDICT
# ================================================================

def predict_split(date_filters: list) -> tuple[np.ndarray, np.ndarray]:
    """Load split, áp imputer → scaler, trả về (y_true, y_prob)."""
    df     = load_split(date_filters)
    y_true = df.pop("fire").values if "fire" in df.columns else None
    X      = scaler.transform(imputer.transform(df[FEATURE_COLS].values))
    y_prob = model.predict_proba(X)[:, 1]
    return y_true, y_prob


#  Val set: tìm threshold tối ưu 
logging.info("Loading val set (%s – %s)...", TRAIN_END_DATE, VAL_END_DATE)
y_val, prob_val = predict_split([
    ("date", ">",  pd.Timestamp(TRAIN_END_DATE)),
    ("date", "<=", pd.Timestamp(VAL_END_DATE)),
])
logging.info("Val size: %s | fire_rate=%.5f", f"{len(y_val):,}", y_val.mean())

best_threshold = _best_threshold_f2(y_val, prob_val)
logging.info("Best threshold (F2 on val): %.4f", best_threshold)
del y_val, prob_val

#  Test set: tính metrics 
logging.info("Loading test set (> %s)...", VAL_END_DATE)
y_test, prob_test = predict_split([("date", ">", pd.Timestamp(VAL_END_DATE))])
logging.info("Test size: %s | fire_rate=%.5f", f"{len(y_test):,}", y_test.mean())

metrics = compute_metrics(y_test, prob_test, best_threshold)
print_metrics(metrics, model_path.stem)


# ================================================================
# SECTION 4 — PR CURVE PLOT
# ================================================================

prec_curve, rec_curve, _ = precision_recall_curve(y_test, prob_test)
fire_rate = y_test.mean()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec_curve, prec_curve, color="#8E44AD", lw=2,
        label=f"Logistic Regression (AUC-PR={metrics['auc_pr']:.4f})")
ax.axhline(y=P_MIN, color="gray", ls="--", lw=1,
           label=f"P_min = {P_MIN:.0%}")
ax.axhline(y=fire_rate, color="#E05C2A", ls=":", lw=1,
           label=f"Baseline (fire rate = {fire_rate:.4%})")
ax.scatter(metrics["recall"], metrics["precision"], color="red", zorder=5,
           label=f"Threshold={metrics['threshold']:.3f}\n(F2={metrics['f2']:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR Curve — Logistic Regression (Test set)\n{model_path.name}")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, max(prec_curve.max() * 1.1, P_MIN * 2))
plt.tight_layout()

run_id  = model_path.stem.replace("logistic_regression_", "")
out_fig = RESULTS_DIR / f"lr_pr_curve_{run_id}.png"
fig.savefig(out_fig, dpi=150)
logging.info("PR curve saved: %s", out_fig)
plt.close(fig)


# ================================================================
# SECTION 5 — LƯU KẾT QUẢ
# ================================================================

out_txt = RESULTS_DIR / f"lr_full_metrics_{run_id}.txt"
lines = [
    "Logistic Regression — Full Evaluation",
    "=" * 60,
    f"Model: {model_path.name}",
    f"Threshold (F2-optimal on val): {metrics['threshold']:.4f}",
    "",
    "[Standard]",
    f"  AUC-PR    = {metrics['auc_pr']:.4f}",
    f"  AUC-ROC   = {metrics['auc_roc']:.4f}",
    f"  Precision = {metrics['precision']:.4f}",
    f"  Recall    = {metrics['recall']:.4f}",
    f"  F1        = {metrics['f1']:.4f}",
    f"  F2        = {metrics['f2']:.4f}",
    f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}  TN={metrics['tn']:,}",
    "",
    f"[Domain — K={K_PCT:.0%}, alpha={FRES_ALPHA}]",
    f"  Recall@P_min({P_MIN:.0%}) = {metrics['recall_at_pmin']:.4f}  "
    f"(prec={metrics['pmin_precision']:.4f}, thresh={metrics['pmin_threshold']})",
    f"  PEI@{K_PCT:.0%}           = {metrics['pei']:.2f}  "
    f"(recall_at_k={metrics['pei_recall_at_k']:.4f}, fires={metrics['pei_fires_caught']}/{metrics['pei_fires_total']})",
    f"  FRES@{K_PCT:.0%}          = {metrics['fres']:.4f}  "
    f"(recall={metrics['fres_recall_at_k']:.4f}, prec={metrics['fres_precision_at_k']:.4f})",
]
out_txt.write_text("\n".join(lines), encoding="utf-8")
logging.info("Metrics saved: %s", out_txt)
