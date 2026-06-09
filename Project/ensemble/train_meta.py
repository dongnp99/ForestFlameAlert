"""
train_meta.py — Thử nghiệm các chiến lược Ensemble
===================================================
Mục đích: so sánh các cách kết hợp XGBoost v5 + ConvLSTM,
chọn ra ensemble tốt nhất, lưu kết quả để so sánh với 4 mô hình đơn.

Chiến lược:
  1. XGBoost v5     (baseline đơn)
  2. ConvLSTM       (baseline đơn)
  3. OR Rule        — max(xgb, clstm)
  4. Weighted Avg   — w*xgb + (1-w)*clstm, tối ưu w trên val
  5. LR Basic       — LogisticRegression([xgb_prob, clstm_prob])
  6. LR + Context   — LogisticRegression([xgb_prob, clstm_prob, lon, lat, sin_doy, cos_doy])

Threshold: F2-optimal trên val set (nhất quán với 4 script đơn lẻ).

Output:
  ensemble_results.csv            — kết quả tất cả chiến lược
  best_ensemble_full_metrics.txt  — metrics đầy đủ của ensemble tốt nhất
                                    (cùng format với *_full_metrics.txt của 4 mô hình đơn)
  meta_lr_basic.pkl
  meta_lr_context.pkl

Chạy từ thư mục này:
    python train_meta.py
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, confusion_matrix,
)

#  Paths ─
HERE = Path(__file__).parent

VAL_PAR  = HERE / "ensemble_val.parquet"
TEST_PAR = HERE / "ensemble_test.parquet"

OUT_LR_BASIC   = HERE / "meta_lr_basic.pkl"
OUT_LR_CONTEXT = HERE / "meta_lr_context.pkl"
OUT_CSV        = HERE / "ensemble_results.csv"
OUT_BEST_TXT   = HERE / "best_ensemble_full_metrics.txt"

#  Cấu hình domain metrics ─
P_MIN      = 0.05
K_PCT      = 0.05
FRES_ALPHA = 0.7


# ================================================================
# SECTION 1 — METRIC FUNCTIONS
# ================================================================

def _best_threshold_f2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F2-optimal threshold trên tập đang xét (dùng trên val set)."""
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    name: str, threshold: float) -> dict:
    """Tính đầy đủ metrics với threshold đã cho (F2-optimal từ val set)."""
    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    y_bin = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    beta2     = 4.0
    f2        = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)
    csi       = tp / (tp + fp + fn + 1e-8)

    rap  = _recall_at_pmin(y_true, y_prob)
    pei  = _pei(y_true, y_prob)
    fres = _fres(y_true, y_prob)

    return {
        "name":      name,
        "threshold": threshold,
        "auc_pr":    float(auc_pr),
        "auc_roc":   float(auc_roc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "f2":        float(f2),
        "csi":       float(csi),
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


def print_metrics(m: dict):
    pmin_th = f"{m['pmin_threshold']:.4f}" if m["pmin_threshold"] is not None else "N/A"
    print(f"    AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}  "
          f"Recall={m['recall']:.4f}  Prec={m['precision']:.4f}  "
          f"F1={m['f1']:.4f}  F2={m['f2']:.4f}  thresh={m['threshold']:.4f}")
    print(f"    TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}")
    print(f"    F2={m['f2']:.4f}  PEI@{K_PCT:.0%}={m['pei']:.2f}  "
          f"FRES@{K_PCT:.0%}={m['fres']:.4f}  "
          f"Recall@P{P_MIN:.0%}={m['recall_at_pmin']:.4f} "
          f"(prec={m['pmin_precision']:.4f} thresh={pmin_th})")


# ================================================================
# SECTION 2 — LOAD DATA
# ================================================================

def load_splits():
    print("Load dữ liệu...")
    val  = pd.read_parquet(VAL_PAR,  engine="pyarrow")
    test = pd.read_parquet(TEST_PAR, engine="pyarrow")
    print(f"  Val : {len(val):,} rows  fire={val['fire'].mean():.4%}")
    print(f"  Test: {len(test):,} rows  fire={test['fire'].mean():.4%}")
    return val, test


# ================================================================
# SECTION 3 — ENSEMBLE STRATEGIES
# ================================================================

def strategy_or_rule(xgb: np.ndarray, clstm: np.ndarray) -> np.ndarray:
    return np.maximum(xgb, clstm)


def strategy_weighted_avg(
    xgb_val: np.ndarray, clstm_val: np.ndarray, y_val: np.ndarray,
    xgb_test: np.ndarray, clstm_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Grid search w trên val → predict test."""
    print("  Grid search w trên val set...")
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        prob = w * xgb_val + (1 - w) * clstm_val
        auc  = average_precision_score(y_val, prob)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  Best w={best_w:.2f}  (val AUC-PR={best_auc:.4f})")
    return best_w * xgb_test + (1 - best_w) * clstm_test, best_w


def strategy_lr(
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray,
    model_name: str,
    out_path: Path,
) -> np.ndarray:
    """Train LogisticRegression trên val → predict test."""
    print(f"  Train {model_name}...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            C=1.0, max_iter=500,
            solver="lbfgs", random_state=42,
        )),
    ])
    pipe.fit(X_val, y_val)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved → {out_path}")
    return pipe.predict_proba(X_test)[:, 1].astype(np.float32)


# ================================================================
# SECTION 4 — SUMMARY & OUTPUT
# ================================================================

def print_summary(results: list[dict]):
    w = 14
    cols  = ["auc_pr", "auc_roc", "precision", "recall", "f1", "f2",
             "pei", "fres", "recall_at_pmin"]
    heads = ["AUC-PR", "AUC-ROC", "Prec", "Recall", "F1", "F2",
             f"PEI@{K_PCT:.0%}", f"FRES@{K_PCT:.0%}", f"R@P{P_MIN:.0%}"]

    print("\n" + "=" * (20 + w * len(cols)))
    print("BẢNG KẾT QUẢ ENSEMBLE (Test set)")
    print("=" * (20 + w * len(cols)))
    print(f"  {'Strategy':<20}" + "".join(f"{h:>{w}}" for h in heads))
    print("  " + "-" * (20 + w * len(cols)))
    for r in results:
        row = f"  {r['name']:<20}"
        for k in cols:
            row += f"{r[k]:>{w}.4f}"
        print(row)
    print("=" * (20 + w * len(cols)))

    print("\n  Best per metric:")
    for k, h in zip(cols, heads):
        best = max(results, key=lambda r, k=k: r[k])
        print(f"    {h:<14}: {best['name']}  ({best[k]:.4f})")


def select_best(results: list[dict]) -> dict:
    """Chọn ensemble tốt nhất: ưu tiên Recall@P_min, fallback AUC-PR."""
    # Chỉ xét các chiến lược ensemble thực sự (không phải model đơn)
    ensembles = [r for r in results if r["name"] not in ("XGBoost v5", "ConvLSTM")]

    achievable = [r for r in ensembles if r["pmin_achievable"]]
    if achievable:
        best = max(achievable, key=lambda r: r["recall_at_pmin"])
        print(f"\n  >>> BEST ENSEMBLE (Recall@P{P_MIN:.0%}): {best['name']}  "
              f"(Recall={best['recall_at_pmin']:.4f}, "
              f"Prec={best['pmin_precision']:.4f})")
    else:
        best = max(ensembles, key=lambda r: r["auc_pr"])
        print(f"\n  >>> BEST ENSEMBLE (fallback AUC-PR): {best['name']}  "
              f"(AUC-PR={best['auc_pr']:.4f})")
    return best


def save_best_metrics(best: dict):
    """Lưu metrics của ensemble tốt nhất theo format chuẩn *_full_metrics.txt."""
    pmin_thresh = best["pmin_threshold"] if best["pmin_threshold"] is not None else "N/A"
    lines = [
        f"{best['name']} — Full Evaluation",
        "=" * 60,
        f"Strategy: {best['name']}",
        f"Threshold (F2-optimal on val): {best['threshold']:.4f}",
        "",
        "[Standard]",
        f"  AUC-PR    = {best['auc_pr']:.4f}",
        f"  AUC-ROC   = {best['auc_roc']:.4f}",
        f"  Precision = {best['precision']:.4f}",
        f"  Recall    = {best['recall']:.4f}",
        f"  F1        = {best['f1']:.4f}",
        f"  F2        = {best['f2']:.4f}",
        f"  TP={best['tp']:,}  FP={best['fp']:,}  FN={best['fn']:,}  TN={best['tn']:,}",
        "",
        f"[Domain — K={K_PCT:.0%}, alpha={FRES_ALPHA}]",
        f"  Recall@P_min({P_MIN:.0%}) = {best['recall_at_pmin']:.4f}  "
        f"(prec={best['pmin_precision']:.4f}, thresh={pmin_thresh})",
        f"  PEI@{K_PCT:.0%}           = {best['pei']:.2f}  "
        f"(recall_at_k={best['pei_recall_at_k']:.4f}, "
        f"fires={best['pei_fires_caught']}/{best['pei_fires_total']})",
        f"  FRES@{K_PCT:.0%}          = {best['fres']:.4f}  "
        f"(recall={best['fres_recall_at_k']:.4f}, "
        f"prec={best['fres_precision_at_k']:.4f})",
    ]
    OUT_BEST_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {OUT_BEST_TXT}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE COMPARISON")
    print(f"P_min={P_MIN:.0%}  K={K_PCT:.0%}  alpha={FRES_ALPHA}")
    print("=" * 60)

    val_df, test_df = load_splits()

    y_val  = val_df["fire"].to_numpy(np.int8)
    y_test = test_df["fire"].to_numpy(np.int8)

    xgb_val    = val_df["xgb_prob"].to_numpy(np.float32)
    clstm_val  = val_df["clstm_prob"].to_numpy(np.float32)
    xgb_test   = test_df["xgb_prob"].to_numpy(np.float32)
    clstm_test = test_df["clstm_prob"].to_numpy(np.float32)

    ctx_cols   = ["lon", "lat", "sin_doy", "cos_doy"]
    X_val_ctx  = val_df[ctx_cols].to_numpy(np.float32)
    X_test_ctx = test_df[ctx_cols].to_numpy(np.float32)

    results = []

    #  1. Baselines đơn ─
    print("\n[1/2] Single model baselines...")
    for name, y_val_prob, y_test_prob in [
        ("XGBoost v5", xgb_val,   xgb_test),
        ("ConvLSTM",   clstm_val, clstm_test),
    ]:
        thresh = _best_threshold_f2(y_val, y_val_prob)
        m = compute_metrics(y_test, y_test_prob, name, thresh)
        print(f"\n  {name}:")
        print_metrics(m)
        results.append(m)

    #  2. OR Rule 
    print("\n[3] OR Rule...")
    or_val  = strategy_or_rule(xgb_val,  clstm_val)
    or_test = strategy_or_rule(xgb_test, clstm_test)
    thresh  = _best_threshold_f2(y_val, or_val)
    m_or    = compute_metrics(y_test, or_test, "OR Rule", thresh)
    print_metrics(m_or)
    results.append(m_or)

    #  3. Weighted Average ─
    print("\n[4] Weighted Average...")
    wa_test, best_w = strategy_weighted_avg(
        xgb_val, clstm_val, y_val, xgb_test, clstm_test,
    )
    wa_val = best_w * xgb_val + (1 - best_w) * clstm_val
    thresh = _best_threshold_f2(y_val, wa_val)
    m_wa   = compute_metrics(y_test, wa_test, "Weighted Avg", thresh)
    m_wa["best_w"] = best_w
    print_metrics(m_wa)
    results.append(m_wa)

    #  4. LR Basic ─
    print("\n[5] LR Basic...")
    X_val_basic  = np.column_stack([xgb_val,  clstm_val])
    X_test_basic = np.column_stack([xgb_test, clstm_test])
    lr_basic_test = strategy_lr(
        X_val_basic, y_val, X_test_basic, "LR Basic", OUT_LR_BASIC,
    )
    # LR dùng val để train → threshold tìm trên chính val predictions của LR
    lr_basic_val = pickle.load(open(OUT_LR_BASIC, "rb")).predict_proba(X_val_basic)[:, 1]
    thresh = _best_threshold_f2(y_val, lr_basic_val)
    m_lr_basic = compute_metrics(y_test, lr_basic_test, "LR Basic", thresh)
    print_metrics(m_lr_basic)
    results.append(m_lr_basic)

    #  5. LR + Context 
    print("\n[6] LR + Context...")
    X_val_ctx_full  = np.column_stack([xgb_val,  clstm_val,  X_val_ctx])
    X_test_ctx_full = np.column_stack([xgb_test, clstm_test, X_test_ctx])
    lr_ctx_test = strategy_lr(
        X_val_ctx_full, y_val, X_test_ctx_full, "LR + Context", OUT_LR_CONTEXT,
    )
    lr_ctx_val = pickle.load(open(OUT_LR_CONTEXT, "rb")).predict_proba(X_val_ctx_full)[:, 1]
    thresh = _best_threshold_f2(y_val, lr_ctx_val)
    m_lr_ctx = compute_metrics(y_test, lr_ctx_test, "LR + Context", thresh)
    print_metrics(m_lr_ctx)
    results.append(m_lr_ctx)

    #  Summary ─
    print_summary(results)

    #  Chọn best ensemble 
    best = select_best(results)

    #  Lưu CSV đầy đủ ─
    df = pd.DataFrame([
        {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        for r in results
    ])
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\n  Saved → {OUT_CSV}")

    #  Lưu file metrics của best ensemble ─
    save_best_metrics(best)

    print("\n" + "=" * 60)
    print("HOÀN TẤT")
    print(f"  {OUT_CSV.name}")
    print(f"  {OUT_BEST_TXT.name}")
    print("=" * 60)
