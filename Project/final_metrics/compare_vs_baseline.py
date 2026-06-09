"""
compare_vs_baseline.py — Chứng minh hiệu quả so với random baseline
=====================================================================
Tính giá trị tuyệt đối + đường baseline để chứng minh các metric
tuy thấp về tuyệt đối nhưng vượt xa dự đoán ngẫu nhiên.

Output:
    fig_standard_vs_baseline.png — Precision, Recall, AUC-PR, AUC-ROC, F2
    fig_domain_vs_baseline.png   — PEI, FRES, Recall@P_min

Chạy từ thư mục này:
    python compare_vs_baseline.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent

MODELS = [
    {"label": "XGBoost v5",          "file": "v5_full_metrics.txt",                "color": "#2A7DE0"},
    {"label": "ConvLSTM",            "file": "clstm_full_metrics.txt",              "color": "#E05C2A"},
    {"label": "Random Forest",       "file": "rf_full_metrics_20260507_210603.txt", "color": "#27AE60"},
    {"label": "Logistic Regression", "file": "lr_full_metrics_20260507_220716.txt", "color": "#8E44AD"},
    {"label": "Ensemble ConvLSTM, XGBoost - OR Rule", "file": "best_ensemble_full_metrics.txt",       "color": "#FE6D86"},
]

K_PCT      = 0.05
FRES_ALPHA = 0.7


# ================================================================
# SECTION 1 — PARSE
# ================================================================

def _extract(pattern, text, cast=float, default=0.0):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def parse(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    tp = _extract(r"TP=([\d,]+)", t, lambda x: int(x.replace(",", "")))
    fp = _extract(r"FP=([\d,]+)", t, lambda x: int(x.replace(",", "")))
    fn = _extract(r"FN=([\d,]+)", t, lambda x: int(x.replace(",", "")))
    tn = _extract(r"TN=([\d,]+)", t, lambda x: int(x.replace(",", "")))
    n_total = tp + fp + fn + tn
    n_fire  = tp + fn
    return {
        "auc_pr":          _extract(r"AUC-PR\s+=\s+([\d.]+)", t),
        "auc_roc":         _extract(r"AUC-ROC\s+=\s+([\d.]+)", t),
        "precision":       _extract(r"Precision\s+=\s+([\d.]+)", t),
        "recall":          _extract(r"Recall\s+=\s+([\d.]+)", t),
        "f2":              _extract(r"F2\s+=\s+([\d.]+)", t),
        "recall_at_k":     _extract(r"recall_at_k=([\d.]+)", t),
        "recall_at_pmin":  _extract(r"Recall@P_min\(5%\)\s+=\s+([\d.]+)", t),
        "pei":             _extract(r"PEI@5%\s+=\s+([\d.]+)", t),
        "fres":            _extract(r"FRES@5%\s+=\s+([\d.]+)", t),
        "fire_rate":       n_fire / n_total if n_total > 0 else 0,
        "n_fire":          n_fire,
        "n_total":         n_total,
    }


records = [{"label": m["label"], "color": m["color"], **parse(HERE / m["file"])}
           for m in MODELS]

# Dùng fire rate trung bình (RF và XGB dùng cùng tập test)
fire_rate = np.mean([r["fire_rate"] for r in records])

#  Baseline values cho từng metric 
# Random classifier: predict 1 với xác suất = fire_rate
#   → Precision  ≈ fire_rate (ở mọi threshold)
#   → AUC-PR     ≈ fire_rate (area under random PR curve)
#   → Recall@K%  ≈ K%        (bắt được K% fires trong K% diện tích ngẫu nhiên)
#   → PEI@K%     = 1.0       (recall_at_K / K = K/K = 1)
#   → FRES@K%    = α×K + (1-α)×fire_rate
#   → AUC-ROC    = 0.5
# F2 baseline: random classifier tại recall=1, precision=fire_rate
_b2 = 4.0
f2_baseline = (1 + _b2) * fire_rate * 1.0 / (_b2 * fire_rate + 1.0 + 1e-9)

BASELINES = {
    "AUC-PR":         fire_rate,
    "AUC-ROC":        0.5,
    "Precision":      fire_rate,
    "Recall":         fire_rate,   # tham chiếu: class prevalence
    "F2":             f2_baseline,
    "PEI@5%":         1.0,
    "FRES@5%":        FRES_ALPHA * K_PCT + (1 - FRES_ALPHA) * fire_rate,
    "Recall@P_min":   0.0,         # random không thể đạt P_min=5%
}

print(f"\nFire rate trên test set: {fire_rate:.5f} ({fire_rate:.3%})")
print(f"F2 baseline (random):    {f2_baseline:.6f}")
for k, v in BASELINES.items():
    print(f"  {k:<18} baseline = {v:.6f}")
print()

x_ticks = ["XGB v5", "CLSTM", "RF", "LR", "Ensemble"]


# ================================================================
# HELPER — vẽ một subplot có baseline
# ================================================================

def _draw_subplot(ax, vals, colors, baseline, title, x_labels,
                  baseline_label, show_lift=True, baseline_visible=True):
    x     = np.arange(len(vals))
    y_max = max(vals) * 1.40 if max(vals) > 0 else 0.01

    if baseline_visible and baseline > 0:
        ax.axhspan(0, baseline, color="red", alpha=0.07, zorder=0)
        ax.axhline(y=baseline, color="red", ls="--", lw=1.6, zorder=1,
                   label=baseline_label)

    bars = ax.bar(x, vals, color=colors, width=0.52,
                  edgecolor="white", linewidth=0.8, zorder=2)

    for bar, val in zip(bars, vals):
        h = bar.get_height()
        if show_lift and baseline > 0:
            lift = val / baseline
            ann  = f"{val:.4f}\n({lift:.0f}×)" if val < 10 else f"{val:.2f}\n({lift:.1f}×)"
        else:
            ann = f"{val:.4f}" if val < 10 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.02,
                ann, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    best = int(np.argmax(vals))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(2.5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if baseline_visible and baseline > 0:
        ax.legend(fontsize=7.5, loc="upper right")


# ================================================================
# SECTION 2 — FIGURE 1: STANDARD METRICS
# ================================================================

STANDARD = [
    ("AUC-PR",   "auc_pr",    "AUC-PR",
     f"Random = {fire_rate:.5f}\n(= fire rate)"),
    ("AUC-ROC",  "auc_roc",   "AUC-ROC",
     "Random = 0.500"),
    ("Precision","precision", "Precision\n(tại threshold F2-optimal)",
     f"Random = {fire_rate:.5f}\n(= fire rate)"),
    ("Recall",   "recall",    "Recall\n(tại threshold F2-optimal)",
     f"Class prevalence = {fire_rate:.5f}"),
    ("F2",       "f2",        "F2-score  (β=2)",
     f"Random = {f2_baseline:.5f}"),
]

colors_all = [r["color"] for r in records]

fig1, axes1 = plt.subplots(1, 5, figsize=(20, 5))
fig1.suptitle(
    f"Standard Metrics vs Random Baseline  (Test set, fire rate = {fire_rate:.3%})",
    fontsize=13, fontweight="bold"
)

for ax, (metric_name, key, title, blabel) in zip(axes1, STANDARD):
    baseline = BASELINES[metric_name]
    vals     = [r[key] for r in records]
    # Recall: baseline quá nhỏ so với giá trị thực → không tô vùng, chỉ hiển thị line
    visible  = metric_name != "Recall"
    _draw_subplot(ax, vals, colors_all, baseline, title,
                  x_ticks, blabel, show_lift=(metric_name != "Recall"),
                  baseline_visible=visible)
    # Recall: vẽ line riêng không tô vùng
    if metric_name == "Recall":
        ax.axhline(y=baseline, color="gray", ls=":", lw=1.2,
                   label=blabel)
        ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
out1 = HERE / "fig_standard_vs_baseline.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")


# ================================================================
# SECTION 3 — FIGURE 2: DOMAIN METRICS
# ================================================================

DOMAIN = [
    ("PEI@5%",      "pei",           "PEI @ K=5%\n(Patrol Efficiency Index)",
     "Random = 1.00\n(tuần tra ngẫu nhiên)"),
    ("FRES@5%",     "fres",          f"FRES @ K=5%\n(α={FRES_ALPHA}×Recall + {round(1-FRES_ALPHA, 2)}×Prec)",
     f"Random = {FRES_ALPHA*K_PCT + (1-FRES_ALPHA)*fire_rate:.5f}"),
    ("Recall@P_min","recall_at_pmin","Recall@P_min (5%)\n(Recall khi Precision ≥ 5%)",
     "Random = 0\n(không thể đạt P_min=5%)"),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle(
    f"Domain Metrics vs Random Baseline  (Test set, K=5%, fire rate = {fire_rate:.3%})",
    fontsize=13, fontweight="bold"
)

for ax, (metric_name, key, title, blabel) in zip(axes2, DOMAIN):
    baseline = BASELINES[metric_name]
    vals     = [r[key] for r in records]
    # Recall@P_min: baseline=0, không tô vùng
    visible  = metric_name != "Recall@P_min"
    _draw_subplot(ax, vals, colors_all, baseline, title,
                  x_ticks, blabel, show_lift=(metric_name != "Recall@P_min"),
                  baseline_visible=visible)
    if metric_name == "Recall@P_min":
        ax.text(0.02, 0.97, "Random không thể đạt P_min=5%\n(fire rate = 0.12%)",
                transform=ax.transAxes, fontsize=7, color="gray",
                va="top", ha="left")

plt.tight_layout()
out2 = HERE / "fig_domain_vs_baseline.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

