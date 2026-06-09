"""
compare_models.py — So sánh tổng hợp 4 mô hình từ *_full_metrics.txt
=====================================================================
Đọc file metrics, vẽ 3 biểu đồ so sánh, lưu bảng CSV tổng hợp.

Output:
    fig_standard_metrics.png  — AUC-PR, AUC-ROC, Precision, Recall, F1, F2
    fig_domain_metrics.png    — PEI, FRES, Recall@P_min
    fig_confusion.png         — TP, FP, FN (stacked bar)
    comparison_table.csv      — bảng tổng hợp tất cả metrics

Chạy từ thư mục này:
    python compare_models.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).parent

#  Màu sắc cố định cho từng mô hình ─
MODELS = [
    {"label": "XGBoost v5",         "file": "v5_full_metrics.txt",                      "color": "#2A7DE0"},
    {"label": "ConvLSTM",           "file": "clstm_full_metrics.txt",                   "color": "#E05C2A"},
    {"label": "Random Forest",      "file": "rf_full_metrics_20260507_210603.txt",       "color": "#27AE60"},
    {"label": "Logistic Regression","file": "lr_full_metrics_20260507_220716.txt",       "color": "#8E44AD"},
    {"label": "Ensemble ConvLSTM, XGBoost - OR Rule","file": "best_ensemble_full_metrics.txt",       "color": "#FE6D86"},
]


# ================================================================
# SECTION 1 — PARSE METRICS FILES
# ================================================================

def _extract(pattern: str, text: str, cast=float, default=0.0):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return {
        "auc_pr":         _extract(r"AUC-PR\s+=\s+([\d.]+)", text),
        "auc_roc":        _extract(r"AUC-ROC\s+=\s+([\d.]+)", text),
        "precision":      _extract(r"Precision\s+=\s+([\d.]+)", text),
        "recall":         _extract(r"Recall\s+=\s+([\d.]+)", text),
        "f1":             _extract(r"F1\s+=\s+([\d.]+)", text),
        "f2":             _extract(r"F2\s+=\s+([\d.]+)", text),
        "tp":             _extract(r"TP=([\d,]+)",  text, cast=lambda x: int(x.replace(",", ""))),
        "fp":             _extract(r"FP=([\d,]+)",  text, cast=lambda x: int(x.replace(",", ""))),
        "fn":             _extract(r"FN=([\d,]+)",  text, cast=lambda x: int(x.replace(",", ""))),
        "tn":             _extract(r"TN=([\d,]+)",  text, cast=lambda x: int(x.replace(",", ""))),
        "recall_at_pmin": _extract(r"Recall@P_min\(5%\)\s+=\s+([\d.]+)", text),
        "pei":            _extract(r"PEI@5%\s+=\s+([\d.]+)", text),
        "fres":           _extract(r"FRES@5%\s+=\s+([\d.]+)", text),
    }


records = []
for m in MODELS:
    path = HERE / m["file"]
    data = parse_metrics(path)
    data["label"] = m["label"]
    data["color"] = m["color"]
    records.append(data)

labels = [r["label"] for r in records]
colors = [r["color"] for r in records]
x      = np.arange(len(labels))


# ================================================================
# SECTION 2 — FIGURE 1: STANDARD METRICS (2×3 grid)
# ================================================================

STANDARD = [
    ("AUC-PR",    "auc_pr",    "Khả năng phân biệt tổng thể\n(cao hơn = tốt hơn)"),
    ("AUC-ROC",   "auc_roc",   "Khả năng ranking\n(cao hơn = tốt hơn)"),
    ("Precision", "precision", "Độ chính xác cảnh báo\n(cao hơn = ít false alarm hơn)"),
    ("Recall",    "recall",    "Tỉ lệ phát hiện cháy\n(cao hơn = bỏ sót ít hơn)"),
    ("F1",        "f1",        "Cân bằng Precision & Recall"),
    ("F2",        "f2",        "Ưu tiên Recall (β=2)\n(metric chính)"),
]

fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle("So sánh Standard Metrics — 5 mô hình (Test set)", fontsize=14, fontweight="bold")

for ax, (title, key, desc) in zip(axes1.flat, STANDARD):
    vals = [r[key] for r in records]
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    # Ghi giá trị trên thanh
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Highlight thanh cao nhất
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    ax.set_title(f"{title}\n{desc}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["XGB v5", "ConvLSTM", "RF", "LR", "Ensemble"], fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = HERE / "fig_standard_metrics.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")


# ================================================================
# SECTION 3 — FIGURE 2: DOMAIN METRICS (1×3 grid)
# ================================================================

DOMAIN = [
    ("Recall@P_min (5%)",
     "recall_at_pmin",
     "Recall khi Precision ≥ 5%\n(cao hơn = tốt hơn)"),
    ("PEI @ K=5%",
     "pei",
     "Hiệu suất tuần tra\nRandom patrol = 1.0"),
    ("FRES @ K=5%",
     "fres",
     "0.7×Recall@K + 0.3×Prec@K\n(metric triển khai tổng hợp)"),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle("So sánh Domain Metrics — 5 mô hình (Test set, K=5%)",
              fontsize=13, fontweight="bold")

for ax, (title, key, desc) in zip(axes2, DOMAIN):
    vals = [r[key] for r in records]
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    # Đường baseline cho PEI
    if key == "pei":
        ax.axhline(y=1.0, color="gray", ls="--", lw=1.2, label="Random (PEI=1)")
        ax.legend(fontsize=8)

    ax.set_title(f"{title}\n{desc}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["XGB v5", "ConvLSTM", "RF", "LR", "Ensemble"], fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = HERE / "fig_domain_metrics.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")


# ================================================================
# SECTION 4 — FIGURE 3: CONFUSION MATRIX (stacked bar TP/FP/FN)
# ================================================================

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Confusion Matrix Breakdown — 5 mô hình (Test set)",
              fontsize=13, fontweight="bold")

#  Subplot trái: TP và FN (tổng = tất cả fires thực) 
ax = axes3[0]
tp_vals = [r["tp"] for r in records]
fn_vals = [r["fn"] for r in records]
b1 = ax.bar(x, tp_vals, color=colors, width=0.55, label="TP (detected)", alpha=0.9)
b2 = ax.bar(x, fn_vals, bottom=tp_vals, color=colors, width=0.55,
            label="FN (missed)", alpha=0.35, hatch="//")

for i, (tp, fn) in enumerate(zip(tp_vals, fn_vals)):
    total = tp + fn
    ax.text(i, tp / 2, f"TP\n{tp:,}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(i, tp + fn / 2, f"FN\n{fn:,}", ha="center", va="center", fontsize=8, color="gray")
    ax.text(i, total + total * 0.02, f"{tp/total:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Fires thực tế được phát hiện\n(TP+FN = tổng fires, % = Recall)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["XGB v5", "ConvLSTM", "RF", "LR", "Ensemble"], fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

#  Subplot phải: TP và FP (tổng = tất cả predicted positive) 
ax = axes3[1]
fp_vals = [r["fp"] for r in records]
b1 = ax.bar(x, tp_vals, color=colors, width=0.55, label="TP (true fire)", alpha=0.9)
b2 = ax.bar(x, fp_vals, bottom=tp_vals, color=colors, width=0.55,
            label="FP (false alarm)", alpha=0.35, hatch="//")

for i, (tp, fp) in enumerate(zip(tp_vals, fp_vals)):
    total = tp + fp
    ax.text(i, tp / 2, f"TP\n{tp:,}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(i, tp + fp / 2, f"FP\n{fp:,}", ha="center", va="center", fontsize=8, color="gray")
    ax.text(i, total + total * 0.02, f"{tp/total:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Cảnh báo được phát ra\n(TP+FP = tổng predicted, % = Precision)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["XGB v5", "ConvLSTM", "RF", "LR", "Ensemble"], fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

plt.tight_layout()
out3 = HERE / "fig_confusion.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")


# ================================================================
# SECTION 5 — LƯU CSV TỔNG HỢP
# ================================================================

rows = []
for r in records:
    rows.append({
        "Model":           r["label"],
        "AUC-PR":          r["auc_pr"],
        "AUC-ROC":         r["auc_roc"],
        "Precision":       r["precision"],
        "Recall":          r["recall"],
        "F1":              r["f1"],
        "F2":              r["f2"],
        "TP":              r["tp"],
        "FP":              r["fp"],
        "FN":              r["fn"],
        "TN":              r["tn"],
        "Recall@P_min(5%)":r["recall_at_pmin"],
        "PEI@5%":          r["pei"],
        "FRES@5%":         r["fres"],
    })

df = pd.DataFrame(rows)

# Đánh dấu best value mỗi cột
numeric_cols = [c for c in df.columns if c != "Model"]
best_row = {}
for col in numeric_cols:
    best_row[col] = df[col].idxmax()

out_csv = HERE / "comparison_table.csv"
df.to_csv(out_csv, index=False, float_format="%.4f")
print(f"Saved: {out_csv}")
