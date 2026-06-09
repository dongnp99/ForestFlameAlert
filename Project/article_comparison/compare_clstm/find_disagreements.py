"""
find_disagreements.py — Phân tích sự khác biệt giữa XGBoost v5 và ConvLSTM
=============================================================================
Phân loại từng ground-truth fire pixel (fire=1) trong test set theo 4 nhóm:

  both_tp    : Cả 2 model đều phát hiện đúng  (TP ∩ TP)
  xgb_only   : Chỉ XGBoost phát hiện, ConvLSTM bỏ sót
  clstm_only : Chỉ ConvLSTM phát hiện, XGBoost bỏ sót
  both_fn    : Cả 2 đều bỏ sót                (FN ∩ FN)

Input:
  prob_map_test.npy + targets_test.npy   (ConvLSTM, model_20260428_1418)
  app_predictions_map.parquet            (XGBoost v5)
  daklak_final_dataset_v3_pathways.parquet  (để lấy sorted grid_id mapping)

Output:
  disagreements.parquet          — tất cả fire pixels với phân loại
  fig_a_agreement_breakdown.png  — pie + bar phân bổ 4 loại
  fig_b_spatial_disagree.png     — bản đồ không gian của từng loại
  fig_c_monthly_breakdown.png    — phân bổ theo tháng
  fig_d_prob_scatter.png         — scatter xgb_prob vs clstm_prob
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import precision_recall_curve

#  Paths ─
HERE  = Path(__file__).parent
ROOT  = HERE.parent.parent

LATEST_MODEL_CLSTM = ROOT / "clstm/models/model_20260428_1418"
CLSTM_PRED  = LATEST_MODEL_CLSTM / "prob_map_test.npy"
CLSTM_TRUE  = LATEST_MODEL_CLSTM / "targets_test.npy"
XGB_PRED    = ROOT / "xgboost/models/v5/app_predictions_map.parquet"
DATA_PATH   = ROOT / "data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"

#  Constants ─
H, W      = 137, 138
SEQ_LEN   = 7
TEST_START = pd.Timestamp("2023-01-01")
FIRST_PRED = TEST_START + pd.Timedelta(days=SEQ_LEN)   # 2023-01-08

#  Màu sắc nhất quán ─
COLORS = {
    "both_tp":    "#27AE60",   # xanh lá
    "xgb_only":   "#2A7DE0",   # xanh dương
    "clstm_only": "#E05C2A",   # cam đỏ
    "both_fn":    "#95A5A6",   # xám
}
LABELS = {
    "both_tp":    "Cả 2 phát hiện (Both TP)",
    "xgb_only":   "Chỉ XGBoost phát hiện",
    "clstm_only": "Chỉ ConvLSTM phát hiện",
    "both_fn":    "Cả 2 bỏ sót (Both FN)",
}
CAT_ORDER = ["both_tp", "xgb_only", "clstm_only", "both_fn"]


# ================================================================
# SECTION 1 — LOAD & THRESHOLD
# ================================================================

def _best_f1_threshold(targets: np.ndarray, preds: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(targets, preds)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = int(f1s.argmax())
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def load_data():
    print("=" * 60)
    print("Load dữ liệu...")

    # ConvLSTM
    clstm_preds   = np.load(CLSTM_PRED)    # (N*H*W,)
    clstm_targets = np.load(CLSTM_TRUE)    # (N*H*W,)
    n_days_clstm  = len(clstm_preds) // (H * W)
    print(f"  ConvLSTM : {n_days_clstm} ngày × {H*W} pixels")

    # XGBoost
    xgb_df = pd.read_parquet(
        XGB_PRED,
        columns=["date", "grid_id", "fire_prob", "fire", "lon", "lat"],
        engine="pyarrow",
    )
    xgb_df["date"] = pd.to_datetime(xgb_df["date"])
    xgb_test = xgb_df[xgb_df["date"] >= FIRST_PRED].reset_index(drop=True)
    print(f"  XGBoost  : {xgb_test['date'].nunique()} ngày")

    # Sorted grid_id mapping (phải trùng với cách build ConvLSTM grid)
    print("  Đọc sorted grid_id mapping...")
    df_grids = pd.read_parquet(DATA_PATH, columns=["grid_id"], engine="pyarrow")
    sorted_grid_ids = np.sort(df_grids["grid_id"].unique())
    del df_grids
    n_valid = min(len(sorted_grid_ids), H * W)
    print(f"  Grid mapping: {n_valid} grids hợp lệ")

    # Thresholds (best F1)
    print("  Tính threshold tối ưu...")
    clstm_thresh = _best_f1_threshold(clstm_targets, clstm_preds)
    xgb_preds_test = xgb_test["fire_prob"].to_numpy(np.float32)
    xgb_targets_test = xgb_test["fire"].to_numpy(np.float32)
    xgb_thresh = _best_f1_threshold(xgb_targets_test, xgb_preds_test)
    print(f"  Threshold — CLSTM: {clstm_thresh:.4f}   XGB: {xgb_thresh:.4f}")

    return (clstm_preds, clstm_targets, xgb_test,
            sorted_grid_ids[:n_valid], clstm_thresh, xgb_thresh)


# ================================================================
# SECTION 2 — BUILD COMPARISON DATAFRAME
# ================================================================

def build_fire_comparison(
    clstm_preds: np.ndarray,
    xgb_test: pd.DataFrame,
    sorted_grid_ids: np.ndarray,
    clstm_thresh: float,
    xgb_thresh: float,
) -> pd.DataFrame:
    """
    Lặp qua từng ngày, map pixel index → grid_id, merge với XGBoost,
    chỉ giữ lại ground truth fire pixels (fire=1).

    Returns: DataFrame với columns:
        date, grid_id, lon, lat, xgb_prob, clstm_prob,
        xgb_bin, clstm_bin, category
    """
    print("\nXây dựng comparison DataFrame...")

    n_days    = len(clstm_preds) // (H * W)
    n_valid   = len(sorted_grid_ids)
    date_range = pd.date_range(FIRST_PRED, periods=n_days, freq="D")

    # Group XGBoost by date cho fast lookup
    xgb_test = xgb_test.copy()
    xgb_test["xgb_bin"] = (xgb_test["fire_prob"] >= xgb_thresh).astype(np.int8)
    xgb_by_date = {
        d: g.reset_index(drop=True)
        for d, g in xgb_test.groupby("date")
    }

    chunks = []
    for i, date in enumerate(date_range):
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{n_days}  ({date.date()})")

        xgb_day = xgb_by_date.get(date)
        if xgb_day is None:
            continue

        # ConvLSTM predictions cho ngày này
        offset = i * H * W
        clstm_flat = clstm_preds[offset : offset + H * W].astype(np.float32)

        # Ánh xạ XGBoost grid_id → vị trí trong sorted_grid_ids
        gids     = xgb_day["grid_id"].values
        pos      = np.searchsorted(sorted_grid_ids, gids)
        in_range = pos < n_valid
        matched  = np.zeros(len(gids), dtype=bool)
        matched[in_range] = (sorted_grid_ids[pos[in_range]] == gids[in_range])

        if not matched.any():
            continue

        # Chỉ lấy grids được match
        xgb_valid   = xgb_day[matched].copy()
        pos_valid   = pos[matched]

        clstm_probs = clstm_flat[pos_valid]
        clstm_bins  = (clstm_probs >= clstm_thresh).astype(np.int8)

        fire_gt  = xgb_valid["fire"].values.astype(np.int8)
        xgb_bins = xgb_valid["xgb_bin"].values

        # Chỉ giữ ground truth fire pixels
        fire_mask = fire_gt == 1
        if not fire_mask.any():
            continue

        xb = xgb_bins[fire_mask]
        cb = clstm_bins[fire_mask]

        cats = np.full(fire_mask.sum(), "both_fn", dtype=object)
        cats[(xb == 1) & (cb == 1)] = "both_tp"
        cats[(xb == 1) & (cb == 0)] = "xgb_only"
        cats[(xb == 0) & (cb == 1)] = "clstm_only"

        fire_df = xgb_valid[fire_mask][["grid_id", "lon", "lat",
                                        "fire_prob", "xgb_bin"]].copy()
        fire_df.rename(columns={"fire_prob": "xgb_prob"}, inplace=True)
        fire_df["date"]       = date
        fire_df["clstm_prob"] = clstm_probs[fire_mask]
        fire_df["clstm_bin"]  = cb
        fire_df["category"]   = cats

        chunks.append(fire_df[["date", "grid_id", "lon", "lat",
                                "xgb_prob", "clstm_prob",
                                "xgb_bin", "clstm_bin", "category"]])

    if not chunks:
        raise RuntimeError("Không tìm thấy fire pixels trong test period!")

    df = pd.concat(chunks, ignore_index=True)
    df["xgb_prob"]   = df["xgb_prob"].astype(np.float32)
    df["clstm_prob"] = df["clstm_prob"].astype(np.float32)
    print(f"\n  Tổng fire pixels: {len(df):,}")
    return df


def print_stats(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("THỐNG KÊ PHÂN LOẠI FIRE PIXELS")
    print("=" * 60)
    total = len(df)
    print(f"  {'Category':<35} {'Count':>8}  {'%':>6}")
    print("  " + "-" * 55)
    for cat in CAT_ORDER:
        n = (df["category"] == cat).sum()
        pct = 100 * n / total if total > 0 else 0
        print(f"  {LABELS[cat]:<35} {n:>8,}  {pct:>5.1f}%")
    print("  " + "-" * 55)
    print(f"  {'Tổng fire pixels':<35} {total:>8,}  100.0%")

    # Tỷ lệ recall từng model
    tp_xgb  = (df["category"].isin(["both_tp", "xgb_only"])).sum()
    tp_clstm = (df["category"].isin(["both_tp", "clstm_only"])).sum()
    print(f"\n  Recall tương đương (fire pixels):")
    print(f"    XGBoost  : {tp_xgb}/{total} = {100*tp_xgb/total:.1f}%")
    print(f"    ConvLSTM : {tp_clstm}/{total} = {100*tp_clstm/total:.1f}%")

    # Điểm cháy độc quyền mỗi model
    xgb_excl  = (df["category"] == "xgb_only").sum()
    clstm_excl = (df["category"] == "clstm_only").sum()
    print(f"\n  Điểm cháy chỉ 1 model phát hiện được:")
    print(f"    XGBoost unique   : {xgb_excl:,} ({100*xgb_excl/total:.1f}%)")
    print(f"    ConvLSTM unique  : {clstm_excl:,} ({100*clstm_excl/total:.1f}%)")
    print("=" * 60)


# ================================================================
# SECTION 3 — FIGURES
# ================================================================

def plot_agreement_breakdown(df: pd.DataFrame, out_path: Path):
    """Fig A: Pie chart + bar chart phân bổ 4 loại."""
    counts = {cat: (df["category"] == cat).sum() for cat in CAT_ORDER}
    total  = sum(counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phân loại ground-truth fire pixels — ConvLSTM vs XGBoost v5\n"
                 "(Test set: 2023-01-08 → 2024-12-31)", fontsize=13)

    #  Pie chart ─
    ax = axes[0]
    sizes  = [counts[c] for c in CAT_ORDER]
    colors = [COLORS[c] for c in CAT_ORDER]
    wedge_labels = [f"{LABELS[c]}\n{counts[c]:,} ({100*counts[c]/total:.1f}%)"
                    for c in CAT_ORDER]
    ax.pie(sizes, labels=wedge_labels, colors=colors,
           startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Phân bổ theo số lượng", fontsize=11)

    #  Bar chart ─
    ax = axes[1]
    x   = np.arange(len(CAT_ORDER))
    pct = [100 * counts[c] / total for c in CAT_ORDER]
    bars = ax.bar(x, pct, color=[COLORS[c] for c in CAT_ORDER],
                  edgecolor="white", width=0.6)
    for bar, p, c in zip(bars, pct, CAT_ORDER):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{counts[c]:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c].replace(" (Both TP)", "").replace(" (Both FN)", "")
                        for c in CAT_ORDER], fontsize=9.5)
    ax.set_ylabel("% fire pixels", fontsize=11)
    ax.set_ylim(0, max(pct) * 1.2)
    ax.set_title("Phần trăm từng loại", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_spatial_disagreement(df: pd.DataFrame, out_path: Path):
    """
    Fig B: Bản đồ không gian.
    - Trái: tần suất xgb_only per grid (bao nhiêu ngày XGB phát hiện, CLSTM bỏ sót)
    - Phải: tần suất clstm_only per grid
    - Dưới trái: both_tp frequency
    - Dưới phải: scatter tất cả fire pixels, màu = category
    """
    # Tổng hợp theo grid_id
    grid_stats = (
        df.groupby(["grid_id", "lon", "lat", "category"])
        .size()
        .reset_index(name="count")
        .pivot_table(index=["grid_id", "lon", "lat"],
                     columns="category", values="count", fill_value=0)
        .reset_index()
    )
    for cat in CAT_ORDER:
        if cat not in grid_stats.columns:
            grid_stats[cat] = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Phân bổ không gian — sự bất đồng giữa ConvLSTM và XGBoost v5\n"
                 "(Đắk Lắk, Test 2023–2024)", fontsize=13)

    panels = [
        ("both_tp",    "Cả 2 phát hiện (Both TP)"),
        ("xgb_only",   "Chỉ XGBoost phát hiện"),
        ("clstm_only", "Chỉ ConvLSTM phát hiện"),
        ("both_fn",    "Cả 2 bỏ sót (Both FN)"),
    ]

    for ax, (cat, title) in zip(axes.flat, panels):
        vmax = max(grid_stats[cat].max(), 1)
        sc = ax.scatter(
            grid_stats["lon"], grid_stats["lat"],
            c=grid_stats[cat],
            cmap="YlOrRd",
            vmin=0, vmax=vmax,
            s=3, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Số ngày")
        ax.set_title(f"{title}\n(màu = số ngày)", fontsize=10,
                     color=COLORS[cat], fontweight="bold")
        ax.set_xlabel("Kinh độ", fontsize=8)
        ax.set_ylabel("Vĩ độ",  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_monthly_breakdown(df: pd.DataFrame, out_path: Path):
    """Fig C: Stacked bar chart số fire pixels theo tháng."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(["month", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CAT_ORDER, fill_value=0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Phân bổ fire pixels theo tháng — ConvLSTM vs XGBoost v5\n"
                 "(Test set: 2023–2024)", fontsize=13)

    x     = np.arange(len(monthly))
    xlabs = [str(p) for p in monthly.index]

    #  Stacked bar ─
    ax = axes[0]
    bottom = np.zeros(len(monthly))
    for cat in CAT_ORDER:
        vals = monthly[cat].values
        ax.bar(x, vals, bottom=bottom, color=COLORS[cat],
               label=LABELS[cat], edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_ylabel("Số fire pixels", fontsize=10)
    ax.set_title("Số lượng tuyệt đối theo tháng", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    #  Tỷ lệ phần trăm ─
    ax = axes[1]
    totals = monthly.sum(axis=1).values
    bottom = np.zeros(len(monthly))
    for cat in CAT_ORDER:
        pcts = 100 * monthly[cat].values / np.maximum(totals, 1)
        ax.bar(x, pcts, bottom=bottom, color=COLORS[cat],
               label=LABELS[cat], edgecolor="white", linewidth=0.4)
        bottom += pcts

    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% fire pixels", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Tỷ lệ phần trăm theo tháng", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_prob_scatter(df: pd.DataFrame, clstm_thresh: float,
                      xgb_thresh: float, out_path: Path):
    """
    Fig D: Scatter plot xgb_prob vs clstm_prob cho từng fire pixel.
    Màu = category, giúp thấy vùng xác suất của từng loại bất đồng.
    """
    # Lấy mẫu để không vẽ quá nhiều điểm (nếu > 20K)
    max_pts = 20_000
    if len(df) > max_pts:
        df_plot = df.sample(max_pts, random_state=42)
    else:
        df_plot = df

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Xác suất cháy của ground-truth fire pixels — XGBoost v5 vs ConvLSTM\n"
        "(chỉ hiển thị fire pixels; màu = phân loại model)",
        fontsize=12,
    )

    #  Scatter: tất cả category 
    ax = axes[0]
    for cat in CAT_ORDER:
        sub = df_plot[df_plot["category"] == cat]
        ax.scatter(sub["xgb_prob"], sub["clstm_prob"],
                   c=COLORS[cat], s=6, alpha=0.5,
                   label=f"{LABELS[cat]} (n={len(sub):,})", rasterized=True)

    ax.axvline(xgb_thresh,   color="#2A7DE0", linestyle="--", lw=1.2,
               alpha=0.7, label=f"XGB thresh {xgb_thresh:.3f}")
    ax.axhline(clstm_thresh, color="#E05C2A", linestyle="--", lw=1.2,
               alpha=0.7, label=f"CLSTM thresh {clstm_thresh:.3f}")
    ax.set_xlabel("XGBoost fire probability", fontsize=10)
    ax.set_ylabel("ConvLSTM fire probability", fontsize=10)
    ax.set_title("Tất cả fire pixels", fontsize=11)
    ax.legend(fontsize=7.5, markerscale=2)
    ax.grid(True, alpha=0.2)

    #  Chỉ "bất đồng": xgb_only + clstm_only 
    ax = axes[1]
    for cat in ["xgb_only", "clstm_only"]:
        sub = df_plot[df_plot["category"] == cat]
        ax.scatter(sub["xgb_prob"], sub["clstm_prob"],
                   c=COLORS[cat], s=8, alpha=0.6,
                   label=f"{LABELS[cat]} (n={len(sub):,})", rasterized=True)

    ax.axvline(xgb_thresh,   color="#2A7DE0", linestyle="--", lw=1.2, alpha=0.7)
    ax.axhline(clstm_thresh, color="#E05C2A", linestyle="--", lw=1.2, alpha=0.7)

    # Vùng bất đồng (highlight quadrants)
    ax.axvspan(xgb_thresh, 1.0, alpha=0.05, color="#2A7DE0")
    ax.axhspan(clstm_thresh, 1.0, alpha=0.05, color="#E05C2A")

    ax.set_xlabel("XGBoost fire probability", fontsize=10)
    ax.set_ylabel("ConvLSTM fire probability", fontsize=10)
    ax.set_title("Chỉ điểm bất đồng (xgb_only + clstm_only)", fontsize=11)
    ax.legend(fontsize=8.5, markerscale=2)
    ax.grid(True, alpha=0.2)

    # Ghi chú góc phần tư
    ax.text(xgb_thresh + 0.01, 0.02,
            "XGB phát hiện\nCLSTM bỏ sót", fontsize=7,
            color="#2A7DE0", va="bottom")
    ax.text(0.01, clstm_thresh + 0.01,
            "CLSTM phát hiện\nXGB bỏ sót", fontsize=7,
            color="#E05C2A", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    #  Load 
    (clstm_preds, clstm_targets,
     xgb_test, sorted_grid_ids,
     clstm_thresh, xgb_thresh) = load_data()

    #  Build comparison 
    fire_df = build_fire_comparison(
        clstm_preds, xgb_test,
        sorted_grid_ids, clstm_thresh, xgb_thresh,
    )

    #  Statistics 
    print_stats(fire_df)

    #  Save parquet 
    out_parquet = HERE / "disagreements.parquet"
    fire_df.to_parquet(out_parquet, engine="pyarrow", index=False)
    print(f"\n  Saved → {out_parquet}")

    #  Figures ─
    print("\nVẽ biểu đồ...")
    plot_agreement_breakdown(fire_df, HERE / "fig_a_agreement_breakdown.png")
    plot_spatial_disagreement(fire_df, HERE / "fig_b_spatial_disagree.png")
    plot_monthly_breakdown(fire_df, HERE / "fig_c_monthly_breakdown.png")
    plot_prob_scatter(fire_df, clstm_thresh, xgb_thresh,
                      HERE / "fig_d_prob_scatter.png")

    print("\n" + "=" * 60)
    print("HOÀN TẤT")
    print("=" * 60)
    print(f"  disagreements.parquet         → {HERE / 'disagreements.parquet'}")
    print(f"  fig_a_agreement_breakdown.png → {HERE / 'fig_a_agreement_breakdown.png'}")
    print(f"  fig_b_spatial_disagree.png    → {HERE / 'fig_b_spatial_disagree.png'}")
    print(f"  fig_c_monthly_breakdown.png   → {HERE / 'fig_c_monthly_breakdown.png'}")
    print(f"  fig_d_prob_scatter.png        → {HERE / 'fig_d_prob_scatter.png'}")