"""
compare_models.py — So sánh trực quan ConvLSTM vs XGBoost v5
=============================================================
Input:
  - ConvLSTM : clstm/models/prob_map_test.npy + targets_test.npy
               (từ run 20260327_1038, 16 features, local training)
  - XGBoost  : xgboost/models/v5/app_predictions_map.parquet
               (v5 pathway-aware, 52 features)
  - Coords   : data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv

Output (cùng thư mục):
  fig1_pr_roc_curves.png      — PR curve + ROC curve overlay
  fig2_metrics_bar.png        — Bar chart so sánh metrics
  fig3_spatial_maps.png       — Bản đồ xác suất cháy cho 1 ngày mẫu
  fig4_temporal.png           — Số fire pixel phát hiện theo thời gian

Lưu ý về evaluation scope:
  ConvLSTM  → per-pixel-day  (137×138 spatial grid × ngày)
  XGBoost   → per-grid-day   (18,803 grids × ngày)
  Cả hai đều evaluate trên test set 2023-01-01 → 2024-12-31.
  PR/ROC curve được tính trên toàn bộ test set (tất cả ngày).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve, f1_score,
    confusion_matrix,
)

HERE        = Path(__file__).parent
ROOT        = HERE.parent

LATEST_MODEL_CLSTM = ROOT / "clstm/models/model_20260428_1418"

CLSTM_PRED  = LATEST_MODEL_CLSTM / "prob_map_test.npy"
CLSTM_TRUE  = LATEST_MODEL_CLSTM / "targets_test.npy"
XGB_PRED    = ROOT / "xgboost/models/v5/app_predictions_map.parquet"

COORDS_PATH = ROOT / "data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"
DATA_PATH   = ROOT / "data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"

# ConvLSTM grid layout
H, W      = 137, 138
SEQ_LEN   = 7
TEST_START = pd.Timestamp("2023-01-01")   # ngày đầu test split

# Ngày sample cho spatial map (chọn ngày mùa khô có nhiều lửa)
SAMPLE_DATE = "2024-03-15"

#  Màu sắc nhất quán cho 2 model 
CLR_CLSTM = "#E05C2A"   # cam đỏ
CLR_XGB   = "#2A7DE0"   # xanh dương


# ================================================================
# SECTION 1 — LOAD PREDICTIONS
# ================================================================

def load_clstm_predictions():
    """
    Trả về (preds, targets) dạng 1-D arrays cho toàn bộ test set.
    prob_map_test.npy: flattened sequential predictions
      - Mỗi sample = H*W pixel values
      - Thứ tự theo ngày (shuffle=False trong DataLoader)
    """
    print("Loading ConvLSTM predictions...")
    preds   = np.load(CLSTM_PRED)    # (n_samples * H * W,)
    targets = np.load(CLSTM_TRUE)    # (n_samples * H * W,)
    print(f"  ConvLSTM: {len(preds):,} pixel-predictions "
          f"({len(preds)//(H*W)} ngày × {H*W} pixels)")
    print(f"  Fire rate: {targets.mean():.4%}")
    return preds, targets


def load_xgb_predictions(test_start: pd.Timestamp):
    """
    Load XGBoost predictions cho test period từ parquet.
    Chỉ load cột cần thiết để tiết kiệm RAM.
    """
    print("Loading XGBoost predictions...")
    df = pd.read_parquet(
        XGB_PRED,
        columns=["date", "grid_id", "fire_prob", "fire", "lon", "lat"],
        engine="pyarrow",
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= test_start].reset_index(drop=True)

    preds   = df["fire_prob"].to_numpy(dtype=np.float32)
    targets = df["fire"].to_numpy(dtype=np.float32)
    print(f"  XGBoost : {len(preds):,} grid-predictions "
          f"({df['date'].nunique()} ngày × ~{len(preds)//df['date'].nunique()} grids)")
    print(f"  Fire rate: {targets.mean():.4%}")
    return preds, targets, df


def compute_metrics(preds, targets, name=""):
    """Tính đầy đủ metrics và threshold tốt nhất theo F1."""
    auc_pr  = average_precision_score(targets, preds)
    auc_roc = roc_auc_score(targets, preds)

    prec, rec, thresholds = precision_recall_curve(targets, preds)
    f1s      = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(f1s.argmax())
    thresh   = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_bin = (preds >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, y_bin).ravel()

    recall    = float(tp / (tp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
    csi       = float(tp / (tp + fp + fn + 1e-8))
    fpr_val   = float(fp / (fp + tn + 1e-8))

    m = {
        "name":      name,
        "auc_pr":    auc_pr,
        "auc_roc":   auc_roc,
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "csi":       csi,
        "fpr":       fpr_val,
        "threshold": thresh,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        # Curve data
        "pr_prec": prec, "pr_rec": rec,
        "roc_fpr": None, "roc_tpr": None,
    }

    fpr_curve, tpr_curve, _ = roc_curve(targets, preds)
    m["roc_fpr"] = fpr_curve
    m["roc_tpr"] = tpr_curve

    if name:
        print(f"\n  [{name}]")
        print(f"    AUC-PR  : {auc_pr:.4f}   AUC-ROC: {auc_roc:.4f}")
        print(f"    Recall  : {recall:.4f}   Precision: {precision:.4f}")
        print(f"    F1      : {f1:.4f}   CSI: {csi:.4f}   FPR: {fpr_val:.4f}")
        print(f"    Thresh  : {thresh:.3f}")
        print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    return m


# ================================================================
# SECTION 2 — PR CURVE + ROC CURVE
# ================================================================

def plot_pr_roc(m_clstm, m_xgb, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ConvLSTM vs XGBoost v5 — Precision-Recall & ROC Curve\n"
                 "(Test set: 2023-01-01 → 2024-12-31)", fontsize=13)

    #  PR Curve 
    ax = axes[0]
    baseline = m_clstm["tp"] / (m_clstm["tp"] + m_clstm["fn"] +
                                 m_xgb["tp"] + m_xgb["fn"]) * 0.5  # rough
    fire_rate_clstm = (m_clstm["tp"] + m_clstm["fn"]) / (
        m_clstm["tp"] + m_clstm["fp"] + m_clstm["fn"] + m_clstm["tn"]
    )

    ax.plot(m_clstm["pr_rec"], m_clstm["pr_prec"],
            color=CLR_CLSTM, lw=2,
            label=f"ConvLSTM  (AUC-PR={m_clstm['auc_pr']:.4f})")
    ax.plot(m_xgb["pr_rec"], m_xgb["pr_prec"],
            color=CLR_XGB, lw=2,
            label=f"XGBoost v5 (AUC-PR={m_xgb['auc_pr']:.4f})")

    # Điểm best threshold
    ax.scatter(m_clstm["recall"], m_clstm["precision"],
               color=CLR_CLSTM, zorder=5, s=80,
               label=f"ConvLSTM thresh={m_clstm['threshold']:.2f}")
    ax.scatter(m_xgb["recall"], m_xgb["precision"],
               color=CLR_XGB, zorder=5, s=80,
               label=f"XGBoost  thresh={m_xgb['threshold']:.2f}")

    ax.axhline(y=fire_rate_clstm, color="gray", linestyle="--",
               alpha=0.6, label=f"Random baseline ({fire_rate_clstm:.4f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    #  ROC Curve ─
    ax = axes[1]
    ax.plot(m_clstm["roc_fpr"], m_clstm["roc_tpr"],
            color=CLR_CLSTM, lw=2,
            label=f"ConvLSTM  (AUC-ROC={m_clstm['auc_roc']:.4f})")
    ax.plot(m_xgb["roc_fpr"], m_xgb["roc_tpr"],
            color=CLR_XGB, lw=2,
            label=f"XGBoost v5 (AUC-ROC={m_xgb['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ================================================================
# SECTION 3 — METRICS BAR CHART
# ================================================================

def plot_metrics_bar(m_clstm, m_xgb, out_path):
    metrics = ["auc_pr", "auc_roc", "recall", "precision", "f1", "csi"]
    labels  = ["AUC-PR", "AUC-ROC", "Recall", "Precision", "F1", "CSI"]

    vals_clstm = [m_clstm[k] for k in metrics]
    vals_xgb   = [m_xgb[k]   for k in metrics]

    x    = np.arange(len(metrics))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, vals_clstm, w, label="ConvLSTM",
                   color=CLR_CLSTM, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, vals_xgb,   w, label="XGBoost v5",
                   color=CLR_XGB,   alpha=0.85, edgecolor="white")

    # Giá trị trên mỗi bar
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5,
                color=CLR_CLSTM, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5,
                color=CLR_XGB, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("So sánh metrics — ConvLSTM vs XGBoost v5\n"
                 "(Test set: 2023-01-01 → 2024-12-31)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Chú thích phạm vi evaluation khác nhau
    ax.text(0.99, 0.97,
            "* ConvLSTM: per-pixel-day\n  XGBoost: per-grid-day",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ================================================================
# SECTION 4 — SPATIAL MAPS (SAMPLE DATE)
# ================================================================

def build_clstm_grid_mapping():
    """
    Trả về mảng sorted_grid_ids (shape: H*W,) — ánh xạ từ pixel index → grid_id.
    Tái tạo lại thứ tự build_grid_maps.py: grid_ids = np.sort(unique_grid_ids).
    Chỉ đọc header parquet để lấy danh sách grid_id.
    """
    print("  Đọc grid_id mapping từ parquet...")
    df_grids = pd.read_parquet(
        DATA_PATH,
        columns=["grid_id"],
        engine="pyarrow",
    )
    grid_ids = np.sort(df_grids["grid_id"].unique())   # sorted — khớp với build_grid_maps
    del df_grids
    return grid_ids   # (n_grids,) — index k → grid_ids[k]


def get_clstm_date_prediction(preds, sample_date: str):
    """
    Trích xuất prediction H×W cho 1 ngày cụ thể.

    sample_date : str — ngày muốn xem (trong test period 2023-01-08 trở đi)
    Returns : np.ndarray shape (H, W), float32
    """
    date_ts = pd.Timestamp(sample_date)
    # Ngày đầu tiên có prediction: TEST_START + seq_len days
    first_pred_date = TEST_START + pd.Timedelta(days=SEQ_LEN)
    delta_days = (date_ts - first_pred_date).days

    if delta_days < 0:
        raise ValueError(f"Ngày {sample_date} trước ngày prediction đầu tiên "
                         f"({first_pred_date.date()})")

    pixel_start = delta_days * H * W
    pixel_end   = pixel_start + H * W

    if pixel_end > len(preds):
        raise ValueError(f"Ngày {sample_date} vượt quá test set (len={len(preds)})")

    grid_probs = preds[pixel_start:pixel_end]   # (H*W,)
    return grid_probs.reshape(H, W)


def plot_spatial_maps(clstm_preds, xgb_df, sample_date: str,
                      coords_path: Path, out_path: Path):
    """
    Vẽ 3 bản đồ cạnh nhau:
      Left   : XGBoost v5 fire probability
      Middle : ConvLSTM fire probability
      Right  : Ground truth fire locations
    """
    print(f"  Spatial map cho ngày {sample_date}...")
    date_ts = pd.Timestamp(sample_date)

    #  XGBoost predictions cho ngày này 
    xgb_day = xgb_df[xgb_df["date"] == date_ts].copy()
    if xgb_day.empty:
        print(f"  WARN: Không có XGBoost prediction cho {sample_date}")
        return

    #  ConvLSTM prediction ─
    try:
        clstm_grid = get_clstm_date_prediction(clstm_preds, sample_date)
    except ValueError as e:
        print(f"  WARN: {e}")
        return

    #  Grid ID → lat/lon mapping ─
    coords = pd.read_csv(coords_path)
    # Cột tên có thể khác nhau — tự detect
    id_col  = [c for c in coords.columns if "grid" in c.lower()][0]
    lat_col = [c for c in coords.columns if "lat"  in c.lower()][0]
    lon_col = [c for c in coords.columns if "lon"  in c.lower()][0]
    coords  = coords.rename(columns={id_col: "grid_id", lat_col: "lat", lon_col: "lon"})

    # Build sorted_grid_ids → pixel_index mapping
    grid_ids  = np.sort(coords["grid_id"].unique())   # sorted, khớp s2_build_grid_maps
    coord_map = coords.set_index("grid_id")[["lat", "lon"]]

    # ConvLSTM: flatten grid → DataFrame với grid_id
    clstm_flat = clstm_grid.flatten()   # (H*W,)
    # Chỉ lấy valid pixels (có grid_id thực tế)
    n_valid   = min(len(grid_ids), len(clstm_flat))
    clstm_df  = pd.DataFrame({
        "grid_id":   grid_ids[:n_valid],
        "fire_prob": clstm_flat[:n_valid],
    }).join(coord_map, on="grid_id")

    # Ground truth từ XGBoost df (cả 2 có cùng ground truth)
    fire_points = xgb_day[xgb_day["fire"] == 1]

    #  Plot 
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    cmap   = plt.cm.YlOrRd
    vmin, vmax = 0, 0.5   # scale cố định để so sánh

    titles = [f"XGBoost v5\n(AUC-PR trên toàn test set)",
              f"ConvLSTM\n(16 features, local run)",
              "Ground Truth\n(Fire locations)"]

    # XGBoost map
    sc0 = axes[0].scatter(xgb_day["lon"], xgb_day["lat"],
                          c=xgb_day["fire_prob"], cmap=cmap,
                          vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
    plt.colorbar(sc0, ax=axes[0], label="Fire probability")

    # ConvLSTM map
    sc1 = axes[1].scatter(clstm_df["lon"], clstm_df["lat"],
                          c=clstm_df["fire_prob"], cmap=cmap,
                          vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
    plt.colorbar(sc1, ax=axes[1], label="Fire probability")

    # Ground truth
    axes[2].scatter(xgb_day["lon"], xgb_day["lat"],
                    c="lightgray", s=1.5, rasterized=True, label="No fire")
    if not fire_points.empty:
        axes[2].scatter(fire_points["lon"], fire_points["lat"],
                        c="red", s=8, zorder=5, label=f"Fire (n={len(fire_points)})")
    axes[2].legend(fontsize=9, markerscale=3)

    for ax, title in zip(axes, titles):
        ax.set_title(f"{title}\n{sample_date}", fontsize=11)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    plt.suptitle("Bản đồ xác suất cháy rừng — Đắk Lắk", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ================================================================
# SECTION 5 — TEMPORAL COMPARISON
# ================================================================

def plot_temporal(clstm_preds, clstm_targets, xgb_df, m_clstm, m_xgb, out_path):
    """
    So sánh số fire pixels/grids được phát hiện theo ngày trong test set.
    Dùng threshold tốt nhất của mỗi model.
    """
    print("  Building temporal comparison...")

    #  ConvLSTM: tính fire pixels phát hiện mỗi ngày 
    n_days_clstm  = len(clstm_preds) // (H * W)
    clstm_thresh  = m_clstm["threshold"]
    clstm_binary  = (clstm_preds >= clstm_thresh).astype(np.int8)
    clstm_targets_bin = clstm_targets.astype(np.int8)

    clstm_daily_tp    = []
    clstm_daily_fire  = []   # ground truth
    first_pred_date   = TEST_START + pd.Timedelta(days=SEQ_LEN)
    date_range_clstm  = pd.date_range(first_pred_date, periods=n_days_clstm, freq="D")

    pixels = H * W
    for i in range(n_days_clstm):
        start = i * pixels
        end   = start + pixels
        clstm_daily_tp.append(int(
            ((clstm_binary[start:end] == 1) & (clstm_targets_bin[start:end] == 1)).sum()
        ))
        clstm_daily_fire.append(int(clstm_targets_bin[start:end].sum()))

    clstm_daily_tp   = np.array(clstm_daily_tp)
    clstm_daily_fire = np.array(clstm_daily_fire)

    #  XGBoost: tính fire grids phát hiện mỗi ngày 
    xgb_thresh  = m_xgb["threshold"]
    xgb_df_test = xgb_df[xgb_df["date"] >= TEST_START].copy()
    xgb_df_test["predicted"] = (xgb_df_test["fire_prob"] >= xgb_thresh).astype(int)
    xgb_df_test["tp"]        = ((xgb_df_test["predicted"] == 1) &
                                 (xgb_df_test["fire"] == 1)).astype(int)

    xgb_daily = xgb_df_test.groupby("date").agg(
        tp=("tp", "sum"),
        fire=("fire", "sum"),
    ).reset_index().sort_values("date")

    #  Plot 
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("So sánh phát hiện cháy theo ngày — Test set 2023–2024", fontsize=13)

    # Ground truth fire count
    ax = axes[0]
    ax.fill_between(xgb_daily["date"], xgb_daily["fire"],
                    alpha=0.4, color="gray", label="Fire grids (ground truth)")
    ax.set_ylabel("Số grid cháy (GT)", fontsize=10)
    ax.set_title("Ground Truth — số grid cháy mỗi ngày", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ConvLSTM TP per day
    ax = axes[1]
    ax.fill_between(date_range_clstm, clstm_daily_tp,
                    alpha=0.6, color=CLR_CLSTM, label="ConvLSTM TP/ngày")
    ax.plot(date_range_clstm, clstm_daily_fire,
            color="gray", lw=1, alpha=0.5, label="GT fire pixels")
    daily_recall_clstm = np.where(
        clstm_daily_fire > 0,
        clstm_daily_tp / np.maximum(clstm_daily_fire, 1),
        np.nan,
    )
    ax2 = ax.twinx()
    ax2.plot(date_range_clstm, daily_recall_clstm,
             color=CLR_CLSTM, lw=1.5, alpha=0.7, linestyle="--")
    ax2.set_ylabel("Daily Recall", fontsize=9, color=CLR_CLSTM)
    ax2.set_ylim(0, 1)
    ax.set_ylabel("Pixels", fontsize=10)
    ax.set_title(f"ConvLSTM — TP/ngày (thresh={clstm_thresh:.3f})", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

    # XGBoost TP per day
    ax = axes[2]
    ax.fill_between(xgb_daily["date"], xgb_daily["tp"],
                    alpha=0.6, color=CLR_XGB, label="XGBoost TP/ngày")
    ax.plot(xgb_daily["date"], xgb_daily["fire"],
            color="gray", lw=1, alpha=0.5, label="GT fire grids")
    daily_recall_xgb = np.where(
        xgb_daily["fire"] > 0,
        xgb_daily["tp"] / np.maximum(xgb_daily["fire"],1),
        np.nan,
    )
    ax2 = ax.twinx()
    ax2.plot(xgb_daily["date"], daily_recall_xgb,
             color=CLR_XGB, lw=1.5, alpha=0.7, linestyle="--")
    ax2.set_ylabel("Daily Recall", fontsize=9, color=CLR_XGB)
    ax2.set_ylim(0, 1)
    ax.set_ylabel("Grids", fontsize=10)
    ax.set_title(f"XGBoost v5 — TP/ngày (thresh={xgb_thresh:.3f})", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("So sánh ConvLSTM vs XGBoost v5")
    print("=" * 60)

    #  Load 
    clstm_preds, clstm_targets  = load_clstm_predictions()
    xgb_preds, xgb_targets, xgb_df = load_xgb_predictions(TEST_START)

    #  Metrics ─
    print("\nComputing metrics...")
    m_clstm = compute_metrics(clstm_preds, clstm_targets, name="ConvLSTM")
    m_xgb   = compute_metrics(xgb_preds,   xgb_targets,   name="XGBoost v5")

    #  Figure 1: PR + ROC 
    print("\nPlotting PR + ROC curves...")
    plot_pr_roc(m_clstm, m_xgb, HERE / "fig1_pr_roc_curves.png")

    #  Figure 2: Metrics bar ─
    print("Plotting metrics bar chart...")
    plot_metrics_bar(m_clstm, m_xgb, HERE / "fig2_metrics_bar.png")

    #  Figure 3: Spatial maps 
    print("Plotting spatial maps...")
    plot_spatial_maps(
        clstm_preds, xgb_df, SAMPLE_DATE,
        COORDS_PATH,
        HERE / "fig3_spatial_maps.png",
    )

    #  Figure 4: Temporal 
    print("Plotting temporal comparison...")
    plot_temporal(clstm_preds, clstm_targets, xgb_df,
                  m_clstm, m_xgb, HERE / "fig4_temporal.png")

    #  Summary ─
    print("\n" + "=" * 60)
    print("TỔNG KẾT SO SÁNH")
    print("=" * 60)
    print(f"{'Metric':<15} {'ConvLSTM':>12} {'XGBoost v5':>12} {'Δ (XGB-CLSTM)':>15}")
    print("-" * 56)
    for k, label in [("auc_pr","AUC-PR"), ("auc_roc","AUC-ROC"),
                      ("recall","Recall"), ("f1","F1"), ("csi","CSI")]:
        delta = m_xgb[k] - m_clstm[k]
        sign  = "+" if delta > 0 else ""
        print(f"  {label:<13} {m_clstm[k]:>12.4f} {m_xgb[k]:>12.4f} "
              f"{sign}{delta:>13.4f}")
    print("=" * 60)
    print("\nOutput files:")
    for f in ["fig1_pr_roc_curves.png", "fig2_metrics_bar.png",
              "fig3_spatial_maps.png", "fig4_temporal.png"]:
        print(f"  {HERE / f}")