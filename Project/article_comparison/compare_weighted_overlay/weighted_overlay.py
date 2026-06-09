"""
weighted_overlay_baseline.py
============================
Tái hiện phương pháp Weighted Overlay từ bài báo Lưu Thế Anh et al. (2014)
để tạo baseline benchmark so sánh với XGBoost.

Input:
  - daklak_comparison_dataset.parquet (đã build sẵn)
  - daklak_final_dataset_v3_pathways.parquet (lấy fire labels)
  - app_predictions_map.parquet (dự đoán XGBoost để so sánh)

Output:
  - baseline_weighted_overlay.parquet
  - benchmark_comparison.csv
  - baseline_map.png
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

#  Paths 
HERE = Path(__file__).parent
COMPARISON_DATA = HERE / "daklak_comparison_dataset.parquet"
FIRE_DATA       = HERE / "../data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"
XGB_PRED        = HERE / "../xgboost/models/v5/app_predictions_map.parquet"

OUT_PARQUET = HERE / "baseline_weighted_overlay.parquet"
OUT_CSV     = HERE / "benchmark_comparison.csv"
OUT_MAP     = HERE / "baseline_map.png"

#  Time window: mùa khô 2023-2024 ─
START_DATE = "2023-11-01"
END_DATE   = "2024-04-30"
DRY_MONTHS = [11, 12, 1, 2, 3, 4]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: AGGREGATE CLIMATOLOGY
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_climatology() -> pd.DataFrame:
    log.info("[1/5] Aggregate climatology mùa khô %s → %s", START_DATE, END_DATE)

    df = pd.read_parquet(COMPARISON_DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
        & (df["date"].dt.month.isin(DRY_MONTHS))
        ]
    log.info("      Rows sau lọc: %d", len(df))

    agg = df.groupby("grid_id").agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        tmean=("tmean", "mean"),
        rain_daily=("rain", "mean"),
        wind=("wind", "mean"),
        lulc_class=("lulc_class", "first"),
        tree_cover_pct=("tree_cover_pct", "first"),
        slope=("slope_mean", "first"),
        aspect_deg=("aspect_deg", "first"),
        dist_settle=("dist_settlement_km", "first"),
        dist_forest_edge=("dist_forest_edge_km", "first"),
        #  Gộp luôn fire labels 
        fire_observed=("fire", "max"),  # ô có cháy ít nhất 1 lần trong mùa khô
    ).reset_index()

    # Quy đổi rain mm/ngày → mm/tháng
    agg["rain_monthly"] = agg["rain_daily"] * 30

    # Đảm bảo fire_observed là int
    agg["fire_observed"] = agg["fire_observed"].fillna(0).astype(int)

    n_fire = int(agg["fire_observed"].sum())
    log.info("      Aggregated: %d grids", len(agg))
    log.info("      Grids có fire: %d / %d (%.2f%%)", n_fire, len(agg), 100 * n_fire / len(agg))
    log.info("      tmean range:        %.1f – %.1f °C", agg["tmean"].min(), agg["tmean"].max())
    log.info("      rain_monthly range: %.1f – %.1f mm", agg["rain_monthly"].min(), agg["rain_monthly"].max())
    log.info("      wind range:         %.2f – %.2f m/s", agg["wind"].min(), agg["wind"].max())

    return agg

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: SCORING FUNCTIONS (Bảng 2 của bài báo)
# ══════════════════════════════════════════════════════════════════════════════

def score_lulc(lulc_class: int, tree_cover: float) -> int:
    """
    Mapping ESA WorldCover + tree_cover → 4 cấp nguy cơ cháy.

    Bảng 2 bài báo (Lưu Thế Anh et al. 2014):
      4 - Rừng lá kim, rừng khộp, rừng trồng      (rừng thưa, dễ cháy)
      3 - Rừng tre nứa
      2 - Rừng hỗn giao, rừng thường xanh nghèo
      1 - Rừng lá rộng thường xanh trung bình và giàu

    ESA WorldCover không phân biệt loại rừng → dùng tree_cover_pct làm proxy:
      class 10 (Tree cover):
        tree_cover < 30%  → điểm 4  (≈ rừng thưa, rừng trồng, rừng khộp)
        30% ≤ cover < 50% → điểm 3  (≈ rừng tre nứa)
        50% ≤ cover < 70% → điểm 2  (≈ rừng hỗn giao, rừng thường xanh nghèo)
        cover ≥ 70%       → điểm 1  (≈ rừng lá rộng thường xanh trung bình-giàu)
      class 20/30 (Shrubland/Grassland): không phải rừng nhưng dễ cháy → điểm 4
      class 40 (Cropland/Nương rẫy): bài báo không xếp hạng rừng, dùng điểm 3
                                      vì tiếp giáp rừng và dễ cháy lan
      class 60 (Bare): → điểm 1 (không có vật liệu cháy)
      class 50/80 (Built/Water): → điểm 1
    """
    if lulc_class == 10:                  # Tree cover
        if tree_cover < 30:   return 4    # Rừng thưa/trồng/khộp
        elif tree_cover < 50: return 3    # Rừng tre nứa
        elif tree_cover < 70: return 2    # Rừng hỗn giao/thường xanh nghèo
        else:                 return 1    # Rừng lá rộng thường xanh trung bình-giàu
    elif lulc_class in (20, 30):          # Shrubland, Grassland → dễ cháy
        return 4
    elif lulc_class == 40:                # Cropland/Nương rẫy
        return 3
    else:                                 # Bare (60), Built (50), Water (80)
        return 1


def score_tmean(x: float) -> int:
    if x > 25:    return 4
    elif x >= 22: return 3
    elif x >= 20: return 2
    else:         return 1


def score_rain(x: float) -> int:  # mm/tháng
    if x < 5:     return 4
    elif x < 15:  return 3
    elif x < 20:  return 2
    else:         return 1


def score_wind(x: float) -> int:  # m/s
    if x > 7:      return 4
    elif x >= 4.3: return 3
    elif x >= 1.4: return 2
    else:          return 1


def score_slope(x: float) -> int:  # độ
    if x > 35:    return 4
    elif x >= 25: return 3
    elif x >= 15: return 2
    else:         return 1


def score_aspect(x: float) -> int:  # độ 0-360
    if 180 <= x < 270:                                       return 4   # tây nam
    elif (270 <= x < 315) or (45 <= x < 90):                 return 3
    elif (315 <= x <= 360) or (0 <= x < 45) or (135 <= x < 180): return 2
    else:                                                    return 1   # đông (90-135)


def score_dist_settle(x: float) -> int:  # km
    if x < 0.5:   return 4
    elif x < 1.0: return 3
    elif x < 1.5: return 2
    else:         return 1


def score_dist_forest(x: float) -> int:  # km
    if x < 0.05:   return 4
    elif x < 0.10: return 3
    elif x < 0.15: return 2
    else:          return 1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: WEIGHTED OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def weighted_overlay(agg: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/5] Áp dụng scoring functions")

    agg["s_lulc"]   = agg.apply(lambda r: score_lulc(r["lulc_class"], r["tree_cover_pct"]), axis=1)
    agg["s_tmean"]  = agg["tmean"].apply(score_tmean)
    agg["s_rain"]   = agg["rain_monthly"].apply(score_rain)
    agg["s_wind"]   = agg["wind"].apply(score_wind)
    agg["s_slope"]  = agg["slope"].apply(score_slope)
    agg["s_aspect"] = agg["aspect_deg"].apply(score_aspect)
    agg["s_settle"] = agg["dist_settle"].apply(score_dist_settle)
    agg["s_forest"] = agg["dist_forest_edge"].apply(score_dist_forest)

    log.info("[3/5] Tính weighted overlay")
    agg["risk_score"] = (
        0.40 * agg["s_lulc"]
        + 0.15 * agg["s_tmean"]
        + 0.15 * agg["s_rain"]
        + 0.10 * agg["s_wind"]
        + 0.05 * agg["s_slope"]
        + 0.05 * agg["s_aspect"]
        + 0.05 * agg["s_settle"]
        + 0.05 * agg["s_forest"]
    )
    agg["risk_level"] = agg["risk_score"].round().clip(1, 4).astype(int)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

PAPER_AREA = {1: 35.9, 2: 21.3, 3: 36.1, 4: 6.8}
PAPER_FIRE = {1: 0.0, 2: 24.0, 3: 30.0, 4: 46.0}


def evaluate(agg: pd.DataFrame) -> dict:
    log.info("[4/5] Đánh giá baseline")

    print("\n" + "=" * 65)
    print("BẢNG 1: Phân bố diện tích theo cấp nguy cơ")
    print("=" * 65)
    print(f"{'Cấp':<6}{'Số grid':>12}{'% (ours)':>14}{'% (paper)':>14}{'Δ':>10}")
    print("-" * 65)
    counts = agg["risk_level"].value_counts().sort_index()
    pct = (counts / len(agg) * 100).round(1)
    for lvl in [1, 2, 3, 4]:
        n = counts.get(lvl, 0)
        p = pct.get(lvl, 0.0)
        diff = p - PAPER_AREA[lvl]
        print(f"{lvl:<6}{n:>12,}{p:>13.1f}%{PAPER_AREA[lvl]:>13.1f}%{diff:>+9.1f}")

    print("\n" + "=" * 65)
    print("BẢNG 2: Kiểm chứng — phân bố grid có fire theo cấp")
    print("=" * 65)
    fire_grids = agg[agg["fire_observed"] == 1]
    if len(fire_grids) == 0:
        log.warning("Không có grid nào có fire trong giai đoạn này!")
        fire_pct = {}
    else:
        fire_counts = fire_grids["risk_level"].value_counts().sort_index()
        fire_pct = (fire_counts / len(fire_grids) * 100).round(1)
        print(f"{'Cấp':<6}{'Số fire':>12}{'% (ours)':>14}{'% (paper)':>14}{'Δ':>10}")
        print("-" * 65)
        for lvl in [1, 2, 3, 4]:
            n = fire_counts.get(lvl, 0)
            p = fire_pct.get(lvl, 0.0)
            diff = p - PAPER_FIRE[lvl]
            print(f"{lvl:<6}{n:>12,}{p:>13.1f}%{PAPER_FIRE[lvl]:>13.1f}%{diff:>+9.1f}")

        recall_high = (fire_grids["risk_level"] >= 3).mean() * 100
        print(f"\nRecall ở cấp III-IV: {recall_high:.1f}%  (paper: 76.0%)")

    # ML metrics
    y_true = agg["fire_observed"].values
    y_score = agg["risk_score"].values
    if y_true.sum() > 0:
        aucpr = average_precision_score(y_true, y_score)
        rocauc = roc_auc_score(y_true, y_score)
    else:
        aucpr = rocauc = float("nan")

    print("\n" + "=" * 65)
    print("BẢNG 3: ML metrics (Weighted Overlay)")
    print("=" * 65)
    print(f"  AUC-PR  : {aucpr:.6f}")
    print(f"  ROC-AUC : {rocauc:.6f}")

    return {"aucpr": aucpr, "rocauc": rocauc, "n_fire": int(y_true.sum())}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: COMPARE WITH XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_xgboost(agg: pd.DataFrame, baseline_metrics: dict) -> pd.DataFrame:
    log.info("[5/5] So sánh với XGBoost")

    if not XGB_PRED.exists():
        log.warning("Không tìm thấy %s, bỏ qua so sánh XGBoost", XGB_PRED)
        return pd.DataFrame()

    xgb = pd.read_parquet(XGB_PRED, columns=["grid_id", "date", "fire_prob"])
    xgb["date"] = pd.to_datetime(xgb["date"])
    xgb = xgb[
        (xgb["date"] >= START_DATE)
        & (xgb["date"] <= END_DATE)
        & (xgb["date"].dt.month.isin(DRY_MONTHS))
    ]
    xgb_agg = xgb.groupby("grid_id")["fire_prob"].mean().reset_index()
    xgb_agg.rename(columns={"fire_prob": "xgb_score"}, inplace=True)

    merged = agg[["grid_id", "fire_observed"]].merge(xgb_agg, on="grid_id", how="inner")
    y_true = merged["fire_observed"].values
    y_score = merged["xgb_score"].values

    xgb_aucpr = average_precision_score(y_true, y_score)
    xgb_rocauc = roc_auc_score(y_true, y_score)

    print("\n" + "=" * 65)
    print("BẢNG 4: So sánh Weighted Overlay vs XGBoost")
    print("=" * 65)
    print(f"{'Metric':<15}{'WeightedOverlay':>20}{'XGBoost':>20}{'Δ':>10}")
    print("-" * 65)
    print(f"{'AUC-PR':<15}{baseline_metrics['aucpr']:>20.6f}{xgb_aucpr:>20.6f}{xgb_aucpr - baseline_metrics['aucpr']:>+10.4f}")
    print(f"{'ROC-AUC':<15}{baseline_metrics['rocauc']:>20.6f}{xgb_rocauc:>20.6f}{xgb_rocauc - baseline_metrics['rocauc']:>+10.4f}")

    comparison = pd.DataFrame({
        "method": ["WeightedOverlay", "XGBoost"],
        "aucpr": [baseline_metrics["aucpr"], xgb_aucpr],
        "rocauc": [baseline_metrics["rocauc"], xgb_rocauc],
        "n_grids": [len(agg), len(merged)],
        "n_fire": [baseline_metrics["n_fire"], int(y_true.sum())],
    })
    comparison.to_csv(OUT_CSV, index=False)
    log.info("Saved %s", OUT_CSV)
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_map(agg: pd.DataFrame):
    log.info("Vẽ bản đồ baseline")
    colors = {1: "#FFEB99", 2: "#FFB347", 3: "#FF6B35", 4: "#C00000"}
    labels = {1: "Cấp I (Thấp)", 2: "Cấp II (TB)", 3: "Cấp III (Cao)", 4: "Cấp IV (Rất cao)"}

    fig, ax = plt.subplots(figsize=(10, 10))
    for lvl in [1, 2, 3, 4]:
        sub = agg[agg["risk_level"] == lvl]
        ax.scatter(sub["lon"], sub["lat"], c=colors[lvl], s=4, label=labels[lvl])

    fire = agg[agg["fire_observed"] == 1]
    ax.scatter(fire["lon"], fire["lat"], facecolors="none", edgecolors="black",
               s=20, linewidths=0.5, label=f"Fire observed (n={len(fire)})")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Bản đồ nguy cơ cháy rừng Đắk Lắk — Weighted Overlay Baseline\n"
                 "(Mùa khô 2023-2024, tái hiện Lưu Thế Anh et al. 2014)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_MAP, dpi=150, bbox_inches="tight")
    log.info("Saved %s", OUT_MAP)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    agg = aggregate_climatology()
    agg = weighted_overlay(agg)
    metrics = evaluate(agg)
    compare_with_xgboost(agg, metrics)
    plot_map(agg)

    agg.to_parquet(OUT_PARQUET, index=False)
    log.info("Saved %s", OUT_PARQUET)
    log.info("Done!")


if __name__ == "__main__":
    main()