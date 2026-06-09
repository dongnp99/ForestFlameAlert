"""
weighted_overlay_baseline_v2.py
================================
Phiên bản cải tiến dùng NDVI phenology để phân biệt rừng khộp
(deciduous dipterocarp forest) khỏi rừng thường xanh — match
chính xác hơn cách phân loại của bài báo Lưu Thế Anh (2014).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).parent
LOG_FILE = HERE / "weighted_overlay_v2_log.txt"


class _Tee:
    """Ghi đồng thời ra stdout và file txt."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def _setup_logging(log_file: Path):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


log = logging.getLogger(__name__)

COMPARISON_DATA = HERE / "daklak_comparison_dataset.parquet"
PHENOLOGY_DATA = HERE / "../data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"
XGB_PRED = HERE / "../xgboost/models/v5/app_predictions_map.parquet"

OUT_PARQUET = HERE / "baseline_weighted_overlay_v2.parquet"
OUT_CSV = HERE / "benchmark_comparison_v2.csv"
OUT_MAP = HERE / "baseline_map_v2.png"

START_DATE = "2023-11-01"
END_DATE = "2024-04-30"
DRY_MONTHS = [11, 12, 1, 2, 3, 4]

# Năm dùng để tính phenology (nguyên năm để có cả mùa khô + mưa)
PHENO_YEAR = 2023


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: COMPUTE NDVI PHENOLOGY (NEW!)
# ══════════════════════════════════════════════════════════════════════════════
def compute_phenology() -> pd.DataFrame:
    log.info("[0/6] Tính NDVI phenology từ time-series %d", PHENO_YEAR)

    df = pd.read_parquet(
        PHENOLOGY_DATA,
        columns=["grid_id", "date", "ndvi"],
        filters=[
            ("date", ">=", pd.Timestamp(f"{PHENO_YEAR}-01-01")),
            ("date", "<=", pd.Timestamp(f"{PHENO_YEAR}-12-31")),
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["ndvi"])

    # Loại NDVI bất thường (mây, snow, water)
    df = df[(df["ndvi"] > 0) & (df["ndvi"] < 1)]
    log.info("      Loaded %d valid rows", len(df))

    df["is_dry"] = df["date"].dt.month.isin(DRY_MONTHS)

    # Dùng percentile robust thay vì max/min
    pheno = df.groupby("grid_id")["ndvi"].agg(
        ndvi_p95=lambda x: x.quantile(0.95),
        ndvi_p05=lambda x: x.quantile(0.05),
        ndvi_median="median",
    ).reset_index()
    pheno["ndvi_amplitude"] = pheno["ndvi_p95"] - pheno["ndvi_p05"]

    # NDVI mùa khô (median, robust)
    dry_med = df[df["is_dry"]].groupby("grid_id")["ndvi"].median().rename("ndvi_dry_median")
    wet_med = df[~df["is_dry"]].groupby("grid_id")["ndvi"].median().rename("ndvi_wet_median")
    pheno = pheno.merge(dry_med, on="grid_id", how="left")
    pheno = pheno.merge(wet_med, on="grid_id", how="left")

    # Drop ratio dựa trên median (robust hơn mean)
    pheno["dry_drop_pct"] = np.where(
        pheno["ndvi_wet_median"] > 0.1,
        (pheno["ndvi_wet_median"] - pheno["ndvi_dry_median"]) / pheno["ndvi_wet_median"] * 100,
        0
    )

    #  Phân loại với 2 tiêu chí 
    # Rừng khộp THỰC SỰ phải:
    #   1. Amplitude > 0.30 (rụng lá rõ rệt)
    #   2. NDVI mùa khô < 0.45 (lá đã rụng nhiều, hoặc
    #      drop > 40% so với mùa mưa)
    #
    # Rừng thường xanh THỰC SỰ phải:
    #   1. NDVI mùa khô > 0.55 (vẫn xanh trong mùa khô)
    #   2. Amplitude < 0.25 (ổn định)

    def classify(row):
        amp = row["ndvi_amplitude"]
        dry = row["ndvi_dry_median"]
        drop = row["dry_drop_pct"]

        # Rừng khộp: cả amplitude cao VÀ NDVI mùa khô thấp
        if amp > 0.30 and (dry < 0.45 or drop > 40):
            return "deciduous"

        # Rừng thường xanh giàu: NDVI mùa khô cao, ổn định
        if dry > 0.60 and amp < 0.25:
            return "evergreen_dense"

        # Rừng thường xanh trung bình
        if dry > 0.50 and amp < 0.35:
            return "evergreen_medium"

        # Rừng nửa rụng lá (semi-deciduous)
        if amp > 0.20 and dry < 0.55:
            return "semi_deciduous"

        return "mixed"

    pheno["phenology_class"] = pheno.apply(classify, axis=1)

    log.info("      Phenology distribution:")
    for cls, n in pheno["phenology_class"].value_counts().items():
        log.info("        %-18s %d grids (%.1f%%)", cls, n, 100 * n / len(pheno))
    log.info("      ndvi_amplitude:    median=%.3f  mean=%.3f",
             pheno["ndvi_amplitude"].median(), pheno["ndvi_amplitude"].mean())
    log.info("      ndvi_dry_median:   median=%.3f  mean=%.3f",
             pheno["ndvi_dry_median"].median(), pheno["ndvi_dry_median"].mean())

    return pheno[["grid_id", "ndvi_amplitude", "ndvi_dry_median",
                  "dry_drop_pct", "phenology_class"]]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: AGGREGATE CLIMATOLOGY
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_climatology(pheno: pd.DataFrame) -> pd.DataFrame:
    log.info("[1/6] Aggregate climatology mùa khô %s → %s", START_DATE, END_DATE)

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
        fire_observed=("fire", "max"),
    ).reset_index()
    agg["rain_monthly"] = agg["rain_daily"] * 30
    agg["fire_observed"] = agg["fire_observed"].fillna(0).astype(int)

    # Merge phenology
    agg = agg.merge(pheno, on="grid_id", how="left")
    agg["phenology_class"] = agg["phenology_class"].fillna("mixed")
    agg["ndvi_amplitude"] = agg["ndvi_amplitude"].fillna(0)

    n_fire = int(agg["fire_observed"].sum())
    log.info("      Aggregated: %d grids", len(agg))
    log.info("      Grids có fire: %d / %d (%.2f%%)", n_fire, len(agg), 100 * n_fire / len(agg))
    log.info("      tmean range:        %.1f – %.1f °C", agg["tmean"].min(), agg["tmean"].max())
    log.info("      rain_monthly range: %.1f – %.1f mm", agg["rain_monthly"].min(), agg["rain_monthly"].max())
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: SCORING — VỚI PHENOLOGY ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
def score_lulc_enhanced(lulc_class, tree_cover, phenology_class, ndvi_amplitude):
    # Non-forest
    if lulc_class in (20, 30, 60): return 4
    if lulc_class == 40:           return 3
    if lulc_class in (50, 80, 90): return 1

    if lulc_class == 10:
        # Rừng khộp xác nhận → cấp 4
        if phenology_class == "deciduous":
            return 4
        # Rừng nửa rụng lá → cấp 3
        if phenology_class == "semi_deciduous":
            return 3
        # Rừng thường xanh giàu → cấp 1
        if phenology_class == "evergreen_dense":
            return 1
        # Rừng thường xanh trung bình → cấp 2
        if phenology_class == "evergreen_medium":
            return 2
        # Mixed → tree_cover fallback
        if tree_cover < 50:   return 3
        elif tree_cover < 75: return 2
        else:                 return 1

    return 2


def score_tmean(x):
    if x > 25:
        return 4
    elif x >= 22:
        return 3
    elif x >= 20:
        return 2
    else:
        return 1


def score_rain(x):
    if x < 5:
        return 4
    elif x < 15:
        return 3
    elif x < 20:
        return 2
    else:
        return 1


def score_wind(x):
    if x > 7:
        return 4
    elif x >= 4.3:
        return 3
    elif x >= 1.4:
        return 2
    else:
        return 1


def score_slope(x):
    if x > 35:
        return 4
    elif x >= 25:
        return 3
    elif x >= 15:
        return 2
    else:
        return 1


def score_aspect(x):
    if 180 <= x < 270:
        return 4
    elif (270 <= x < 315) or (45 <= x < 90):
        return 3
    elif (315 <= x <= 360) or (0 <= x < 45) or (135 <= x < 180):
        return 2
    else:
        return 1


def score_dist_settle(x):
    if x < 0.5:
        return 4
    elif x < 1.0:
        return 3
    elif x < 1.5:
        return 2
    else:
        return 1


def score_dist_forest(x):
    if x < 0.05:
        return 4
    elif x < 0.10:
        return 3
    elif x < 0.15:
        return 2
    else:
        return 1


def weighted_overlay(agg: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/6] Áp dụng scoring functions với phenology")

    agg["s_lulc"] = agg.apply(
        lambda r: score_lulc_enhanced(
            r["lulc_class"], r["tree_cover_pct"],
            r["phenology_class"], r["ndvi_amplitude"]
        ), axis=1
    )
    agg["s_tmean"] = agg["tmean"].apply(score_tmean)
    agg["s_rain"] = agg["rain_monthly"].apply(score_rain)
    agg["s_wind"] = agg["wind"].apply(score_wind)
    agg["s_slope"] = agg["slope"].apply(score_slope)
    agg["s_aspect"] = agg["aspect_deg"].apply(score_aspect)
    agg["s_settle"] = agg["dist_settle"].apply(score_dist_settle)
    agg["s_forest"] = agg["dist_forest_edge"].apply(score_dist_forest)

    log.info("      LULC score distribution:")
    for s in [1, 2, 3, 4]:
        n = (agg["s_lulc"] == s).sum()
        log.info("        score=%d: %d grids (%.1f%%)", s, n, 100 * n / len(agg))

    log.info("[3/6] Tính weighted overlay")
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

    #  Phân cấp theo quantile match với phân bố bài báo 
    # Paper: 35.9% cấp I, 21.3% cấp II, 36.1% cấp III, 6.8% cấp IV
    # Cumulative cutoffs: 0.359, 0.572, 0.932
    PAPER_CUMULATIVE = [0.359, 0.572, 0.932]
    cutoffs = agg["risk_score"].quantile(PAPER_CUMULATIVE).values
    log.info("      Quantile cutoffs: I≤%.3f, II≤%.3f, III≤%.3f, IV>%.3f",
             cutoffs[0], cutoffs[1], cutoffs[2], cutoffs[2])

    agg["risk_level"] = pd.cut(
        agg["risk_score"],
        bins=[-np.inf, cutoffs[0], cutoffs[1], cutoffs[2], np.inf],
        labels=[1, 2, 3, 4]
    ).astype(int)

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4-6: EVALUATE, COMPARE, PLOT (giống script cũ)
# ══════════════════════════════════════════════════════════════════════════════

PAPER_AREA = {1: 35.9, 2: 21.3, 3: 36.1, 4: 6.8}
PAPER_FIRE = {1: 0.0, 2: 24.0, 3: 30.0, 4: 46.0}


def evaluate(agg):
    log.info("[4/6] Đánh giá baseline")

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

    y_true = agg["fire_observed"].values
    y_score = agg["risk_score"].values
    aucpr = average_precision_score(y_true, y_score)
    rocauc = roc_auc_score(y_true, y_score)

    print("\n" + "=" * 65)
    print("BẢNG 3: ML metrics (Weighted Overlay v2)")
    print("=" * 65)
    print(f"  AUC-PR  : {aucpr:.6f}")
    print(f"  ROC-AUC : {rocauc:.6f}")

    return {"aucpr": aucpr, "rocauc": rocauc, "n_fire": int(y_true.sum())}


def compare_with_xgboost(agg, baseline_metrics):
    log.info("[5/6] So sánh với XGBoost")
    if not XGB_PRED.exists():
        return
    xgb = pd.read_parquet(XGB_PRED, columns=["grid_id", "date", "fire_prob"])
    xgb["date"] = pd.to_datetime(xgb["date"])
    xgb = xgb[(xgb["date"] >= START_DATE) & (xgb["date"] <= END_DATE)
              & (xgb["date"].dt.month.isin(DRY_MONTHS))]
    xgb_agg = xgb.groupby("grid_id")["fire_prob"].mean().reset_index()
    xgb_agg.rename(columns={"fire_prob": "xgb_score"}, inplace=True)

    merged = agg[["grid_id", "fire_observed"]].merge(xgb_agg, on="grid_id", how="inner")
    y_true = merged["fire_observed"].values
    y_score = merged["xgb_score"].values
    xgb_aucpr = average_precision_score(y_true, y_score)
    xgb_rocauc = roc_auc_score(y_true, y_score)

    print("\n" + "=" * 65)
    print("BẢNG 4: So sánh Weighted Overlay v2 vs XGBoost")
    print("=" * 65)
    print(f"{'Metric':<15}{'WeightedOverlay':>20}{'XGBoost':>20}{'Δ':>10}")
    print("-" * 65)
    print(
        f"{'AUC-PR':<15}{baseline_metrics['aucpr']:>20.6f}{xgb_aucpr:>20.6f}{xgb_aucpr - baseline_metrics['aucpr']:>+10.4f}")
    print(
        f"{'ROC-AUC':<15}{baseline_metrics['rocauc']:>20.6f}{xgb_rocauc:>20.6f}{xgb_rocauc - baseline_metrics['rocauc']:>+10.4f}")

    pd.DataFrame({
        "method": ["WeightedOverlay_v2", "XGBoost"],
        "aucpr": [baseline_metrics["aucpr"], xgb_aucpr],
        "rocauc": [baseline_metrics["rocauc"], xgb_rocauc],
    }).to_csv(OUT_CSV, index=False)


def plot_map(agg):
    log.info("[6/6] Vẽ bản đồ baseline")
    colors = {1: "#FFEB99", 2: "#FFB347", 3: "#FF6B35", 4: "#C00000"}
    labels = {1: "Cấp I (Thấp)", 2: "Cấp II (TB)", 3: "Cấp III (Cao)", 4: "Cấp IV (Rất cao)"}

    fig, ax = plt.subplots(figsize=(10, 10))
    for lvl in [1, 2, 3, 4]:
        sub = agg[agg["risk_level"] == lvl]
        ax.scatter(sub["lon"], sub["lat"], c=colors[lvl], s=4, label=labels[lvl])
    fire = agg[agg["fire_observed"] == 1]
    ax.scatter(fire["lon"], fire["lat"], facecolors="none", edgecolors="black",
               s=20, linewidths=0.5, label=f"Fire (n={len(fire)})")
    ax.set_xlabel("Longitude");
    ax.set_ylabel("Latitude")
    ax.set_title("Bản đồ nguy cơ cháy rừng — Weighted Overlay v2 (with NDVI Phenology)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_MAP, dpi=150, bbox_inches="tight")


def main():
    _setup_logging(LOG_FILE)
    log.info("Log file: %s", LOG_FILE)

    with open(LOG_FILE, "a", encoding="utf-8") as _f:
        sys.stdout = _Tee(_f)
        try:
            pheno = compute_phenology()
            agg = aggregate_climatology(pheno)
            agg = weighted_overlay(agg)
            metrics = evaluate(agg)
            compare_with_xgboost(agg, metrics)
            plot_map(agg)
            agg.to_parquet(OUT_PARQUET, index=False)
            log.info("Done! Log saved to: %s", LOG_FILE)
        finally:
            sys.stdout = sys.stdout._stdout


if __name__ == "__main__":
    main()