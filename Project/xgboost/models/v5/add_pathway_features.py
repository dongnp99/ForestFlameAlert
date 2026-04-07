"""
add_pathway_features.py
========================
Bước 1 của Option C: Thêm pathway-aware features vào dataset.

Tạo 5 features mới:
  - is_pathway_p1       (int8)  : post-harvest agricultural burning
  - is_pathway_p2       (int8)  : deforestation + fire history
  - is_pathway_p3       (int8)  : repeat ignition during burn season
  - human_pathway_score (int8)  : tổng số pathway matched (0-3)
  - human_signal_strength (float32) : soft continuous signal từ normalized features

Các features này cho model BIẾT CONTEXT để tự học cách xử lý khác nhau
cho cháy tự nhiên vs nhân tạo, thay vì ép qua weighting.

Input : daklak_final_dataset_v2_human.parquet (original)
Output: daklak_final_dataset_v3_pathways.parquet (original + 5 features mới)

Chạy: python add_pathway_features.py
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Sửa lại cho đúng cấu trúc thư mục của bạn
INPUT_PATH  = "../../../data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet"
OUTPUT_PATH = "../../../data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"

# ── Chunk size (số dòng mỗi lần xử lý — giảm nếu OOM) ───────────────────────
CHUNK_SIZE = 5_000_000

# ── Các cột cần để tính pathway features ──────────────────────────────────────
PATHWAY_DEPS = [
    "burn_season_flag",
    "days_since_harvest",
    "deforestation_lag_1y",
    "fire_count_prev_year",
]

# Các cột dùng cho human_signal_strength (soft signal)
SOFT_SIGNAL_COLS = [
    "burn_season_flag",
    "days_since_harvest",
    "deforestation_lag_1y",
    "fire_count_prev_year",
    "dist_road_km",
    "dist_settlement_km",
    "nightlight_mean",
    "pop_density",
    "cropland_frac_1km",
]

# ══════════════════════════════════════════════════════════════════════════════
# PATHWAY DEFINITIONS (V4 — phải khớp với xgb_config.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pathway_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 5 pathway-aware features vào df (in-place columns)."""

    # ── Binary pathway indicators ──────────────────────────────────────────
    # P1: post-harvest agricultural burning
    df["is_pathway_p1"] = (
        (df["burn_season_flag"] == 1)
        & (df["days_since_harvest"] < 30)
    ).astype("int8")

    # P2: deforestation with fire history
    df["is_pathway_p2"] = (
        (df["deforestation_lag_1y"] > 1.5)
        & (df["fire_count_prev_year"] > 0)
    ).astype("int8")

    # P3: repeat ignition during burn season
    df["is_pathway_p3"] = (
        (df["fire_count_prev_year"] > 1)
        & (df["burn_season_flag"] == 1)
    ).astype("int8")

    # ── Aggregate score ────────────────────────────────────────────────────
    df["human_pathway_score"] = (
        df["is_pathway_p1"] + df["is_pathway_p2"] + df["is_pathway_p3"]
    ).astype("int8")

    # ── Soft human signal (continuous, 0–1 range) ─────────────────────────
    # Chuẩn hóa từng feature về [0, 1] rồi lấy trung bình.
    # Hướng: cao = nhiều tín hiệu nhân tạo hơn.
    signal = np.zeros(len(df), dtype="float32")
    n_components = 0

    # burn_season_flag: đã binary
    signal += df["burn_season_flag"].values.astype("float32")
    n_components += 1

    # days_since_harvest: nhỏ hơn = gần thu hoạch hơn = nhân tạo hơn
    # Chuẩn hóa: 0 ngày → 1.0, 365 ngày → 0.0
    dsh = df["days_since_harvest"].values.astype("float32")
    signal += np.clip(1.0 - dsh / 365.0, 0.0, 1.0)
    n_components += 1

    # deforestation_lag_1y: cao hơn = nhân tạo hơn
    # Clip tại percentile 99 để tránh outlier chi phối
    defor = df["deforestation_lag_1y"].values.astype("float32")
    defor_cap = np.percentile(defor[~np.isnan(defor)], 99)
    signal += np.clip(defor / max(defor_cap, 1e-6), 0.0, 1.0)
    n_components += 1

    # fire_count_prev_year: cao hơn = nhân tạo hơn
    fc = df["fire_count_prev_year"].values.astype("float32")
    fc_cap = np.percentile(fc[~np.isnan(fc)], 99)
    signal += np.clip(fc / max(fc_cap, 1e-6), 0.0, 1.0)
    n_components += 1

    # dist_road_km: nhỏ hơn = gần đường = nhân tạo hơn
    # Lưu ý: đơn lẻ anti-selective, nhưng kết hợp với các signal khác
    # vẫn có giá trị. Soft signal không có hard threshold nên ổn.
    if "dist_road_km" in df.columns:
        dr = df["dist_road_km"].values.astype("float32")
        dr_cap = np.percentile(dr[~np.isnan(dr)], 99)
        signal += np.clip(1.0 - dr / max(dr_cap, 1e-6), 0.0, 1.0)
        n_components += 1

    # dist_settlement_km: nhỏ hơn = gần khu dân cư = nhân tạo hơn
    if "dist_settlement_km" in df.columns:
        ds = df["dist_settlement_km"].values.astype("float32")
        ds_cap = np.percentile(ds[~np.isnan(ds)], 99)
        signal += np.clip(1.0 - ds / max(ds_cap, 1e-6), 0.0, 1.0)
        n_components += 1

    # nightlight_mean: cao hơn = đô thị hóa = nhân tạo hơn
    if "nightlight_mean" in df.columns:
        nl = df["nightlight_mean"].values.astype("float32")
        nl_cap = np.percentile(nl[~np.isnan(nl)], 99)
        signal += np.clip(nl / max(nl_cap, 1e-6), 0.0, 1.0)
        n_components += 1

    # pop_density: cao hơn = đông dân = nhân tạo hơn
    if "pop_density" in df.columns:
        pd_vals = df["pop_density"].values.astype("float32")
        pd_cap = np.percentile(pd_vals[~np.isnan(pd_vals)], 99)
        signal += np.clip(pd_vals / max(pd_cap, 1e-6), 0.0, 1.0)
        n_components += 1

    # cropland_frac_1km: cao hơn = nông nghiệp = nhân tạo hơn
    if "cropland_frac_1km" in df.columns:
        cf = df["cropland_frac_1km"].values.astype("float32")
        signal += cf  # đã trong [0, 1]
        n_components += 1

    # Trung bình → [0, 1]
    df["human_signal_strength"] = (signal / n_components).astype("float32")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Chunked processing
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Reading input: %s", INPUT_PATH)

    # Đọc toàn bộ (nếu RAM đủ) hoặc chunked
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Kiểm tra cột dependencies
    missing = [c for c in PATHWAY_DEPS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Tính pathway features
    log.info("Computing pathway features...")
    df = compute_pathway_features(df)

    # Verify
    new_cols = ["is_pathway_p1", "is_pathway_p2", "is_pathway_p3",
                "human_pathway_score", "human_signal_strength"]
    for col in new_cols:
        log.info("  %-25s  dtype=%-8s  mean=%.4f  min=%.4f  max=%.4f",
                 col, str(df[col].dtype), df[col].mean(),
                 df[col].min(), df[col].max())

    # Thống kê trên fire=1
    fire_mask = df["fire"] == 1
    log.info("--- Stats on fire=1 samples ---")
    for col in new_cols:
        log.info("  %-25s  mean=%.4f", col, df.loc[fire_mask, col].mean())

    # Save
    log.info("Saving to: %s", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="zstd", index=False)
    log.info("Done. Output: %s  (%d rows, %d columns)",
             OUTPUT_PATH, len(df), len(df.columns))


if __name__ == "__main__":
    main()