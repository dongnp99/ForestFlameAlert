"""
build_ensemble_dataset.py
=========================
Căn chỉnh predictions của ConvLSTM và XGBoost v5 theo (date, grid_id),
tạo 2 file parquet dùng để train/evaluate meta-learner:

  ensemble_val.parquet   — 2022-01-08 → 2022-12-31  (~6.7M rows)
  ensemble_test.parquet  — 2023-01-08 → 2024-12-31  (~13.6M rows)

Columns:
  date        — ngày
  grid_id     — ID lưới không gian
  lon, lat    — tọa độ
  fire        — ground truth (0/1)
  xgb_prob    — xác suất từ XGBoost v5
  clstm_prob  — xác suất từ ConvLSTM
  month       — tháng (1–12, cho meta-learner)
  sin_doy     — sin(day_of_year / 365 * 2π)
  cos_doy     — cos(day_of_year / 365 * 2π)

Cách align:
  ConvLSTM pixel k của ngày i → grid_id = sorted_grid_ids[k]
  sorted_grid_ids = np.sort(unique grid_ids từ dataset parquet)
  → dùng np.tile + date.repeat để build CLSTM DataFrame → merge với XGB
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

#  Paths ─
HERE = Path(__file__).parent
ROOT = HERE.parent

CLSTM_VAL_NPY  = HERE / "clstm/prob_map_val.npy"
CLSTM_TEST_NPY = HERE / "clstm/prob_map_test.npy"
XGB_VAL_PAR    = HERE / "xgboost/val_predictions_map.parquet"
XGB_TEST_PAR   = HERE / "xgboost/test_predictions_map.parquet"
DATA_PATH      = ROOT / "data/Daklak/final_inputs/daklak_final_dataset_v3_pathways.parquet"

OUT_VAL  = HERE / "ensemble_val.parquet"
OUT_TEST = HERE / "ensemble_test.parquet"

#  Constants ─
H, W    = 137, 138
SEQ_LEN = 7

VAL_START  = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2023-01-01")

FIRST_PRED_VAL  = VAL_START  + pd.Timedelta(days=SEQ_LEN)   # 2022-01-08
FIRST_PRED_TEST = TEST_START + pd.Timedelta(days=SEQ_LEN)   # 2023-01-08


# ================================================================
# HELPERS
# ================================================================

def load_sorted_grid_ids() -> np.ndarray:
    """
    Đọc sorted grid_ids từ dataset parquet.
    Thứ tự này phải trùng với cách ConvLSTM map pixel → grid khi build input.
    """
    print("  Đọc sorted grid_id mapping từ dataset...")
    df = pd.read_parquet(DATA_PATH, columns=["grid_id"], engine="pyarrow")
    ids = np.sort(df["grid_id"].unique())
    del df
    print(f"  → {len(ids):,} unique grid_ids")
    return ids


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm month, sin_doy, cos_doy để meta-learner dùng."""
    doy = df["date"].dt.day_of_year.astype(np.float32)
    df["month"]   = df["date"].dt.month.astype(np.int8)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)
    return df


# ================================================================
# CORE: BUILD ALIGNED SPLIT
# ================================================================

def build_split(
    split_name: str,
    clstm_npy_path: Path,
    xgb_parquet_path: Path,
    first_pred_date: pd.Timestamp,
    sorted_grid_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Xây dựng DataFrame căn chỉnh cho một split (val hoặc test).

    Logic:
      1. ConvLSTM npy → reshape (n_days, H*W), lấy [:, :n_valid]
      2. Build (date, grid_id, clstm_prob) bằng repeat + tile
      3. Merge với XGBoost parquet theo (date, grid_id)

    Chỉ giữ ngày từ first_pred_date trở đi (CLSTM cần seq_len ngày đầu).
    """
    print(f"\n{'='*55}")
    print(f"Build split: {split_name}  (first pred: {first_pred_date.date()})")
    print(f"{'='*55}")

    n_valid = len(sorted_grid_ids)

    #  Load ConvLSTM ─
    print("  Load ConvLSTM npy...")
    clstm_preds = np.load(clstm_npy_path)   # (n_days * H * W,)
    n_days = len(clstm_preds) // (H * W)
    print(f"  → {n_days} ngày × {H*W} pixels  ({len(clstm_preds):,} values)")

    date_range = pd.date_range(first_pred_date, periods=n_days, freq="D")

    # Reshape → (n_days, n_valid): chỉ lấy pixel có grid_id thực
    clstm_2d = (
        clstm_preds[: n_days * H * W]
        .reshape(n_days, H * W)[:, :n_valid]
        .astype(np.float32)
    )   # (n_days, n_valid)

    #  Build CLSTM DataFrame ─
    print(f"  Build CLSTM DataFrame ({n_days * n_valid:,} rows)...")
    dates_col  = date_range.repeat(n_valid)             # d1,d1,...,d2,d2,...
    grid_col   = np.tile(sorted_grid_ids, n_days)       # g1..gn,g1..gn,...
    prob_col   = clstm_2d.flatten()

    clstm_df = pd.DataFrame({
        "date":       dates_col,
        "grid_id":    grid_col.astype(np.int32),
        "clstm_prob": prob_col,
    })
    del clstm_2d, clstm_preds, dates_col, grid_col, prob_col

    #  Load XGBoost 
    print("  Load XGBoost parquet...")
    xgb_df = pd.read_parquet(
        xgb_parquet_path,
        columns=["date", "grid_id", "fire_prob", "fire", "lon", "lat"],
        engine="pyarrow",
    )
    xgb_df["date"] = pd.to_datetime(xgb_df["date"])
    xgb_df = xgb_df[xgb_df["date"] >= first_pred_date].reset_index(drop=True)
    xgb_df.rename(columns={"fire_prob": "xgb_prob"}, inplace=True)
    print(f"  → {len(xgb_df):,} rows  ({xgb_df['date'].nunique()} ngày)")

    #  Merge ─
    print("  Merge theo (date, grid_id)...")
    merged = xgb_df.merge(clstm_df, on=["date", "grid_id"], how="inner")
    del xgb_df, clstm_df

    print(f"  → Sau merge: {len(merged):,} rows")

    # Kiểm tra coverage
    n_matched = merged["date"].nunique()
    pct = 100 * len(merged) / (n_days * n_valid)
    print(f"  Coverage: {n_matched} ngày,  {pct:.1f}% pixels matched")

    #  Temporal features ─
    merged = _add_temporal_features(merged)

    #  Sắp xếp cột 
    merged = merged[[
        "date", "grid_id", "lon", "lat", "fire",
        "xgb_prob", "clstm_prob",
        "month", "sin_doy", "cos_doy",
    ]].sort_values(["date", "grid_id"]).reset_index(drop=True)

    #  Stats ─
    fire_rate = merged["fire"].mean()
    n_fire    = merged["fire"].sum()
    print(f"\n  Fire rate : {fire_rate:.4%}  ({n_fire:,} fire pixels)")
    print(f"  xgb_prob  : mean={merged['xgb_prob'].mean():.4f}  "
          f"max={merged['xgb_prob'].max():.4f}")
    print(f"  clstm_prob: mean={merged['clstm_prob'].mean():.4f}  "
          f"max={merged['clstm_prob'].max():.4f}")

    return merged


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("BUILD ENSEMBLE DATASET")
    print("=" * 55)

    sorted_grid_ids = load_sorted_grid_ids()

    #  Val split ─
    val_df = build_split(
        split_name      = "val",
        clstm_npy_path  = CLSTM_VAL_NPY,
        xgb_parquet_path= XGB_VAL_PAR,
        first_pred_date = FIRST_PRED_VAL,
        sorted_grid_ids = sorted_grid_ids,
    )
    print(f"\n  Lưu → {OUT_VAL}")
    val_df.to_parquet(OUT_VAL, engine="pyarrow", index=False)
    del val_df

    #  Test split 
    test_df = build_split(
        split_name      = "test",
        clstm_npy_path  = CLSTM_TEST_NPY,
        xgb_parquet_path= XGB_TEST_PAR,
        first_pred_date = FIRST_PRED_TEST,
        sorted_grid_ids = sorted_grid_ids,
    )
    print(f"\n  Lưu → {OUT_TEST}")
    test_df.to_parquet(OUT_TEST, engine="pyarrow", index=False)
    del test_df

    print("\n" + "=" * 55)
    print("HOÀN TẤT")
    print(f"  {OUT_VAL}")
    print(f"  {OUT_TEST}")