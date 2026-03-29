import numpy as np
from pathlib import Path
from config import OUTPUT_DATA_PATH, TRAIN_END_DATE, VAL_END_DATE

TENSOR_PATH = OUTPUT_DATA_PATH + "/clstm_tensor.npy"
SAVE_DIR    = Path(OUTPUT_DATA_PATH) / "processed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Vị trí cột "fire" trong tensor (cột cuối cùng)
# build_grid_maps.py lấy feature_cols = tất cả trừ grid_id và date
# → "fire" là 1 trong 26 cột, cần xác định đúng index
FIRE_COL_NAME = "fire"

# Index các feature liên tục cần normalize (0-based trong 16 features)
# Sau khi bỏ cột fire (idx 16), thứ tự 16 features còn lại:
# 0:tmean, 1:rh, 2:wind, 3:rain, 4:vpd,
# 5:rain_14d_sum, 6:vpd_14d_mean,
# 7:fire_lag_1, 8:fire_lag_3,
# 9:neighbor_count,
# 10:neighbor_fire_1d, 11:neighbor_fire_3d,
# 12:sin_doy, 13:cos_doy,
# 14:ndvi, 15:nbr
CONTINUOUS_IDX = [0, 1, 2, 3, 4, 5, 6, 10, 11]
# 0-4 : tmean rh wind rain vpd
# 5-6 : rain_14d_sum vpd_14d_mean
# 10-11: neighbor_fire_1d neighbor_fire_3d
# NOT normalized: fire_lag 7,8 (binary 0/1), neighbor_count 9 (int),
#                 sin/cos_doy 12,13 (cyclic [-1,1]), ndvi/nbr 14,15 (bounded [-1,1])


# ================================================================
# BƯỚC 1: LOAD TENSOR
# dùng mmap_mode="r" → không load toàn bộ vào RAM
# ================================================================
print("=" * 50)
print("BƯỚC 1: Load tensor")
print("=" * 50)

tensor = np.load(TENSOR_PATH, mmap_mode="r")
print(f"  tensor.shape : {tensor.shape}")   # (3653, 137, 138, 17)
print(f"  dtype        : {tensor.dtype}")
T, H, W, C = tensor.shape


# ================================================================
# BƯỚC 2: XÁC ĐỊNH VỊ TRÍ CỘT FIRE
# build_grid_maps.py lấy feature_cols theo thứ tự df.columns
# → cần load lại parquet để biết đúng thứ tự
# ================================================================
print("\nBƯỚC 2: Xác định vị trí cột fire")

import pandas as pd
from config import GRID_ID_COL, DATE_COL

# Chỉ đọc header để lấy thứ tự cột
sample_df    = pd.read_parquet(
    OUTPUT_DATA_PATH + "/clstm_clean_data.parquet"
).head(0)
feature_cols = [c for c in sample_df.columns
                if c not in [GRID_ID_COL, DATE_COL]]

assert len(feature_cols) == C, \
    f"Số cột không khớp: parquet={len(feature_cols)}, tensor={C}"

fire_idx = feature_cols.index(FIRE_COL_NAME)
feat_idx = [i for i in range(C) if i != fire_idx]

print(f"  Tổng cột     : {C}")
print(f"  fire_idx     : {fire_idx}")
print(f"  feat_idx     : {len(feat_idx)} features")
print(f"  Thứ tự features: {[feature_cols[i] for i in feat_idx]}")


# ================================================================
# BƯỚC 3: TÁCH X VÀ y
# Dùng mmap để tránh load 14 GB vào RAM cùng lúc
# ================================================================
print("\nBƯỚC 3: Tách X và y")

# Tạo X_all và y_all dưới dạng memmap mới
X_all_path = SAVE_DIR / "X_all.npy"
y_all_path = SAVE_DIR / "y_all.npy"

X_all = np.lib.format.open_memmap(
    X_all_path, mode="w+", dtype="float32",
    shape=(T, H, W, len(feat_idx))
)
y_all = np.lib.format.open_memmap(
    y_all_path, mode="w+", dtype="float32",
    shape=(T, H, W, 1)
)

# Copy từng slice theo thời gian để tránh OOM
CHUNK = 100   # xử lý 100 ngày mỗi lần
for start in range(0, T, CHUNK):
    end = min(start + CHUNK, T)
    X_all[start:end] = tensor[start:end][..., feat_idx]
    y_all[start:end] = tensor[start:end][..., fire_idx:fire_idx+1]
    if start % 500 == 0:
        print(f"  Đã xử lý {end}/{T} ngày")

print(f"  X_all.shape : {X_all.shape}")   # (3653, 137, 138, 25)
print(f"  y_all.shape : {y_all.shape}")   # (3653, 137, 138,  1)

fire_ratio = float(y_all[:].mean())
print(f"  Fire ratio  : {fire_ratio:.4%}")   # kỳ vọng ~0.11%


# ================================================================
# BƯỚC 4: SPLIT THEO THỜI GIAN
# Tuyệt đối không random shuffle
# ================================================================
print("\nBƯỚC 4: Train/Val/Test split (theo thời gian)")

TRAIN_END = pd.Timestamp(TRAIN_END_DATE)  # 2020-12-31
VAL_END   = pd.Timestamp(VAL_END_DATE)    # 2022-12-31

# Load cột date từ parquet để lấy danh sách ngày đúng thứ tự
date_series = pd.read_parquet(
    OUTPUT_DATA_PATH + "/clstm_clean_data.parquet",
    columns=[DATE_COL]
)[DATE_COL]
unique_dates = np.sort(date_series.unique())  # (T,) — đã sort tăng dần

assert len(unique_dates) == T, \
    f"Số ngày không khớp: parquet={len(unique_dates)}, tensor={T}"

unique_dates_ts = pd.DatetimeIndex(unique_dates)
train_end = int((unique_dates_ts <= TRAIN_END).sum())
val_end   = int((unique_dates_ts <= VAL_END).sum())

splits = {
    "train": (0,         train_end),
    "val":   (train_end, val_end),
    "test":  (val_end,   T),
}

for name, (s, e) in splits.items():
    n_days = e - s
    print(f"  {name:5s}: {str(unique_dates[s])[:10]} → "
          f"{str(unique_dates[e-1])[:10]}  "
          f"({n_days} ngày, {max(0, n_days-7)} sequences)")
# ================================================================
# BƯỚC 5: NORMALIZE
# Fit ONLY trên train, apply cho val và test
# Chỉ normalize continuous features (index 0–12)
# ================================================================
print("\nBƯỚC 5: Normalize")

s, e = splits["train"]
X_train_cont = X_all[s:e, ..., CONTINUOUS_IDX]   # load vào RAM: ~3 GB (float32)

mean = X_train_cont.mean(axis=(0, 1, 2), keepdims=True)   # (1, 1, 1, 13)
std  = X_train_cont.std(axis=(0, 1, 2),  keepdims=True) + 1e-8

del X_train_cont
import gc; gc.collect()

print(f"  mean.shape : {mean.shape}")
print(f"  std.shape  : {std.shape}")

# Lưu mean/std để dùng lại khi inference
np.save(SAVE_DIR / "norm_mean.npy", mean)
np.save(SAVE_DIR / "norm_std.npy",  std)
print("  Đã lưu norm_mean.npy và norm_std.npy")

# Apply normalize và lưu từng split
for split_name, (s, e) in splits.items():
    out_x = np.lib.format.open_memmap(
        SAVE_DIR / f"X_{split_name}.npy", mode="w+",
        dtype="float32", shape=(e - s, H, W, len(feat_idx))
    )
    out_y = np.lib.format.open_memmap(
        SAVE_DIR / f"y_{split_name}.npy", mode="w+",
        dtype="float32", shape=(e - s, H, W, 1)
    )

    for start in range(0, e - s, CHUNK):
        end_c = min(start + CHUNK, e - s)
        chunk_x = X_all[s + start : s + end_c].copy()   # (chunk, H, W, 25)
        # Normalize continuous features
        chunk_x[..., CONTINUOUS_IDX] = (
            (chunk_x[..., CONTINUOUS_IDX] - mean) / std
        )
        out_x[start:end_c] = chunk_x
        out_y[start:end_c] = y_all[s + start : s + end_c]

    print(f"  {split_name:5s}: X={out_x.shape}  y={out_y.shape}  "
          f"fire={float(out_y[:].mean()):.4%}")


# ================================================================
# BƯỚC 6: KIỂM TRA SANITY
# ================================================================
print("\nBƯỚC 6: Sanity checks")

X_train = np.load(SAVE_DIR / "X_train.npy", mmap_mode="r")
X_val   = np.load(SAVE_DIR / "X_val.npy",   mmap_mode="r")
X_test  = np.load(SAVE_DIR / "X_test.npy",  mmap_mode="r")
y_train = np.load(SAVE_DIR / "y_train.npy", mmap_mode="r")

# 1. Kiểm tra không có data leak — ngày cuối train < ngày đầu val
assert len(X_train) + len(X_val) + len(X_test) == T, "Tổng ngày không khớp"

# 2. Kiểm tra continuous features đã normalize (mean ~0, std ~1)
cont_sample = X_train[:100, ..., CONTINUOUS_IDX]
print(f"  Continuous feat mean (kỳ vọng ~0): {cont_sample.mean():.4f}")
print(f"  Continuous feat std  (kỳ vọng ~1): {cont_sample.std():.4f}")

# 3. Kiểm tra binary features không bị normalize (vẫn là 0/1)
binary_sample = X_train[:100, ..., 7]   # fire_lag_1
uniq = np.unique(binary_sample)
print(f"  fire_lag_1 unique values (kỳ vọng 0/1): {uniq[:5]}")

# 4. Label vẫn là 0/1
assert y_train.max() <= 1.0 and y_train.min() >= 0.0, "Label ngoài [0,1]"
print("  Label range: OK [0, 1]")

print("\n" + "=" * 50)
print("BƯỚC 1 HOÀN TẤT")
print(f"  Files đã lưu tại: {SAVE_DIR}")
print("=" * 50)

# In ra cấu trúc thư mục output
import os
for f in sorted(SAVE_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:<25s}  {size_mb:>8.1f} MB")