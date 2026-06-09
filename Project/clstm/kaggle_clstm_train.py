"""
kaggle_clstm_train.py — ConvLSTM Wildfire Full Pipeline (Kaggle)
================================================================
Dataset : daklak_final_dataset_v3_pathways.parquet  (same scope as XGBoost v5)
Features: 52 features (weather, terrain, vegetation, human activity, pathway)
Target  : fire (binary 0/1, ~0.11% positive)

Improvements vs local version:
  - 52 features matching XGBoost v5 scope (từ 16 → 52)
  - Composite loss: FocalLoss + TverskyLoss  (tăng Recall trực tiếp)
  - Skip connection ConvLSTM-1 → Decoder    (multi-scale context)
  - torch.compile() với inductor backend     (fix bug backend="eager")
  - NUM_WORKERS=2 + persistent_workers       (Linux-compatible trên Kaggle)
  - Gradient accumulation (effective batch = BATCH_SIZE × ACCUM_STEPS)
  - Direct split building — không lưu tensor trung gian (~13 GB tổng disk)

Output (/kaggle/working/):
  best_model_<run_id>.pt   — checkpoint tốt nhất
  history_<run_id>.csv     — loss/metrics mỗi epoch
  summary_<run_id>.txt     — kết quả cuối
  loss_curve_<run_id>.png

Cách dùng:
  1. Add dataset daklak_final_dataset_v3_pathways.parquet vào notebook
  2. Run script này (có thể paste vào notebook cell hoặc chạy !python kaggle_clstm_train.py)
"""

# ================================================================
# IMPORTS
# ================================================================
import gc
import csv
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, confusion_matrix,
)


# ================================================================
# ═══════════════════  CẤU HÌNH  ═══════════════════════════════
# ================================================================

#  Paths 
DATA_PATH  = "/kaggle/input/datasets/flyingcow781/forestflameprediction/daklak_final_dataset_v3_pathways.parquet"
OUTPUT_DIR = Path("/kaggle/working")
SPLIT_DIR  = OUTPUT_DIR / "splits"

GRID_ID_COL = "grid_id"
DATE_COL    = "date"
TARGET_COL  = "fire"

#  Time split ─
TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"

#  Features (28 — tối ưu cho ConvLSTM) ─
# Loại bỏ so với XGBoost v5:
#   - Pathway features (5): derived từ features khác, ConvLSTM tự học
#   - XGBoost interactions (3): vpd_neighbor_1d, vpd_fire_lag_1, neighbor_fire_7d
#   - Rolling 30d (2): redundant với 14d + 7-day sequence window
#   - Vegetation chồng lặp (4): ndwi, ndii, ndvi_std, has_s2 (has_s2 gây nhiễu)
#   - Fire history dư (2): fire_count_prev_3y, fire_freq_5y
#   - Human activity overlap (5): lulc_class (categorical!), cropland_frac_1km,
#                                  dist_powerline_km, nightlight_mean, deforestation_lag_1y
#   - Terrain phụ (2): dem_stdev, slp_stdev
FEATURE_COLS = [
    # Weather dynamic (7)
    "tmean", "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "vpd_14d_mean",

    # Fire lag (2)
    "fire_lag_1", "fire_lag_3",

    # Spatial fire spread (3) — neighbor_fire_7d bỏ vì seq_len=7 đã cover
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d",

    # Seasonality (2)
    "sin_doy", "cos_doy",

    # Vegetation core (4) — giữ ndvi, nbr và delta; bỏ ndwi/ndii/ndvi_std/has_s2
    "ndvi", "nbr", "delta_ndvi_14d", "delta_nbr_7d",

    # Fire history temporal (4)
    "days_since_last_fire", "burn_season_flag",
    "days_since_harvest", "fire_count_prev_year",

    # Quasi-static spatial context (6) — terrain + human cốt lõi
    "dem_mean", "slp_mean",
    "dist_road_km", "dist_settlement_km",
    "tree_cover_pct", "pop_density",
]

N_FEAT = len(FEATURE_COLS)   # 28

# Features KHÔNG normalize (binary, cyclic, đã bounded [-1,1])
_NO_NORM = {
    "fire_lag_1", "fire_lag_3",       # binary 0/1
    "sin_doy", "cos_doy",             # cyclic [-1, 1]
    "ndvi", "nbr",                    # bounded [-1, 1]
    "burn_season_flag",               # binary 0/1
}
CONTINUOUS_IDX = [i for i, f in enumerate(FEATURE_COLS) if f not in _NO_NORM]

#  Grid dimensions 
# Sẽ được tính tự động từ data
H, W = None, None   # (137, 138) — filled in build_splits()

#  Model config ─
MODEL_CFG = {
    "seq_len":  7,
    "n_feat":   N_FEAT,      # 28
    "filters":  [64, 32],
    "kernel":   3,
    "dropout":  0.4,
    # FocalLoss params
    "fl_gamma": 2.0,
    "fl_alpha": 0.80,
    # TverskyLoss params  (beta > alpha → penalize FN harder)
    "tv_alpha": 0.3,
    "tv_beta":  0.7,
    # Composite weight
    "loss_focal_w": 0.5,
    "loss_tversky_w": 0.5,
    # Optimizer
    "lr":       3e-4,
    "clipnorm": 1.0,
    "weight_decay": 1e-4,
}

#  Training config 
TRAIN_CFG = {
    "epochs":      200,
    "batch_size":  4,       # nhỏ hơn vì tensor lớn hơn (52 features)
    "accum_steps": 4,       # effective batch = 4 × 4 = 16
    "patience":    30,
    "lr_patience": 8,
    "lr_factor":   0.5,
    "min_lr":      1e-6,
    "num_workers": 2,       # Linux-compatible trên Kaggle
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")


# ================================================================
# ══════════════  BƯỚC 1: BUILD SPATIAL SPLITS  ════════════════
# Xây dựng X/y splits trực tiếp từ parquet (không lưu tensor trung gian)
# Tiết kiệm ~15 GB disk so với cách cũ
# ================================================================

def build_splits(data_path: str, output_dir: Path):
    """
    Load parquet → chia train/val/test theo thời gian
    → build spatial grid (H, W, C) cho mỗi ngày
    → lưu X_{split}.npy và y_{split}.npy dạng memmap

    Returns
    -------
    H, W : int — kích thước lưới không gian
    unique_dates_per_split : dict — danh sách ngày của từng split
    """
    global H, W

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BƯỚC 1: Build spatial splits từ parquet")
    print("=" * 60)

    #  Load data (chỉ load cột cần thiết) ─
    load_cols = [GRID_ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    print(f"\nLoading {data_path}...")
    df = pd.read_parquet(data_path, columns=load_cols, engine="pyarrow")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, GRID_ID_COL]).reset_index(drop=True)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Fire rate: {df[TARGET_COL].mean():.4%}")

    #  Xác định grid layout ─
    grid_ids = np.sort(df[GRID_ID_COL].unique())
    grid_count = len(grid_ids)
    H = int(np.sqrt(grid_count))
    W = int(np.ceil(grid_count / H))
    grid_index = {gid: i for i, gid in enumerate(grid_ids)}
    print(f"\n  Grids: {grid_count}  →  H={H}, W={W}")

    #  Tách split theo thời gian ─
    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end   = pd.Timestamp(VAL_END_DATE)

    all_dates = np.sort(df[DATE_COL].unique())
    dates_ts  = pd.DatetimeIndex(all_dates)

    split_dates = {
        "train": all_dates[dates_ts <= train_end],
        "val":   all_dates[(dates_ts > train_end) & (dates_ts <= val_end)],
        "test":  all_dates[dates_ts > val_end],
    }
    for name, dates in split_dates.items():
        print(f"  {name:5s}: {str(dates[0])[:10]} → {str(dates[-1])[:10]}"
              f"  ({len(dates)} ngày)")

    #  Build từng split 
    n_feat = len(FEATURE_COLS)

    for split_name, dates in split_dates.items():
        x_path = output_dir / f"X_{split_name}.npy"
        y_path = output_dir / f"y_{split_name}.npy"

        if x_path.exists() and y_path.exists():
            print(f"\n  [{split_name}] Files đã tồn tại — skip")
            continue

        T = len(dates)
        print(f"\n  [{split_name}] Building {T} days → {T - MODEL_CFG['seq_len']} sequences...")

        X = np.lib.format.open_memmap(
            x_path, mode="w+", dtype="float32", shape=(T, H, W, n_feat)
        )
        y = np.lib.format.open_memmap(
            y_path, mode="w+", dtype="float32", shape=(T, H, W, 1)
        )

        # Dùng pd.Timestamp làm key vì groupby trả về Timestamp, không phải numpy.datetime64
        date_to_t = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        date_set  = set(pd.Timestamp(d) for d in dates)

        df_split = df[df[DATE_COL].isin(date_set)]
        grouped  = df_split.groupby(DATE_COL)

        for date, day_df in grouped:
            t = date_to_t[date]
            day_df = day_df.fillna(0)

            gidx = day_df[GRID_ID_COL].map(grid_index).to_numpy()
            rows = gidx // W
            cols = gidx % W

            X[t][rows, cols] = day_df[FEATURE_COLS].to_numpy(dtype=np.float32)
            y[t][rows, cols, 0] = day_df[TARGET_COL].to_numpy(dtype=np.float32)

        fire_rate = float(y[:].mean())
        print(f"    X={X.shape}  fire_rate={fire_rate:.4%}")

    del df
    gc.collect()

    print(f"\n  Splits saved → {output_dir}")
    return H, W, split_dates


# ================================================================
# ══════════════  BƯỚC 2: NORMALIZE  ════════════════════════════
# Fit ONLY trên train, apply cho val và test
# ================================================================

def normalize_splits(split_dir: Path):
    """
    Normalize continuous features (z-score).
    Fit trên train, apply cho tất cả splits.
    Lưu norm_mean.npy và norm_std.npy để dùng lại khi inference.
    """
    print("\n" + "=" * 60)
    print("BƯỚC 2: Normalize")
    print("=" * 60)

    mean_path = split_dir / "norm_mean.npy"
    std_path  = split_dir / "norm_std.npy"

    X_train = np.load(split_dir / "X_train.npy", mmap_mode="r+")

    if not mean_path.exists():
        print(f"\n  Fit mean/std trên train ({len(CONTINUOUS_IDX)} continuous features)...")
        cont = X_train[:, ..., CONTINUOUS_IDX]
        mean = cont.mean(axis=(0, 1, 2), keepdims=True)
        std  = cont.std(axis=(0, 1, 2),  keepdims=True) + 1e-8
        np.save(mean_path, mean)
        np.save(std_path,  std)
        print(f"  Saved norm_mean.npy + norm_std.npy")
    else:
        mean = np.load(mean_path)
        std  = np.load(std_path)
        print("  Loaded existing norm_mean/std")

    # Apply normalize (chunk để tránh OOM)
    CHUNK = 50
    for split_name in ["train", "val", "test"]:
        x_path = split_dir / f"X_{split_name}.npy"
        X = np.load(x_path, mmap_mode="r+")
        T = len(X)

        print(f"\n  Normalizing {split_name} ({T} days)...")
        for start in range(0, T, CHUNK):
            end = min(start + CHUNK, T)
            chunk = X[start:end]                             # copy to RAM
            chunk[..., CONTINUOUS_IDX] = (
                (chunk[..., CONTINUOUS_IDX] - mean) / std
            )
            X[start:end] = chunk                             # write back

        print(f"    mean(cont)={float(X[:50, ..., CONTINUOUS_IDX[0]].mean()):.3f}")

    print("\nNormalize hoàn tất.")


# ================================================================
# ══════════════  BƯỚC 3: DATASET & DATALOADER  ════════════════
# ================================================================

class WildfireDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = X
        self.y       = y
        self.seq_len = seq_len
        self.indices = np.arange(seq_len, len(X))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i     = self.indices[idx]
        x_seq = self.X[i - self.seq_len : i].astype(np.float32)  # (T, H, W, C)
        y_map = self.y[i].astype(np.float32)                      # (H, W, 1)

        # channel-last → channel-first (PyTorch convention)
        x_seq = x_seq.transpose(0, 3, 1, 2)   # (T, C, H, W)
        y_map = y_map.transpose(2, 0, 1)       # (1, H, W)

        return torch.from_numpy(x_seq), torch.from_numpy(y_map)


def build_dataloaders(split_dir: Path, seq_len: int, batch_size: int,
                      num_workers: int):
    print("\n" + "=" * 60)
    print("BƯỚC 3: Build DataLoaders")
    print("=" * 60)

    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = {
            "X": np.load(split_dir / f"X_{name}.npy", mmap_mode="r"),
            "y": np.load(split_dir / f"y_{name}.npy", mmap_mode="r"),
        }
        n_seq = len(splits[name]["X"]) - seq_len
        print(f"  {name:5s}: {len(splits[name]['X'])} ngày → {n_seq} sequences")

    # pos_weight từ train
    y_tr = splits["train"]["y"]
    n_pos = float(y_tr.sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = n_neg / (n_pos + 1e-8)
    print(f"\n  pos_weight: {pos_weight:.1f}x  (n_pos={n_pos:,.0f})")

    # Datasets
    train_ds = WildfireDataset(splits["train"]["X"], splits["train"]["y"], seq_len)
    val_ds   = WildfireDataset(splits["val"]["X"],   splits["val"]["y"],   seq_len)
    test_ds  = WildfireDataset(splits["test"]["X"],  splits["test"]["y"],  seq_len)

    # WeightedRandomSampler: oversample fire sequences
    y_arr = splits["train"]["y"]
    fire_per_day = np.array([float(y_arr[t].sum()) for t in range(len(y_arr))])
    sample_w = np.array([
        float(np.log1p(fire_per_day[i - seq_len : i].max())) + 1.0
        for i in train_ds.indices
    ], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )
    n_fire_seq = int((sample_w > 1.0).sum())
    print(f"  Sequences with fire: {n_fire_seq}/{len(sample_w)}")

    pin_mem = DEVICE.type == "cuda"
    pworker = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pworker, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pworker,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pworker,
    )

    # Sanity check
    x_b, y_b = next(iter(train_loader))
    print(f"\n  Batch check: x={x_b.shape}  y={y_b.shape}"
          f"  fire_ratio={y_b.float().mean():.4%}")

    return train_loader, val_loader, test_loader, pos_weight


# ================================================================
# ══════════════  BƯỚC 4: MODEL  ════════════════════════════════
# ================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size, padding=pad, bias=True,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        i, f, o, g = self.conv(combined).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        z = torch.zeros
        return (z(B, self.hidden_channels, H, W, device=device),
                z(B, self.hidden_channels, H, W, device=device))


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3,
                 return_sequences=True):
        super().__init__()
        self.cell             = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels  = hidden_channels
        self.return_sequences = return_sequences

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W, x.device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)   # (B, T, C_hid, H, W)
        return h                                  # (B, C_hid, H, W)


class ConvLSTMWildfire(nn.Module):
    """
    Kiến trúc với skip connection từ ConvLSTM-1 → Decoder:

    Input (B, T, 52, H, W)
      → ConvLSTM-1 [64 filters, return_seq=True]
      → BN3d + Dropout3d
      → skip = temporal mean pooling  → (B, 64, H, W) → project → (B, 16, H, W)
      → ConvLSTM-2 [32 filters, return_seq=False]
      → BN2d + Dropout2d
      → Attention gate
      → concat(x, skip): (B, 48, H, W)
      → Decoder: 48→32→16→1 → Sigmoid
    """

    def __init__(self, cfg: dict):
        super().__init__()
        filters = cfg["filters"]     # [64, 32]
        k       = cfg["kernel"]      # 3
        drop    = cfg["dropout"]     # 0.4
        n_feat  = cfg["n_feat"]      # 52

        #  ConvLSTM stack 
        self.convlstm1 = ConvLSTMLayer(n_feat,      filters[0], k, return_sequences=True)
        self.bn1       = nn.BatchNorm3d(filters[0])
        self.drop1     = nn.Dropout3d(drop)

        self.convlstm2 = ConvLSTMLayer(filters[0],  filters[1], k, return_sequences=False)
        self.bn2       = nn.BatchNorm2d(filters[1])
        self.drop2     = nn.Dropout2d(drop)

        #  Skip connection: ConvLSTM-1 → project → concat ─
        # Temporal mean của ConvLSTM-1 output → project xuống 16 channels
        self.skip_proj = nn.Sequential(
            nn.Conv2d(filters[0], 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        #  Spatial attention 
        self.attention = nn.Sequential(
            nn.Conv2d(filters[1], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        #  Decoder: nhận concat(32 + 16 = 48 channels) 
        self.decoder = nn.Sequential(
            nn.Conv2d(filters[1] + 16, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,  1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # ConvLSTM-1: (B, T, 52, H, W) → (B, T, 64, H, W)
        x1 = self.convlstm1(x)

        # BN3d cần (B, C, T, H, W)
        x1 = self.bn1(x1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x1 = self.drop1(x1)

        # Skip: temporal mean pool → (B, 64, H, W) → project → (B, 16, H, W)
        skip = self.skip_proj(x1.mean(dim=1))

        # ConvLSTM-2: (B, T, 64, H, W) → (B, 32, H, W)
        x2 = self.convlstm2(x1)
        x2 = self.bn2(x2)
        x2 = self.drop2(x2)

        # Attention
        attn = self.attention(x2)
        x2   = x2 * attn

        # Concat skip + decode
        out = self.decoder(torch.cat([x2, skip], dim=1))   # (B, 1, H, W)
        return out


# ================================================================
# ══════════════  BƯỚC 5: LOSS FUNCTIONS  ══════════════════════
# ================================================================

class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    pos_weight: nhân thêm penalty cho fire pixels (BCE level)
    """
    def __init__(self, gamma=2.0, alpha=0.80, pos_weight=1.0):
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        bce  = -(self.pos_weight * target * torch.log(pred)
                 + (1 - target) * torch.log(1 - pred))
        p_t     = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    TL = 1 - TP / (TP + alpha*FP + beta*FN)

    beta > alpha  →  penalize FN (bỏ sót lửa) nặng hơn FP
    Dùng beta=0.7, alpha=0.3 để tăng Recall trực tiếp.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class CompositeLoss(nn.Module):
    """
    Kết hợp FocalLoss + TverskyLoss với trọng số điều chỉnh được.
    focal_w + tversky_w = 1.0 (nên)
    """
    def __init__(self, focal_w=0.5, tversky_w=0.5,
                 focal_kw=None, tversky_kw=None):
        super().__init__()
        self.focal_w   = focal_w
        self.tversky_w = tversky_w
        self.focal     = FocalLoss(**(focal_kw or {}))
        self.tversky   = TverskyLoss(**(tversky_kw or {}))

    def forward(self, pred, target):
        return (self.focal_w   * self.focal(pred, target)
                + self.tversky_w * self.tversky(pred, target))


# ================================================================
# ══════════════  BƯỚC 6: TRAINING LOOP  ════════════════════════
# ================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = criterion(pred, y)
            total_loss += loss.item()
            all_preds.append(pred.cpu().float().numpy().flatten())
            all_targets.append(y.cpu().float().numpy().flatten())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(loader)
    auc_pr   = average_precision_score(all_targets, all_preds)

    # Best threshold theo F1
    prec, rec, thresholds = precision_recall_curve(all_targets, all_preds)
    f1s         = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx    = int(f1s.argmax())
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    binary = (all_preds >= best_thresh).astype(float)
    tp     = ((binary == 1) & (all_targets == 1)).sum()
    fn     = ((binary == 0) & (all_targets == 1)).sum()
    recall = tp / (tp + fn + 1e-8)

    # Debug pred distribution
    print(f"    pred: min={all_preds.min():.4f} max={all_preds.max():.4f}"
          f" mean={all_preds.mean():.6f} p99={np.percentile(all_preds,99):.4f}")

    return avg_loss, float(auc_pr), float(recall), best_thresh


def plot_history(history, output_dir, run_id):
    epochs     = [h["epoch"]       for h in history]
    train_loss = [h["train_loss"]  for h in history]
    val_loss   = [h["val_loss"]    for h in history]
    val_auc    = [h["val_auc_pr"]  for h in history]
    val_recall = [h["val_recall"]  for h in history]
    lrs        = [h["lr"]          for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"ConvLSTM Training — {run_id}", fontsize=13)

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val")
    axes[0].set_title("Composite Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_auc, color="tab:green")
    axes[1].axhline(y=0.0011, color="red", linestyle="--", label="baseline")
    axes[1].set_title("Val AUC-PR"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, val_recall, color="tab:orange")
    axes[2].set_title("Val Recall"); axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, lrs, color="tab:gray")
    axes[3].set_title("Learning Rate"); axes[3].set_yscale("log")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"loss_curve_{run_id}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve → {path}")


def train(model_cfg: dict, train_cfg: dict, split_dir: Path, output_dir: Path):
    print("\n" + "=" * 60)
    print(f"BƯỚC 4: Training — run_id={RUN_ID}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM  : {vram:.1f} GB")
    print("=" * 60)

    #  DataLoaders ─
    train_loader, val_loader, test_loader, pos_weight = build_dataloaders(
        split_dir,
        seq_len     = model_cfg["seq_len"],
        batch_size  = train_cfg["batch_size"],
        num_workers = train_cfg["num_workers"],
    )

    #  Model + Loss + Optimizer 
    model = ConvLSTMWildfire(model_cfg).to(DEVICE)

    # torch.compile với inductor backend (default) — thực sự tối ưu hóa
    if DEVICE.type == "cuda":
        print("\nCompiling model (inductor)...", end=" ", flush=True)
        try:
            model = torch.compile(model)
            print("done")
        except Exception as e:
            print(f"skip ({e})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # Pos_weight cap tại 200 — quá cao gây mất ổn định gradient
    pos_weight_capped = min(pos_weight, 200.0)
    print(f"pos_weight: {pos_weight:.1f}x → capped at {pos_weight_capped:.1f}x")

    criterion = CompositeLoss(
        focal_w    = model_cfg["loss_focal_w"],
        tversky_w  = model_cfg["loss_tversky_w"],
        focal_kw   = dict(
            gamma      = model_cfg["fl_gamma"],
            alpha      = model_cfg["fl_alpha"],
            pos_weight = pos_weight_capped,
        ),
        tversky_kw = dict(
            alpha = model_cfg["tv_alpha"],
            beta  = model_cfg["tv_beta"],
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = model_cfg["lr"],
        weight_decay = model_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",
        factor   = train_cfg["lr_factor"],
        patience = train_cfg["lr_patience"],
        min_lr   = train_cfg["min_lr"],
    )

    scaler       = GradScaler("cuda")
    accum_steps  = train_cfg["accum_steps"]

    #  Tracking 
    model_path   = output_dir / f"best_model_{RUN_ID}.pt"
    history_path = output_dir / f"history_{RUN_ID}.csv"
    summary_path = output_dir / f"summary_{RUN_ID}.txt"

    history      = []
    best_auc_pr  = 0.0
    patience_cnt = 0

    csv_file = open(history_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "val_loss", "val_auc_pr",
        "val_recall", "best_thresh", "lr",
    ])
    writer.writeheader()

    #  Training loop ─
    print(f"\nTraining (max {train_cfg['epochs']} epochs, "
          f"accum={accum_steps}, "
          f"effective_batch={train_cfg['batch_size'] * accum_steps})...\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss   = 0.0
        optimizer.zero_grad()

        n_batches = len(train_loader)
        log_every = max(1, n_batches // 4)   # print 4 lần mỗi epoch

        for step, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                pred = model(x)
                loss = criterion(pred, y) / accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * accum_steps   # unscale for logging

            if (step + 1) % log_every == 0:
                print(f"  [{epoch}/{train_cfg['epochs']}] step {step+1}/{n_batches}"
                      f"  loss={train_loss / (step + 1):.4f}", flush=True)

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), model_cfg["clipnorm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss /= len(train_loader)

        val_loss, val_auc_pr, val_recall, best_thresh = evaluate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_auc_pr)
        current_lr = optimizer.param_groups[0]["lr"]

        # VRAM log
        vram_str = ""
        if DEVICE.type == "cuda" and epoch % 10 == 0:
            used = torch.cuda.memory_allocated() / 1024**3
            vram_str = f"  VRAM={used:.1f}GB"

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | auc_pr={val_auc_pr:.4f} | "
              f"recall={val_recall:.4f} | thresh={best_thresh:.3f} | "
              f"lr={current_lr:.2e}{vram_str}")

        row = {
            "epoch":       epoch,
            "train_loss":  round(train_loss,  6),
            "val_loss":    round(val_loss,    6),
            "val_auc_pr":  round(val_auc_pr,  6),
            "val_recall":  round(val_recall,  4),
            "best_thresh": round(best_thresh, 4),
            "lr":          current_lr,
        }
        history.append(row)
        writer.writerow(row)
        csv_file.flush()

        # Save best
        if val_auc_pr > best_auc_pr:
            best_auc_pr  = val_auc_pr
            patience_cnt = 0

            # Strip torch.compile prefix nếu có
            state = model.state_dict()
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

            torch.save({
                "epoch":       epoch,
                "model_state": state,
                "val_auc_pr":  val_auc_pr,
                "val_recall":  val_recall,
                "best_thresh": best_thresh,
                "model_cfg":   model_cfg,
                "train_cfg":   train_cfg,
                "pos_weight":  pos_weight_capped,
                "run_id":      RUN_ID,
            }, model_path)
            print(f"  ✓ Best saved  (auc_pr={best_auc_pr:.4f}  "
                  f"recall={val_recall:.4f}  thresh={best_thresh:.3f})")
        else:
            patience_cnt += 1

        if patience_cnt >= train_cfg["patience"]:
            print(f"\nEarly stopping tại epoch {epoch} "
                  f"(không cải thiện sau {train_cfg['patience']} epochs)")
            break

    csv_file.close()

    #  Test set evaluation ─
    print("\n" + "=" * 60)
    print("BƯỚC 5: Evaluate trên test set")
    print("=" * 60)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Load vào model mới chưa compile — tránh key mismatch "_orig_mod." prefix
    # khi model training đã qua torch.compile()
    eval_model = ConvLSTMWildfire(model_cfg).to(DEVICE)
    eval_model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_full(eval_model, test_loader, DEVICE)

    print(f"\n{'='*60}")
    print("KẾT QUẢ CUỐI CÙNG")
    print(f"{'='*60}")
    print(f"  Best epoch  : {ckpt['epoch']}")
    print(f"  Val AUC-PR  : {ckpt['val_auc_pr']:.4f}")
    print(f"  Test AUC-PR : {test_metrics['auc_pr']:.4f}  (baseline=0.0011)")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  Test Recall : {test_metrics['recall']:.4f}")
    print(f"  Test Prec   : {test_metrics['precision']:.4f}")
    print(f"  Test F1     : {test_metrics['f1']:.4f}")
    print(f"  Test CSI    : {test_metrics['csi']:.4f}")
    print(f"  Threshold   : {test_metrics['threshold']:.3f}")
    print(f"  TP={test_metrics['tp']:,}  FP={test_metrics['fp']:,}"
          f"  FN={test_metrics['fn']:,}  TN={test_metrics['tn']:,}")

    # Save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"run_id         : {RUN_ID}\n")
        f.write(f"device         : {DEVICE}\n")
        f.write(f"n_features     : {N_FEAT}\n")
        f.write(f"best_epoch     : {ckpt['epoch']}\n")
        f.write(f"val_auc_pr     : {ckpt['val_auc_pr']:.6f}\n")
        f.write(f"val_recall     : {ckpt['val_recall']:.6f}\n")
        f.write(f"--- TEST SET ---\n")
        for k, v in test_metrics.items():
            if k not in ("precisions", "recalls"):
                f.write(f"{k:<15}: {v}\n")
        f.write(f"model_cfg      : {model_cfg}\n")
        f.write(f"train_cfg      : {train_cfg}\n")
    print(f"\n  Summary → {summary_path}")

    plot_history(history, output_dir, RUN_ID)

    return model, history, model_path, test_metrics


# ================================================================
# ══════════════  BƯỚC 7: EVALUATE ĐẦY ĐỦ  ═════════════════════
# ================================================================

def evaluate_full(model, loader, device):
    """Tính đầy đủ metrics cho báo cáo."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            all_preds.append(pred.cpu().float().numpy().flatten())
            all_targets.append(y.float().numpy().flatten())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc_pr  = average_precision_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)

    prec, rec, thresholds = precision_recall_curve(all_targets, all_preds)
    f1s         = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx    = int(f1s.argmax())
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred = (all_preds >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_targets, y_pred).ravel()

    recall    = float(tp / (tp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
    csi       = float(tp / (tp + fp + fn + 1e-8))
    fpr       = float(fp / (fp + tn + 1e-8))

    return {
        "auc_pr":    auc_pr,
        "auc_roc":   auc_roc,
        "threshold": best_thresh,
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "csi":       csi,
        "fpr":       fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precisions": prec,
        "recalls":    rec,
    }


# ================================================================
# ══════════════  BƯỚC 8: LƯU PROBABILITY MAPS  ════════════════
# Input cho compare_models.py — cần download về local sau khi chạy
# ================================================================

def save_prob_maps(model_path: Path, split_dir: Path,
                   output_dir: Path, model_cfg: dict, train_cfg: dict):
    """
    Chạy inference trên train/val/test, lưu prob_map_*.npy + targets_*.npy.

    Output:
      prob_map_train.npy   — xác suất cháy mỗi pixel (train)
      prob_map_val.npy
      prob_map_test.npy
      targets_train.npy    — ground truth tương ứng
      targets_val.npy
      targets_test.npy

    Format: 1-D float32 array, flattened theo thứ tự [day0_pixels, day1_pixels, ...]
    Mỗi ngày có H*W pixels (137*138 = 18,906).
    """
    print("\n" + "=" * 60)
    print("BƯỚC 8: Lưu probability maps")
    print("=" * 60)

    # Load model
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = ConvLSTMWildfire(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded: epoch={ckpt['epoch']}  val_auc_pr={ckpt['val_auc_pr']:.4f}")

    seq_len     = model_cfg["seq_len"]
    batch_size  = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    pin_mem     = DEVICE.type == "cuda"
    pworker     = num_workers > 0

    for split_name in ["train", "val", "test"]:
        print(f"\n  [{split_name}] Running inference...")

        X = np.load(split_dir / f"X_{split_name}.npy", mmap_mode="r")
        y = np.load(split_dir / f"y_{split_name}.npy", mmap_mode="r")

        ds = WildfireDataset(X, y, seq_len)
        loader = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_mem,
            persistent_workers = pworker,
        )

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(DEVICE, non_blocking=True)
                pred = model(x_b)
                all_preds.append(pred.cpu().float().numpy().flatten())
                all_targets.append(y_b.float().numpy().flatten())

        preds   = np.concatenate(all_preds).astype(np.float32)
        targets = np.concatenate(all_targets).astype(np.float32)

        pred_path   = output_dir / f"prob_map_{split_name}.npy"
        target_path = output_dir / f"targets_{split_name}.npy"
        np.save(pred_path,   preds)
        np.save(target_path, targets)

        auc = average_precision_score(targets, preds)
        n_days = len(ds)
        print(f"    Samples  : {len(preds):,}  ({n_days} ngày × {len(preds)//n_days} pixels)")
        print(f"    AUC-PR   : {auc:.4f}")
        print(f"    Fire rate: {targets.mean():.4%}")
        print(f"    Saved → {pred_path.name}  +  {target_path.name}")

    print("\n  Hoàn tất — download các file prob_map_*.npy + targets_*.npy về local")
    print("  để dùng với Project/article_comparison/compare_models.py")


# ================================================================
# ══════════════  ENTRY POINT  ══════════════════════════════════
# ================================================================

if __name__ == "__main__":
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"Device  : {DEVICE}")
    print(f"Run ID  : {RUN_ID}")
    print(f"Features: {N_FEAT}  |  Continuous to normalize: {len(CONTINUOUS_IDX)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #  Bước 1: Build splits 
    H, W, split_dates = build_splits(DATA_PATH, SPLIT_DIR)

    # Cập nhật H, W vào model config
    MODEL_CFG["H"] = H
    MODEL_CFG["W"] = W

    #  Bước 2: Normalize ─
    normalize_splits(SPLIT_DIR)

    #  Bước 3–7: Train + Evaluate ─
    model, history, model_path, test_metrics = train(
        MODEL_CFG, TRAIN_CFG, SPLIT_DIR, OUTPUT_DIR
    )

    #  Bước 8: Lưu probability maps (để dùng với compare_models.py) 
    save_prob_maps(model_path, SPLIT_DIR, OUTPUT_DIR, MODEL_CFG, TRAIN_CFG)

    print("\n" + "=" * 60)
    print("HOÀN TẤT")
    print(f"  Model    : {model_path}")
    print(f"  AUC-PR   : {test_metrics['auc_pr']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  F1       : {test_metrics['f1']:.4f}")
    print(f"  CSI      : {test_metrics['csi']:.4f}")
    print("\nFiles để download về local:")
    for name in ["prob_map_train", "prob_map_val", "prob_map_test",
                 "targets_train",  "targets_val",  "targets_test"]:
        print(f"  {OUTPUT_DIR / (name + '.npy')}")
    print("=" * 60)