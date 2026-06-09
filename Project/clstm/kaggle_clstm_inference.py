"""
kaggle_clstm_inference.py — Inference only, không cần retrain
==============================================================
Dùng khi đã có checkpoint từ lần train trước.

Cách dùng trên Kaggle:
  1. Add checkpoint (.pt) vào notebook dưới dạng dataset, hoặc để trong /kaggle/working/
  2. Chỉnh MODEL_PATH bên dưới cho đúng
  3. Chạy script — sẽ build splits nếu chưa có, rồi chạy inference

Output (/kaggle/working/):
  prob_map_train.npy  targets_train.npy
  prob_map_val.npy    targets_val.npy
  prob_map_test.npy   targets_test.npy
  inference_summary_<run_id>.txt

Download 2 file cần cho compare_models.py:
  prob_map_test.npy + targets_test.npy
"""

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import average_precision_score

# ================================================================
# CẤU HÌNH — chỉnh 2 dòng này
# ================================================================

MODEL_PATH = "/kaggle/working/best_model_20260428_1418.pt"   # ← checkpoint có sẵn

DATA_PATH  = "/kaggle/input/datasets/flyingcow781/dataset-v3-new/daklak_final_dataset_v3_pathways.parquet"
OUTPUT_DIR = Path("/kaggle/working")
SPLIT_DIR  = OUTPUT_DIR / "splits"

#  Feature list (phải khớp với lúc train) 
FEATURE_COLS = [
    "tmean", "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "vpd_14d_mean",
    "fire_lag_1", "fire_lag_3",
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d",
    "sin_doy", "cos_doy",
    "ndvi", "nbr", "delta_ndvi_14d", "delta_nbr_7d",
    "days_since_last_fire", "burn_season_flag",
    "days_since_harvest", "fire_count_prev_year",
    "dem_mean", "slp_mean",
    "dist_road_km", "dist_settlement_km",
    "tree_cover_pct", "pop_density",
]
N_FEAT = len(FEATURE_COLS)   # 28

_NO_NORM = {
    "fire_lag_1", "fire_lag_3",
    "sin_doy", "cos_doy",
    "ndvi", "nbr",
    "burn_season_flag",
}
CONTINUOUS_IDX = [i for i, f in enumerate(FEATURE_COLS) if f not in _NO_NORM]

GRID_ID_COL = "grid_id"
DATE_COL    = "date"
TARGET_COL  = "fire"

TRAIN_END_DATE = "2021-12-31"
VAL_END_DATE   = "2022-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")


# ================================================================
# MODEL DEFINITION (phải khớp với kaggle_clstm_train.py)
# ================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 4,
            kernel_size, padding=kernel_size // 2, bias=True,
        )

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_next = f * c + i * torch.tanh(g)
        return o * torch.tanh(c_next), c_next

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
        return torch.stack(outputs, dim=1) if self.return_sequences else h


class ConvLSTMWildfire(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        filters = cfg["filters"]
        k, drop, n_feat = cfg["kernel"], cfg["dropout"], cfg["n_feat"]

        self.convlstm1 = ConvLSTMLayer(n_feat,      filters[0], k, True)
        self.bn1       = nn.BatchNorm3d(filters[0])
        self.drop1     = nn.Dropout3d(drop)

        self.convlstm2 = ConvLSTMLayer(filters[0],  filters[1], k, False)
        self.bn2       = nn.BatchNorm2d(filters[1])
        self.drop2     = nn.Dropout2d(drop)

        self.skip_proj = nn.Sequential(nn.Conv2d(filters[0], 16, 1), nn.ReLU(inplace=True))
        self.attention = nn.Sequential(nn.Conv2d(filters[1], 1, 1), nn.Sigmoid())
        self.decoder   = nn.Sequential(
            nn.Conv2d(filters[1] + 16, 32, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),                           nn.ReLU(inplace=True),
            nn.Conv2d(16,  1, 1),                                      nn.Sigmoid(),
        )

    def forward(self, x):
        x1   = self.convlstm1(x)
        x1   = self.bn1(x1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x1   = self.drop1(x1)
        skip = self.skip_proj(x1.mean(dim=1))
        x2   = self.convlstm2(x1)
        x2   = self.drop2(self.bn2(x2))
        x2   = x2 * self.attention(x2)
        return self.decoder(torch.cat([x2, skip], dim=1))


# ================================================================
# DATASET
# ================================================================

class WildfireDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y   = X, y
        self.seq_len     = seq_len
        self.indices     = np.arange(seq_len, len(X))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i     = self.indices[idx]
        x_seq = self.X[i - self.seq_len : i].astype(np.float32).transpose(0, 3, 1, 2)
        y_map = self.y[i].astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(x_seq), torch.from_numpy(y_map)


# ================================================================
# BUILD SPLITS (nếu chưa có)
# ================================================================

def build_splits_if_needed(split_dir: Path, H_ref=None, W_ref=None):
    """
    Kiểm tra splits đã tồn tại chưa.
    Nếu chưa → build từ parquet (mất ~20 phút).
    Nếu rồi  → skip, tiết kiệm thời gian.
    """
    needed = ["X_train.npy", "X_val.npy", "X_test.npy",
              "y_train.npy", "y_val.npy", "y_test.npy",
              "norm_mean.npy", "norm_std.npy"]
    missing = [f for f in needed if not (split_dir / f).exists()]

    if not missing:
        print(f"Splits đã tồn tại tại {split_dir} — skip build.")
        # Lấy H, W từ shape của X_train
        X_shape = np.load(split_dir / "X_train.npy", mmap_mode="r").shape
        H, W = X_shape[1], X_shape[2]
        print(f"  H={H}, W={W}")
        return H, W

    print(f"Splits chưa có ({missing}) — building từ parquet...")
    print("  (Bước này mất ~20–30 phút trên Kaggle)")

    split_dir.mkdir(parents=True, exist_ok=True)

    load_cols = [GRID_ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    df = pd.read_parquet(DATA_PATH, columns=load_cols, engine="pyarrow")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, GRID_ID_COL]).reset_index(drop=True)

    grid_ids  = np.sort(df[GRID_ID_COL].unique())
    H = int(np.sqrt(len(grid_ids)))
    W = int(np.ceil(len(grid_ids) / H))
    grid_index = {gid: i for i, gid in enumerate(grid_ids)}

    all_dates = np.sort(df[DATE_COL].unique())
    dates_ts  = pd.DatetimeIndex(all_dates)
    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end   = pd.Timestamp(VAL_END_DATE)

    split_dates = {
        "train": all_dates[dates_ts <= train_end],
        "val":   all_dates[(dates_ts > train_end) & (dates_ts <= val_end)],
        "test":  all_dates[dates_ts > val_end],
    }

    n_feat = len(FEATURE_COLS)
    for split_name, dates in split_dates.items():
        T = len(dates)
        X = np.lib.format.open_memmap(split_dir / f"X_{split_name}.npy",
                                       mode="w+", dtype="float32", shape=(T, H, W, n_feat))
        y = np.lib.format.open_memmap(split_dir / f"y_{split_name}.npy",
                                       mode="w+", dtype="float32", shape=(T, H, W, 1))
        date_to_t = {pd.Timestamp(d): i for i, d in enumerate(dates)}
        date_set  = set(pd.Timestamp(d) for d in dates)

        for date, day_df in df[df[DATE_COL].isin(date_set)].groupby(DATE_COL):
            t = date_to_t[date]
            day_df = day_df.fillna(0)
            gidx = day_df[GRID_ID_COL].map(grid_index).to_numpy()
            X[t][gidx // W, gidx % W] = day_df[FEATURE_COLS].to_numpy(dtype=np.float32)
            y[t][gidx // W, gidx % W, 0] = day_df[TARGET_COL].to_numpy(dtype=np.float32)
        print(f"  [{split_name}] {T} days done. fire={float(y[:].mean()):.4%}")

    del df; gc.collect()

    # Normalize
    X_tr = np.load(split_dir / "X_train.npy", mmap_mode="r+")
    cont = X_tr[:, ..., CONTINUOUS_IDX]
    mean = cont.mean(axis=(0, 1, 2), keepdims=True)
    std  = cont.std(axis=(0, 1, 2),  keepdims=True) + 1e-8
    np.save(split_dir / "norm_mean.npy", mean)
    np.save(split_dir / "norm_std.npy",  std)

    for split_name in ["train", "val", "test"]:
        X = np.load(split_dir / f"X_{split_name}.npy", mmap_mode="r+")
        for s in range(0, len(X), 50):
            chunk = X[s:s+50].copy()
            chunk[..., CONTINUOUS_IDX] = (chunk[..., CONTINUOUS_IDX] - mean) / std
            X[s:s+50] = chunk

    print("Build + normalize hoàn tất.")
    return H, W


# ================================================================
# INFERENCE + LƯU PROB MAPS
# ================================================================

def run_inference(model_path: str, split_dir: Path,
                  output_dir: Path, model_cfg: dict):
    print("=" * 60)
    print(f"INFERENCE — {model_path}")
    print("=" * 60)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    print(f"  Epoch     : {ckpt['epoch']}")
    print(f"  Val AUC-PR: {ckpt['val_auc_pr']:.4f}")
    print(f"  Val Recall: {ckpt['val_recall']:.4f}")

    model = ConvLSTMWildfire(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len     = model_cfg["seq_len"]
    batch_size  = 8
    num_workers = 2
    pin_mem     = DEVICE.type == "cuda"

    results = {}
    for split_name in ["train", "val", "test"]:
        print(f"\n  [{split_name}] Inference...")
        X = np.load(split_dir / f"X_{split_name}.npy", mmap_mode="r")
        y = np.load(split_dir / f"y_{split_name}.npy", mmap_mode="r")

        ds = WildfireDataset(X, y, seq_len)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem,
            persistent_workers=(num_workers > 0),
        )

        all_preds, all_targets = [], []
        with torch.no_grad():
            for step, (x_b, y_b) in enumerate(loader):
                x_b  = x_b.to(DEVICE, non_blocking=True)
                pred = model(x_b)
                all_preds.append(pred.cpu().float().numpy().flatten())
                all_targets.append(y_b.float().numpy().flatten())
                if (step + 1) % 50 == 0:
                    print(f"    step {step+1}/{len(loader)}", flush=True)

        preds   = np.concatenate(all_preds).astype(np.float32)
        targets = np.concatenate(all_targets).astype(np.float32)

        pred_path   = output_dir / f"prob_map_{split_name}.npy"
        target_path = output_dir / f"targets_{split_name}.npy"
        np.save(pred_path,   preds)
        np.save(target_path, targets)

        auc = average_precision_score(targets, preds)
        n_days = len(ds)
        print(f"    {len(preds):,} pixels  ({n_days} ngày)")
        print(f"    AUC-PR   : {auc:.4f}")
        print(f"    Fire rate: {targets.mean():.4%}")
        print(f"    → {pred_path.name}")
        results[split_name] = {"auc_pr": auc, "n_pixels": len(preds), "n_days": n_days}

    # Save summary
    summary_path = output_dir / f"inference_summary_{RUN_ID}.txt"
    with open(summary_path, "w") as f:
        f.write(f"model_path : {model_path}\n")
        f.write(f"checkpoint : epoch={ckpt['epoch']}  val_auc_pr={ckpt['val_auc_pr']:.6f}\n")
        f.write(f"infer_time : {RUN_ID}\n\n")
        for split_name, r in results.items():
            f.write(f"{split_name:6s}: auc_pr={r['auc_pr']:.6f}  "
                    f"n_days={r['n_days']}  n_pixels={r['n_pixels']:,}\n")
    print(f"\n  Summary → {summary_path}")

    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print(f"Device   : {DEVICE}")
    print(f"Model    : {MODEL_PATH}")
    print(f"Features : {N_FEAT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #  Bước 1: Load config từ checkpoint 
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    print(f"\nConfig từ checkpoint: {model_cfg}")

    #  Bước 2: Build splits nếu chưa có ─
    H, W = build_splits_if_needed(SPLIT_DIR)
    model_cfg["H"] = H
    model_cfg["W"] = W

    #  Bước 3: Inference + lưu prob maps 
    results = run_inference(MODEL_PATH, SPLIT_DIR, OUTPUT_DIR, model_cfg)

    print("\n" + "=" * 60)
    print("HOÀN TẤT — Download các file sau về local:")
    for name in ["prob_map_test", "targets_test"]:
        print(f"  {OUTPUT_DIR / (name + '.npy')}")
    print("Đặt vào Project/clstm/models/ rồi chạy compare_models.py")
    print("=" * 60)