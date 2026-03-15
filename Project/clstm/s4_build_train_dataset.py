import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ================================================================
# CẤU HÌNH
# ================================================================
from config import OUTPUT_DATA_PATH

SAVE_DIR = Path(OUTPUT_DATA_PATH) / "processed"

SEQ_LEN    = 7
BATCH_SIZE  = 8
NUM_WORKERS = 0
PIN_MEMORY  = True

H, W, N_FEAT = 137, 138, 25


# ================================================================
# DATASET CLASS
# ================================================================
class WildfireDataset(Dataset):
    """
    PyTorch Dataset cho dữ liệu cháy rừng dạng spatiotemporal.

    Mỗi sample:
      x : (seq_len, N_FEAT, H, W)  — PyTorch dùng channel-first
      y : (1, H, W)                — label bản đồ cháy

    Lưu ý channel order:
      numpy/TF : (H, W, C)   — channel last
      PyTorch  : (C, H, W)   — channel first  ← cần transpose
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        print(f"    Loading {len(X)} ngày vào RAM...", end=" ", flush=True)
        self.X = np.array(X)  # ← quan trọng nhất
        self.y = np.array(y)
        print("done")
        self.seq_len = seq_len
        self.T = len(self.X)
        self.indices = np.arange(seq_len, self.T)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Lấy seq_len ngày liên tiếp
        x_seq = self.X[i - self.seq_len : i]   # (seq_len, H, W, N_FEAT)
        y_map = self.y[i]                        # (H, W, 1)

        # Chuyển sang float32
        x_seq = x_seq.astype(np.float32)
        y_map = y_map.astype(np.float32)

        # Transpose: channel last → channel first (PyTorch convention)
        # (seq_len, H, W, C) → (seq_len, C, H, W)
        x_seq = x_seq.transpose(0, 3, 1, 2)
        # (H, W, 1) → (1, H, W)
        y_map = y_map.transpose(2, 0, 1)

        return torch.from_numpy(x_seq), torch.from_numpy(y_map)


# ================================================================
# TÍNH CLASS WEIGHT
# ================================================================
def compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Tính tỷ lệ negative/positive cho focal loss.
    Với 0.11% fire: pos_weight ≈ 900
    """
    n_pos = float(y_train[:].sum())
    n_neg = float((y_train[:] == 0).sum())
    pos_weight = n_neg / (n_pos + 1e-8)
    return pos_weight, n_pos, n_neg


# ================================================================
# BUILD TẤT CẢ DATALOADERS
# ================================================================
def build_all_dataloaders(
    save_dir  = SAVE_DIR,
    seq_len   = SEQ_LEN,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
):
    """
    Load splits → tạo Datasets → tạo DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    pos_weight                            : float
    info                                  : dict
    """
    print("=" * 50)
    print("BƯỚC 2: Build PyTorch DataLoaders")
    print("=" * 50)

    # --- Load mmap arrays ---
    print("\nLoad splits (mmap)...")
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = {
            "X": np.load(save_dir / f"X_{name}.npy", mmap_mode="r"),
            "y": np.load(save_dir / f"y_{name}.npy", mmap_mode="r"),
        }
        n_samples = len(splits[name]["X"]) - seq_len
        print(f"  {name:5s}: {len(splits[name]['X'])} ngày "
              f"→ {n_samples} samples")

    # --- Class weight ---
    print("\nTính class weight...")
    pos_weight, n_pos, n_neg = compute_pos_weight(splits["train"]["y"])
    print(f"  n_pos      : {n_pos:,.0f}")
    print(f"  n_neg      : {n_neg:,.0f}")
    print(f"  pos_weight : {pos_weight:.1f}x")

    # --- Datasets ---
    train_ds = WildfireDataset(splits["train"]["X"],
                               splits["train"]["y"], seq_len)
    val_ds   = WildfireDataset(splits["val"]["X"],
                               splits["val"]["y"],   seq_len)
    test_ds  = WildfireDataset(splits["test"]["X"],
                               splits["test"]["y"],  seq_len)

    # --- DataLoaders ---
    # train: shuffle=True — xáo thứ tự sample (không ảnh hưởng time order bên trong sequence)
    # val/test: shuffle=False — giữ nguyên thứ tự để evaluate đúng
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = PIN_MEMORY,
        drop_last   = True,    # bỏ batch cuối nếu không đủ batch_size
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = PIN_MEMORY,
    )

    # --- Sanity check ---
    print("\nSanity check — lấy 1 batch từ train_loader...")
    x_batch, y_batch = next(iter(train_loader))
    print(f"  x_batch.shape : {x_batch.shape}")
    # → (4, 7, 25, 137, 138)
    #    B  seq C   H   W
    print(f"  y_batch.shape : {y_batch.shape}")
    # → (4, 1, 137, 138)
    print(f"  x dtype       : {x_batch.dtype}")
    print(f"  y dtype       : {y_batch.dtype}")
    print(f"  y fire ratio  : {y_batch.float().mean():.4%}")
    print(f"  x value range : [{x_batch.min():.2f}, {x_batch.max():.2f}]")

    info = {
        "seq_len":     seq_len,
        "batch_size":  batch_size,
        "n_train":     len(train_ds),
        "n_val":       len(val_ds),
        "n_test":      len(test_ds),
        "H": H, "W": W, "N_FEAT": N_FEAT,
        "pos_weight":  pos_weight,
    }

    print(f"\n  train_loader : {len(train_ds)} samples "
          f"→ {len(train_loader)} batches/epoch")
    print(f"  val_loader   : {len(val_ds)} samples "
          f"→ {len(val_loader)} batches/epoch")
    print(f"  test_loader  : {len(test_ds)} samples "
          f"→ {len(test_loader)} batches/epoch")

    print("\n" + "=" * 50)
    print("BƯỚC 2 HOÀN TẤT")
    print("=" * 50)

    return train_loader, val_loader, test_loader, pos_weight, info


# ================================================================
# CHẠY TRỰC TIẾP ĐỂ KIỂM TRA
# ================================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, pos_weight, info = \
        build_all_dataloaders()