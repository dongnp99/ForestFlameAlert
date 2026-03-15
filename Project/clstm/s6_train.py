import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from s4_build_train_dataset import build_all_dataloaders
from s5_model import build_model

# ================================================================
# CẤU HÌNH
# ================================================================
from config import OUTPUT_DATA_PATH

OUTPUT_DIR = Path(OUTPUT_DATA_PATH) / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")

MODEL_CFG = {
    "seq_len":  14,
    "H":        137,
    "W":        138,
    "n_feat":   25,
    "filters":  [64, 32],
    "kernel":   3,
    "dropout":  0.3,
    "fl_gamma": 2.0,
    "fl_alpha": 0.75,   # tăng từ 0.25 → 0.75
    "lr":       5e-4,
    "clipnorm": 1.0,
}

TRAIN_CFG = {
    "epochs":      100,
    "batch_size":  4,       # tăng từ 4 → 8
    "patience":    10,
    "lr_patience": 5,
    "lr_factor":   0.5,
    "min_lr":      1e-6,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# TÍNH METRICS TRÊN VAL SET
# ================================================================
def evaluate(model, loader, criterion, device):
    """
    Chạy model trên toàn bộ loader, trả về:
      - loss trung bình
      - AUC-PR (metric chính)
      - Recall @ best threshold (tối ưu F1)
      - best_threshold
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # enabled=False khi chạy CPU để tránh warning
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

    # Debug: in phân phối pred để kiểm tra model có đang học không
    print(f"    pred stats: min={all_preds.min():.4f} "
          f"max={all_preds.max():.4f} "
          f"mean={all_preds.mean():.6f} "
          f"p95={np.percentile(all_preds, 95):.4f} "
          f"p99={np.percentile(all_preds, 99):.4f}")

    # Tìm threshold tối ưu theo F1 thay vì dùng 0.5 cứng
    precisions, recalls, thresholds = precision_recall_curve(
        all_targets, all_preds
    )
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx    = f1s.argmax()
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) \
                  else 0.5

    binary = (all_preds >= best_thresh).astype(float)
    tp     = ((binary == 1) & (all_targets == 1)).sum()
    fn     = ((binary == 0) & (all_targets == 1)).sum()
    recall = tp / (tp + fn + 1e-8)

    return avg_loss, auc_pr, float(recall), best_thresh


# ================================================================
# VẼ LOSS CURVE
# ================================================================
def plot_history(history: list, output_dir: Path, run_id: str):
    epochs     = [h["epoch"]        for h in history]
    train_loss = [h["train_loss"]   for h in history]
    val_loss   = [h["val_loss"]     for h in history]
    val_auc    = [h["val_auc_pr"]   for h in history]
    val_recall = [h["val_recall"]   for h in history]
    lrs        = [h["lr"]           for h in history]
    thresholds = [h["best_thresh"]  for h in history]

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    fig.suptitle(f"Training History — {run_id}", fontsize=13)

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss,   label="val")
    axes[0].set_title("Focal Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_auc, color="tab:green")
    axes[1].axhline(y=0.0011, color="red", linestyle="--", label="baseline")
    axes[1].set_title("Val AUC-PR  (metric chính)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_recall, color="tab:orange")
    axes[2].set_title("Val Recall @ best threshold")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, thresholds, color="tab:purple")
    axes[3].set_title("Best threshold")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(epochs, lrs, color="tab:gray")
    axes[4].set_title("Learning Rate")
    axes[4].set_xlabel("Epoch")
    axes[4].set_yscale("log")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"loss_curve_{run_id}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve → {path}")


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def train():
    print("=" * 55)
    print(f"BƯỚC 4: Training PyTorch — run_id={RUN_ID}")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM   : {vram:.1f} GB")
    print("=" * 55)

    # ── 1. DataLoaders ───────────────────────────────────────────
    print("\n[1/5] Build DataLoaders...")
    train_loader, val_loader, test_loader, pos_weight, info = \
        build_all_dataloaders(
            seq_len    = MODEL_CFG["seq_len"],
            batch_size = TRAIN_CFG["batch_size"],
        )
    print(f"  pos_weight raw : {pos_weight:.1f}x")
    pos_weight = min(pos_weight, 50.0)  # cap tối đa 50, không để ~900
    print(f"  pos_weight cap : {pos_weight:.1f}x")

    # ── 2. Model ─────────────────────────────────────────────────
    print("\n[2/5] Build model...")
    # Truyền pos_weight từ data vào FocalLoss
    model, criterion, optimizer = build_model(MODEL_CFG,
                                              pos_weight=pos_weight)
    model = model.to(DEVICE)
    if DEVICE.type == "cuda":
        print("  Compiling model (torch.compile)... ", end="", flush=True)
        model = torch.compile(model, backend="eager")
        print("done")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total_params:,}")
    print(f"  FocalLoss    : gamma={MODEL_CFG['fl_gamma']} "
          f"alpha={MODEL_CFG['fl_alpha']} "
          f"pos_weight={pos_weight:.1f}")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",       # maximize AUC-PR
        factor   = TRAIN_CFG["lr_factor"],
        patience = TRAIN_CFG["lr_patience"],
        min_lr   = TRAIN_CFG["min_lr"],
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")

    # ── 3. Setup tracking ────────────────────────────────────────
    print("\n[3/5] Setup tracking...")
    model_path   = OUTPUT_DIR / f"best_model_{RUN_ID}.pt"
    history_path = OUTPUT_DIR / f"history_{RUN_ID}.csv"
    summary_path = OUTPUT_DIR / f"summary_{RUN_ID}.txt"

    history      = []
    best_auc_pr  = 0.0
    best_epoch   = 1
    patience_cnt = 0

    csv_file = open(history_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "val_loss",
        "val_auc_pr", "val_recall", "best_thresh", "lr"
    ])
    writer.writeheader()

    print(f"  Model path   → {model_path}")
    print(f"  History CSV  → {history_path}")

    # ── 4. Training loop ─────────────────────────────────────────
    print(f"\n[4/5] Training (max {TRAIN_CFG['epochs']} epochs)...\n")

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader,
                         desc=f"Epoch {epoch:3d}/{TRAIN_CFG['epochs']}",
                         leave=False):
            x = x.to(DEVICE, non_blocking=True)   # (B,T,C,H,W)
            y = y.to(DEVICE, non_blocking=True)   # (B,1,H,W)

            optimizer.zero_grad()

            with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                pred = model(x)               # (B,1,H,W)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), MODEL_CFG["clipnorm"]
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        val_loss, val_auc_pr, val_recall, best_thresh = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # LR schedule dựa trên val_auc_pr
        scheduler.step(val_auc_pr)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
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

        # VRAM log mỗi 10 epoch
        vram_str = ""
        if DEVICE.type == "cuda" and epoch % 10 == 0:
            used = torch.cuda.memory_allocated() / 1024**3
            vram_str = f"  VRAM={used:.1f}GB"

        print(f"Epoch {epoch:3d}/{TRAIN_CFG['epochs']} | "
              f"loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_auc_pr={val_auc_pr:.4f} | "
              f"val_recall={val_recall:.4f} | "
              f"thresh={best_thresh:.3f} | "
              f"lr={current_lr:.2e}"
              f"{vram_str}")

        # --- Save best model ---
        if val_auc_pr > best_auc_pr:
            best_auc_pr  = val_auc_pr
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc_pr":      val_auc_pr,
                "val_recall":      val_recall,
                "best_thresh":     best_thresh,
                "model_cfg":       MODEL_CFG,
                "train_cfg":       TRAIN_CFG,
                "pos_weight":      pos_weight,
            }, model_path)
            print(f"  ✓ Best model saved  "
                  f"(val_auc_pr={best_auc_pr:.4f}  "
                  f"recall={val_recall:.4f}  "
                  f"thresh={best_thresh:.3f})")
        else:
            patience_cnt += 1

        # --- Early stopping ---
        if patience_cnt >= TRAIN_CFG["patience"]:
            print(f"\nEarly stopping tại epoch {epoch} "
                  f"(không cải thiện sau {TRAIN_CFG['patience']} epochs)")
            break

    csv_file.close()

    # ── 5. Kết quả ───────────────────────────────────────────────
    print("\n[5/5] Lưu kết quả...")

    ckpt = torch.load(model_path, map_location=DEVICE)
    print(f"\n{'='*55}")
    print("KẾT QUẢ TRAINING")
    print(f"{'='*55}")
    print(f"  Best epoch   : {ckpt['epoch']}")
    print(f"  val AUC-PR   : {ckpt['val_auc_pr']:.4f}  (baseline=0.0011)")
    print(f"  val Recall   : {ckpt['val_recall']:.4f}")
    print(f"  Best thresh  : {ckpt['best_thresh']:.3f}")
    print(f"  Model saved  : {model_path}")

    with open(summary_path, "w") as f:
        f.write(f"run_id       : {RUN_ID}\n")
        f.write(f"device       : {DEVICE}\n")
        f.write(f"best_epoch   : {ckpt['epoch']}\n")
        f.write(f"val_auc_pr   : {ckpt['val_auc_pr']:.6f}\n")
        f.write(f"val_recall   : {ckpt['val_recall']:.6f}\n")
        f.write(f"best_thresh  : {ckpt['best_thresh']:.4f}\n")
        f.write(f"pos_weight   : {pos_weight:.1f}\n")
        f.write(f"model_cfg    : {MODEL_CFG}\n")
        f.write(f"train_cfg    : {TRAIN_CFG}\n")
    print(f"  Summary      : {summary_path}")

    plot_history(history, OUTPUT_DIR, RUN_ID)

    return model, history, model_path


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"Device   : {DEVICE}\n")

    model, history, model_path = train()

    print("\nBước tiếp theo:")
    print(f"  python evaluate.py --model {model_path}")