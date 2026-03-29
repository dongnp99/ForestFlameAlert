import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix,
)
import matplotlib.pyplot as plt
import sys

from s4_build_train_dataset import build_all_dataloaders
from s5_model import build_model
from config import OUTPUT_MODEL_PATH

# ================================================================
# CẤU HÌNH
# ================================================================
OUTPUT_DIR = Path(OUTPUT_MODEL_PATH)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# COLLECT PREDICTIONS
# ================================================================
def collect_predictions(model, loader, device):
    """Chạy model trên toàn bộ loader, trả về pred và target."""
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x    = x.to(device, non_blocking=True)
            pred = model(x)
            all_preds.append(pred.cpu().float().numpy().flatten())
            all_targets.append(y.float().numpy().flatten())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


# ================================================================
# TÍNH METRICS ĐẦY ĐỦ
# ================================================================
def compute_metrics(all_preds, all_targets):
    """
    Tính toàn bộ metrics cần thiết cho báo cáo.

    Returns dict với tất cả metrics.
    """
    auc_pr  = average_precision_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)

    # Threshold tối ưu theo F1
    precisions, recalls, thresholds = precision_recall_curve(
        all_targets, all_preds
    )
    f1s         = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx    = int(f1s.argmax())
    best_thresh = float(thresholds[best_idx]) \
                  if best_idx < len(thresholds) else 0.5

    y_pred = (all_preds >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_targets, y_pred).ravel()

    recall    = float(tp / (tp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
    csi       = float(tp / (tp + fp + fn + 1e-8))  # Critical Success Index
    fpr       = float(fp / (fp + tn + 1e-8))        # False Positive Rate

    return {
        "auc_pr":     auc_pr,
        "auc_roc":    auc_roc,
        "threshold":  best_thresh,
        "recall":     recall,
        "precision":  precision,
        "f1":         f1,
        "csi":        csi,
        "fpr":        fpr,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "precisions": precisions,
        "recalls":    recalls,
    }


# ================================================================
# VẼ PR CURVE
# ================================================================
def plot_pr_curve(metrics: dict, output_dir: Path, split: str = "test"):
    precisions = metrics["precisions"]
    recalls    = metrics["recalls"]
    auc_pr     = metrics["auc_pr"]
    best_rec   = metrics["recall"]
    best_prec  = metrics["precision"]
    best_thresh = metrics["threshold"]

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, color="tab:blue", lw=1.5,
             label=f"ConvLSTM (AUC-PR={auc_pr:.4f})")
    plt.axhline(y=0.0011, color="red", linestyle="--",
                label="Baseline (random)", alpha=0.7)
    plt.scatter(best_rec, best_prec, color="tab:orange", zorder=5, s=80,
                label=f"Best threshold={best_thresh:.3f}")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve — ConvLSTM ({split} set)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"pr_curve_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PR curve   → {path}")


# ================================================================
# EVALUATE TRÊN TEST SET
# ================================================================
def evaluate_test(model_path: str):
    print("=" * 50)
    print("EVALUATE CONVLSTM TRÊN TEST SET")
    print("=" * 50)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE,
                      weights_only=False)
    print(f"\nModel      : {model_path}")
    print(f"Best epoch : {ckpt['epoch']}")
    print(f"Val AUC-PR : {ckpt['val_auc_pr']:.4f}  (lúc training)")

    # Load dataloaders
    print("\nLoad data...")
    _, _, test_loader, pos_weight, info = build_all_dataloaders(
        seq_len    = ckpt["model_cfg"]["seq_len"],
        batch_size = 8,
    )

    # Load model
    model, _, _ = build_model(
        ckpt["model_cfg"],
        pos_weight = ckpt.get("pos_weight", pos_weight),
    )
    # Xử lý trường hợp model được lưu sau torch.compile
    # → keys có prefix "_orig_mod." cần strip bỏ
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    print(f"Model loaded → {DEVICE}")

    # Collect predictions
    print("\nChạy inference trên test set...")
    all_preds, all_targets = collect_predictions(model, test_loader, DEVICE)
    print(f"  Predictions : {len(all_preds):,} pixels")
    print(f"  Fire pixels : {int(all_targets.sum()):,} "
          f"({all_targets.mean()*100:.3f}%)")
    print(f"  Pred stats  : min={all_preds.min():.4f} "
          f"max={all_preds.max():.4f} "
          f"mean={all_preds.mean():.6f}")

    # Tính metrics
    metrics = compute_metrics(all_preds, all_targets)

    # In kết quả
    print(f"\n{'='*50}")
    print("KẾT QUẢ TRÊN TEST SET")
    print(f"{'='*50}")
    print(f"  AUC-PR     : {metrics['auc_pr']:.4f}  (baseline=0.0011)")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  Threshold  : {metrics['threshold']:.3f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  CSI        : {metrics['csi']:.4f}")
    print(f"  FPR        : {metrics['fpr']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP = {metrics['tp']:>8,}  |  FP = {metrics['fp']:>8,}")
    print(f"    FN = {metrics['fn']:>8,}  |  TN = {metrics['tn']:>8,}")

    # Lưu kết quả text
    result_path = OUTPUT_DIR / "test_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("CONVLSTM — TEST SET RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"model_path  : {model_path}\n")
        f.write(f"best_epoch  : {ckpt['epoch']}\n")
        f.write(f"val_auc_pr  : {ckpt['val_auc_pr']:.6f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"auc_pr      : {metrics['auc_pr']:.6f}\n")
        f.write(f"auc_roc     : {metrics['auc_roc']:.6f}\n")
        f.write(f"threshold   : {metrics['threshold']:.4f}\n")
        f.write(f"recall      : {metrics['recall']:.6f}\n")
        f.write(f"precision   : {metrics['precision']:.6f}\n")
        f.write(f"f1          : {metrics['f1']:.6f}\n")
        f.write(f"csi         : {metrics['csi']:.6f}\n")
        f.write(f"fpr         : {metrics['fpr']:.6f}\n")
        f.write(f"tp={metrics['tp']} fp={metrics['fp']} "
                f"fn={metrics['fn']} tn={metrics['tn']}\n")
    print(f"\n  Results    → {result_path}")

    # Vẽ PR curve
    plot_pr_curve(metrics, OUTPUT_DIR, split="test")

    return all_preds, all_targets, metrics


# ================================================================
# LƯU PROBABILITY MAPS CHO FUSION LAYER
# ================================================================
def save_all_prob_maps(model_path: str):
    """
    Lưu probability map của ConvLSTM trên train/val/test.
    Các file .npy này là input cho fusion layer.

    Output files:
        prob_map_train.npy  — xác suất cháy từng pixel (train set)
        prob_map_val.npy
        prob_map_test.npy
        targets_train.npy   — ground truth tương ứng
        targets_val.npy
        targets_test.npy
    """
    print("\n" + "=" * 50)
    print("LƯU PROBABILITY MAPS CHO FUSION LAYER")
    print("=" * 50)

    ckpt = torch.load(model_path, map_location=DEVICE,
                      weights_only=False)

    train_loader, val_loader, test_loader, pos_weight, _ = \
        build_all_dataloaders(
            seq_len    = ckpt["model_cfg"]["seq_len"],
            batch_size = 8,
        )

    model, _, _ = build_model(
        ckpt["model_cfg"],
        pos_weight = ckpt.get("pos_weight", pos_weight),
    )
    state_dict = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    for name, loader in [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]:
        print(f"\n  Processing {name} set...")
        preds, targets = collect_predictions(model, loader, DEVICE)

        pred_path   = OUTPUT_DIR / f"prob_map_{name}.npy"
        target_path = OUTPUT_DIR / f"targets_{name}.npy"

        np.save(pred_path,   preds)
        np.save(target_path, targets)

        auc = average_precision_score(targets, preds)
        print(f"    Samples  : {len(preds):,}")
        print(f"    AUC-PR   : {auc:.4f}")
        print(f"    Saved    → {pred_path}")
        print(f"    Saved    → {target_path}")

    print("\nHoàn tất — các file prob_map_*.npy sẵn sàng cho fusion layer.")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    # Tìm best model tự động nếu không truyền argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        candidates = sorted(OUTPUT_DIR.glob("best_model_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"Không tìm thấy best_model_*.pt trong {OUTPUT_DIR}"
            )
        model_path = str(candidates[-1])
        print(f"Auto-detected model: {model_path}")

    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {DEVICE}\n")

    # 1. Evaluate trên test set
    all_preds, all_targets, metrics = evaluate_test(model_path)

    # 2. Lưu prob maps cho fusion
    save_all_prob_maps(model_path)

    print("\n" + "=" * 50)
    print("CONVLSTM HOÀN TẤT")
    print("=" * 50)
    print("Files sẵn sàng cho fusion layer:")
    for name in ["train", "val", "test"]:
        print(f"  prob_map_{name}.npy")
        print(f"  targets_{name}.npy")