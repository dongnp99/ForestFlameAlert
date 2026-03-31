"""
xgb_export_prob_maps.py
========================
Load trained XGBoost model, run inference on train/val/test splits,
save prob_map_*.npy + targets_*.npy để dùng cho fusion layer.

Fix OOM: đọc từng DATE_CHUNK ngày một thay vì load cả split.
"""

import gc
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

import xgb_config

# =============================
# CONFIG
# =============================

MODEL_PATH = "models/v3/xgb_human_features_tuned_v4_pathways.json"
OUTPUT_DIR = Path("models/prob_maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Số ngày xử lý mỗi lần — giảm nếu vẫn OOM
DATE_CHUNK = 30

# =============================
# LOGGING
# =============================

os.makedirs("logs", exist_ok=True)
_log_file = f"logs/xgb_export_prob_maps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file),
    ],
)

# =============================
# LOAD MODEL
# =============================

logging.info("Loading XGBoost model from: %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
logging.info("Model loaded.")


# =============================
# GET DATE RANGE FOR A SPLIT
# =============================

def get_dates(filters):
    """Lấy danh sách unique dates của split mà không load toàn bộ data."""
    df = pd.read_parquet(
        xgb_config.DATA_PATH,
        columns=["date"],
        filters=filters,
        engine="pyarrow",
    )
    dates = np.sort(df["date"].unique())
    del df
    gc.collect()
    return dates


# =============================
# CHUNKED INFERENCE
# =============================

def predict_and_save(split_name: str, filters: list):
    logging.info("=" * 50)
    logging.info("Processing split: %s", split_name)

    # Lấy toàn bộ ngày của split
    all_dates = get_dates(filters)
    logging.info("  Total dates: %d", len(all_dates))

    all_probs        = []
    all_targets      = []
    all_dates_meta   = []
    all_gridids_meta = []

    # Xử lý từng chunk ngày
    n_chunks = int(np.ceil(len(all_dates) / DATE_CHUNK))
    for chunk_idx in range(n_chunks):
        chunk_dates = all_dates[chunk_idx * DATE_CHUNK : (chunk_idx + 1) * DATE_CHUNK]
        date_start  = pd.Timestamp(chunk_dates[0])
        date_end    = pd.Timestamp(chunk_dates[-1])

        chunk_filters = [
            ("date", ">=", date_start),
            ("date", "<=", date_end),
        ]

        # Load only the columns we actually need — avoids reading unused cols
        chunk_df = pd.read_parquet(
            xgb_config.DATA_PATH,
            columns=xgb_config.FEATURE_COLS + ["fire", "date", "grid_id"],
            filters=chunk_filters,
            engine="pyarrow",
        )

        # Cast types in-place
        for col in xgb_config.FEATURE_COLS:
            if col in chunk_df.columns:
                chunk_df[col] = chunk_df[col].astype("float32")

        # Sort in-place — avoids creating a sorted copy of the chunk
        chunk_df.sort_values(["date", "grid_id"], inplace=True)

        # Save meta (cheap — numpy views of the sorted columns)
        all_dates_meta.append(chunk_df["date"].to_numpy())
        all_gridids_meta.append(chunk_df["grid_id"].to_numpy())
        del chunk_df["date"], chunk_df["grid_id"]

        # pop("fire") removes label column in-place → chunk_df == FEATURE_COLS only
        y = chunk_df.pop("fire").to_numpy(dtype=np.float32)

        # Pass chunk_df directly — no X copy needed
        dmat  = xgb.DMatrix(chunk_df)
        probs = model.predict(dmat).astype(np.float32)

        all_probs.append(probs)
        all_targets.append(y)

        del chunk_df, dmat, probs, y
        gc.collect()

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            logging.info("  Chunks done: %d / %d", chunk_idx + 1, n_chunks)

    # Concat
    all_probs        = np.concatenate(all_probs)
    all_targets      = np.concatenate(all_targets)
    all_dates_meta   = np.concatenate(all_dates_meta)
    all_gridids_meta = np.concatenate(all_gridids_meta)

    logging.info("  Total samples : %d", len(all_probs))
    logging.info("  Fire rate     : %.6f", all_targets.mean())

    # Metrics
    auc_pr  = average_precision_score(all_targets, all_probs)
    auc_roc = roc_auc_score(all_targets, all_probs)
    logging.info("  AUC-PR        : %.6f", auc_pr)
    logging.info("  AUC-ROC       : %.6f", auc_roc)
    logging.info("  Pred range    : [%.4f, %.4f]  mean=%.6f",
                 all_probs.min(), all_probs.max(), all_probs.mean())

    # Lưu numpy
    np.save(OUTPUT_DIR / f"xgb_prob_map_{split_name}.npy",  all_probs)
    np.save(OUTPUT_DIR / f"xgb_targets_{split_name}.npy",   all_targets)

    # Lưu meta để align với ConvLSTM
    meta_df = pd.DataFrame({
        "date":    all_dates_meta,
        "grid_id": all_gridids_meta,
    })
    meta_df.to_parquet(OUTPUT_DIR / f"xgb_meta_{split_name}.parquet", index=False)

    logging.info("  Saved → xgb_prob_map_%s.npy  (%d samples)", split_name, len(all_probs))
    logging.info("  Saved → xgb_targets_%s.npy",   split_name)
    logging.info("  Saved → xgb_meta_%s.parquet",  split_name)

    del all_probs, all_targets, all_dates_meta, all_gridids_meta, meta_df
    gc.collect()

    return auc_pr, auc_roc


# =============================
# RUN ALL SPLITS
# =============================

splits = {
    "train": [
        ("date", "<=", pd.Timestamp(xgb_config.TRAIN_END_DATE))
    ],
    "val": [
        ("date", ">",  pd.Timestamp(xgb_config.TRAIN_END_DATE)),
        ("date", "<=", pd.Timestamp(xgb_config.VAL_END_DATE)),
    ],
    "test": [
        ("date", ">", pd.Timestamp(xgb_config.VAL_END_DATE))
    ],
}

results = {}
for name, filters in splits.items():
    auc_pr, auc_roc = predict_and_save(name, filters)
    results[name] = {"auc_pr": auc_pr, "auc_roc": auc_roc}

# =============================
# SUMMARY
# =============================

logging.info("=" * 50)
logging.info("SUMMARY")
logging.info("=" * 50)
for name, m in results.items():
    logging.info("  %-5s  AUC-PR=%.4f  AUC-ROC=%.4f",
                 name, m["auc_pr"], m["auc_roc"])
logging.info("Output dir: %s", OUTPUT_DIR)
logging.info("Files ready for fusion:")
for name in splits:
    logging.info("  xgb_prob_map_%s.npy  /  xgb_targets_%s.npy  /  xgb_meta_%s.parquet",
                 name, name, name)
logging.info("Log written to: %s", _log_file)