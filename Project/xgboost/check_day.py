import sys
import pandas as pd
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from xgb_config import DATA_PATH, FEATURE_COLS

MODEL_PATH = Path(__file__).parent / "models/xgb_human_features_tuned_v4_pathways.json"

# ── Config ─────────────────────────────────────────────────────────────────────
CHECK_DATE = "2023-01-14"   # <-- change this
CHECK_GRID = 31363          # <-- change this

# ── Load ───────────────────────────────────────────────────────────────────────
parquet_path = Path(__file__).parent.parent / DATA_PATH.lstrip("../")
row = pd.read_parquet(
    parquet_path,
    columns=["grid_id", "date"] + FEATURE_COLS + ["fire"],
    filters=[
        ("date", "==", pd.Timestamp(CHECK_DATE)),
        ("grid_id", "==", CHECK_GRID),
    ],
).copy()

if row.empty:
    print(f"No data for grid_id={CHECK_GRID} on {CHECK_DATE}")
    sys.exit(1)

# ── Predict ────────────────────────────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
row["fire_prob"] = model.predict_proba(row[FEATURE_COLS])[:, 1]

# ── Report ─────────────────────────────────────────────────────────────────────
r = row.iloc[0]
print(f"\n{'='*45}")
print(f"  Grid {CHECK_GRID}  |  Date: {CHECK_DATE}")
print(f"{'='*45}")
print(f"  fire label : {int(r['fire'])}")
print(f"  fire_prob  : {r['fire_prob']:.6f}")
print(f"{'-'*45}")
for feat in FEATURE_COLS:
    print(f"  {feat:<22} {r[feat]:.6f}")
print(f"{'='*45}")