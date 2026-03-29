"""
check_dataset.py -- Validate daklak_final_dataset.parquet before XGBoost retraining

Checks:
  1. Schema -- all expected columns present, dtypes correct
  2. Date range & split sizes
  3. Fire label balance per split (scale_pos_weight hint)
  4. Missing values per feature per split
  5. Feature value sanity (range checks)
  6. Veg (NDVI) coverage by year
  7. Grid completeness
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── Config (mirrors xgb_config.py) ────────────────────────────────────────
PARQUET_PATH   = Path("daklak_final_dataset.parquet")
TRAIN_END_DATE = pd.Timestamp("2021-12-31")
VAL_END_DATE   = pd.Timestamp("2022-12-31")
TEST_END_DATE  = pd.Timestamp("2024-12-31")

CURRENT_FEATURE_COLS = [
    "tmean", "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    "wind_vpd",
    "fire_lag_1", "fire_lag_3",
    "dem_mean", "dem_stdev", "slp_mean", "slp_stdev", "aspect_sin", "aspect_cos",
    "sin_doy", "cos_doy",
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
]

NEW_VEG_COLS = ["ndvi", "ndvi_std", "ndwi", "nbr", "ndii",
                "delta_ndvi_14d", "delta_nbr_7d", "has_s2"]

EXPECTED_RANGES = {
    "tmean":          (-10,   50),
    "rh":             (  0,  100),
    "wind":           (  0,   30),
    "rain":           (  0,  500),
    "vpd":            (  0,   10),
    "dem_mean":       (  0, 3000),
    "ndvi":           ( -1,    1),
    "ndwi":           ( -1,    1),
    "nbr":            ( -1,    1),
    "ndii":           ( -1,    1),
    "sin_doy":        ( -1,    1),
    "cos_doy":        ( -1,    1),
    "neighbor_count": (  0,   20),
    "has_s2":         (  0,    1),
}

OK   = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

errors   = []
warnings = []

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)

def read_split(filters):
    df = pd.read_parquet(PARQUET_PATH, filters=filters)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── 0. File exists ─────────────────────────────────────────────────────────
section("0. File check")
if not PARQUET_PATH.exists():
    print(f"{FAIL} {PARQUET_PATH} not found -- run build_dataset.py first")
    sys.exit(1)

meta    = pq.read_metadata(PARQUET_PATH)
size_gb = PARQUET_PATH.stat().st_size / 1e9
print(f"{OK}  File found : {PARQUET_PATH}")
print(f"     Rows       : {meta.num_rows:,}")
print(f"     Row groups : {meta.num_row_groups}")
print(f"     File size  : {size_gb:.2f} GB")

# ── 1. Schema (no full load needed) ───────────────────────────────────────
section("1. Schema")
schema_pq = pq.read_schema(PARQUET_PATH)
cols      = set(schema_pq.names)

missing_current = [c for c in CURRENT_FEATURE_COLS if c not in cols]
present_new_veg = [c for c in NEW_VEG_COLS if c in cols]
absent_new_veg  = [c for c in NEW_VEG_COLS if c not in cols]

if missing_current:
    print(f"{FAIL} Missing FEATURE_COLS (fix build_dataset.py or xgb_config.py):")
    for c in missing_current:
        print(f"       - {c}")
    errors.append(f"Missing features: {missing_current}")
else:
    print(f"{OK}  All current FEATURE_COLS present")

if present_new_veg:
    print(f"{OK}  New veg columns ready: {present_new_veg}")
if absent_new_veg:
    print(f"{WARN} Veg columns missing (merge may have failed): {absent_new_veg}")
    warnings.append(f"Absent veg cols: {absent_new_veg}")

# Dtype check via tiny sample (one grid cell)
sample = read_split([("grid_id", "==", 75)])
print(f"\n  Dtypes (sample grid_id=75, {len(sample):,} rows):")
for col in sorted(cols):
    dtype = str(sample[col].dtype) if col in sample.columns else "?"
    flag  = ""
    if col in CURRENT_FEATURE_COLS + NEW_VEG_COLS and dtype == "float64":
        flag = f"  <- {WARN} float64 (should be float32)"
        warnings.append(f"{col} is float64")
    print(f"    {col:<24} {dtype}")

del sample

# ── 2. Read splits via parquet filters (avoids OOM from slicing full df) ──
section("2. Date range & splits")
print("  Reading each split directly from parquet (memory-efficient)...")

train = read_split([("date", "<=", TRAIN_END_DATE)])
val   = read_split([("date", ">",  TRAIN_END_DATE), ("date", "<=", VAL_END_DATE)])
test  = read_split([("date", ">",  VAL_END_DATE)])

print(f"  Date range : {train['date'].min().date()}  ->  {test['date'].max().date()}")
print(f"  Expected   : 2015-01-01 -> {TEST_END_DATE.date()}")
print(f"\n  Split sizes:")
print(f"    Train (<= {TRAIN_END_DATE.date()}) : {len(train):>12,} rows")
print(f"    Val   (<= {VAL_END_DATE.date()})   : {len(val):>12,} rows")
print(f"    Test  (>  {VAL_END_DATE.date()})   : {len(test):>12,} rows")

for name, split in [("Train", train), ("Val", val), ("Test", test)]:
    if len(split) == 0:
        errors.append(f"{name} split is empty")

# ── 3. Fire label balance ──────────────────────────────────────────────────
section("3. Fire label balance")
print(f"  {'Split':<8} {'Total':>10} {'Fire=1':>10} {'Fire=0':>10} "
      f"{'Pos%':>8} {'scale_pos_weight':>18}")
print(f"  {'-' * 62}")
for name, split in [("Train", train), ("Val", val), ("Test", test)]:
    n1  = int((split["fire"] == 1).sum())
    n0  = int((split["fire"] == 0).sum())
    pct = 100 * n1 / len(split) if len(split) else 0
    spw = round(n0 / n1, 1) if n1 > 0 else float("inf")
    note = f"  <- {WARN} very low fire rate" if pct < 0.05 else ""
    print(f"  {name:<8} {len(split):>10,} {n1:>10,} {n0:>10,} "
          f"{pct:>7.3f}% {spw:>18.1f}{note}")

# ── 4. Missing values per split ───────────────────────────────────────────
section("4. Missing values")
check_cols = [c for c in CURRENT_FEATURE_COLS + NEW_VEG_COLS if c in cols]

for name, split in [("Train", train), ("Val", val), ("Test", test)]:
    missing = split[check_cols].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  {OK}  {name}: no missing values")
    else:
        print(f"  {WARN} {name}: missing values found:")
        for col, cnt in missing.items():
            pct  = 100 * cnt / len(split)
            flag = FAIL if pct > 5 else WARN
            print(f"    {flag} {col:<24}: {cnt:>10,}  ({pct:.2f}%)")
            if pct > 5:
                errors.append(f"{name}/{col}: {pct:.1f}% missing")
            else:
                warnings.append(f"{name}/{col}: {pct:.2f}% missing")

# ── 5. Feature value sanity (train split) ─────────────────────────────────
section("5. Feature sanity (train split)")
for col, (lo, hi) in EXPECTED_RANGES.items():
    if col not in train.columns:
        continue
    mn = float(train[col].min())
    mx = float(train[col].max())
    out = (mn < lo) or (mx > hi)
    flag = WARN if out else OK
    note = f"  <- expected [{lo}, {hi}]" if out else ""
    print(f"  {flag} {col:<24} min={mn:>10.3f}  max={mx:>10.3f}{note}")
    if out:
        warnings.append(f"{col} out of range [{mn:.3f}, {mx:.3f}]")

# ── 6. Veg coverage by year ───────────────────────────────────────────────
if "has_s2" in cols:
    section("6. Veg (NDVI) coverage by year")
    # Read only date + has_s2 to avoid loading all columns
    veg_check = pd.read_parquet(PARQUET_PATH, columns=["date", "has_s2"])
    veg_check["date"] = pd.to_datetime(veg_check["date"])
    print(f"  {'Year':<6} {'Total rows':>12} {'has_s2=1':>10} {'Coverage':>10}")
    print(f"  {'-' * 42}")
    for year, grp in veg_check.groupby(veg_check["date"].dt.year):
        n_s2 = int((grp["has_s2"] == 1).sum())
        pct  = 100 * n_s2 / len(grp)
        note = f"  <- {WARN} low" if pct < 5 else ""
        print(f"  {year:<6} {len(grp):>12,} {n_s2:>10,} {pct:>9.1f}%{note}")
        if pct < 5:
            warnings.append(f"Year {year}: low veg coverage {pct:.1f}%")
    del veg_check
else:
    section("6. Veg coverage")
    print(f"  {WARN} has_s2 not found -- veg merge may have failed")

# ── 7. Grid completeness (train split) ────────────────────────────────────
section("7. Grid completeness (train split)")
n_grids      = train["grid_id"].nunique()
n_dates      = train["date"].nunique()
expected     = n_grids * n_dates
actual       = len(train)
pct_complete = 100 * actual / expected
print(f"  Unique grid cells : {n_grids:,}")
print(f"  Unique dates      : {n_dates:,}")
print(f"  Expected rows     : {expected:,}")
print(f"  Actual rows       : {actual:,}  ({pct_complete:.1f}%)")
if pct_complete < 99:
    print(f"  {WARN} Completeness below 99%")
    warnings.append(f"Grid completeness: {pct_complete:.1f}%")
else:
    print(f"  {OK}  Grid completeness OK")

# ── Summary ───────────────────────────────────────────────────────────────
section("SUMMARY")
if errors:
    print(f"  {FAIL} {len(errors)} error(s) -- fix before retraining:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  {OK}  No errors -- data looks ready to retrain")

if warnings:
    print(f"\n  {WARN} {len(warnings)} warning(s):")
    for w in warnings:
        print(f"    - {w}")

if not errors:
    print(f"\n  Suggested next steps:")
    if missing_current:
        print(f"    1. Fix xgb_config.FEATURE_COLS -- remove: {missing_current}")
        print(f"       or add wind_vpd/aspect_sin/aspect_cos to build_dataset.py")
    if present_new_veg:
        print(f"    2. Add new veg features to xgb_config.FEATURE_COLS:")
        print(f"       {present_new_veg}")
    print(f"    3. Run: python xgb_train.py")