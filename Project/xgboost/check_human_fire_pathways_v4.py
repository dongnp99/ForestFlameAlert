"""
check_human_fire_pathways_v4.py
================================
Changes from V3:
  - P1: drop cropland_frac_1km (anti-discriminatory, ratio < 1.0x)
        keep burn_season_flag & days_since_harvest < 30
  - P2: unchanged — deforestation_lag_1y > 1.5 & fire_count_prev_year > 0
  - P3: drop dist_road_km (anti-discriminatory, ratio 0.85x)
        tighten fire_count_prev_year to > 1 for better precision
  - Rain bonus: DROPPED — rain_14d_sum is anti-correlated with fire (ratio 0.33-0.48x)
"""

import sys
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, ".")
import xgb_config

# =============================================================
# 1. LOAD DATA
# =============================================================

COLS_NEEDED = [
    "fire",
    "burn_season_flag",
    "days_since_harvest",
    "cropland_frac_1km",
    "deforestation_lag_1y",
    "fire_count_prev_year",
    "dist_road_km",
    "date",
]

print("=" * 60)
print("Loading data...")
print("=" * 60)

df = pd.read_parquet(
    xgb_config.DATA_PATH,
    columns=COLS_NEEDED,
    engine="pyarrow",
)

total    = len(df)
fire_df  = df[df["fire"] == 1]
n_fire   = len(fire_df)
n_nofire = total - n_fire

print(f"Total samples     : {total:>12,}")
print(f"  Fire = 1        : {n_fire:>12,}  ({n_fire/total:.4%})")
print(f"  Fire = 0        : {n_nofire:>12,}")
print()

# =============================================================
# 2. PATHWAY DEFINITIONS
# =============================================================

# ── V4 pathways ──

# P1: post-harvest agricultural burning (cropland_frac dropped)
p1 = (
    (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 30)
)

# P2: active deforestation with fire history (unchanged from V3)
p2 = (
    (df["deforestation_lag_1y"] > 1.5)
    & (df["fire_count_prev_year"] > 0)
)

# P3: repeat ignition during burn season (dist_road_km dropped, threshold tightened)
p3 = (
    (df["fire_count_prev_year"] > 1)
    & (df["burn_season_flag"] == 1)
)

human_fire_mask = (df["fire"] == 1) & (p1 | p2 | p3)

# ── Comparison baselines ──

# V3
p1_v3 = (
    (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 30)
    & (df["cropland_frac_1km"] > 0.2)
)
p2_v3 = (
    (df["deforestation_lag_1y"] > 1.5)
    & (df["fire_count_prev_year"] > 0)
)
p3_v3 = (
    (df["fire_count_prev_year"] > 0)
    & (df["dist_road_km"] < 1.0)
    & (df["burn_season_flag"] == 1)
)
v3_mask = (df["fire"] == 1) & (p1_v3 | p2_v3 | p3_v3)

# OLD (AND)
old_mask = (
    (df["fire"] == 1)
    & (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 14)
    & (df["deforestation_lag_1y"] > 1.5)
    & (df["rain_14d_sum"] > 500)
) if "rain_14d_sum" in df.columns else pd.Series(False, index=df.index)

# =============================================================
# 3. PATHWAY MATCH RATES
# =============================================================

print("=" * 60)
print("PATHWAY MATCH RATES  (V4)")
print("=" * 60)

pathways = {
    "P1 (burn_season + harvest<30)":   p1,
    "P2 (deforest + fire_hist>0)":     p2,
    "P3 (fire_hist>1 + burn_season)":  p3,
}

for name, mask in pathways.items():
    on_fire    = (mask & (df["fire"] == 1)).sum()
    on_nofire  = (mask & (df["fire"] == 0)).sum()
    fire_rate  = on_fire  / n_fire   if n_fire   else 0
    nofire_rate = on_nofire / n_nofire if n_nofire else 0
    sel  = fire_rate / nofire_rate if nofire_rate > 0 else float("inf")
    flag = "✓" if sel >= 1.5 else "✗"
    print(f"\n  {name}:")
    print(f"    Matches fire=1   : {on_fire:>10,} / {n_fire:,}  = {fire_rate:.2%}")
    print(f"    Matches fire=0   : {on_nofire:>10,} / {n_nofire:,}  = {nofire_rate:.2%}")
    print(f"    Selectivity ratio: {sel:.2f}x  {flag}")

print(f"\n  -- COMBINED V4 (p1 | p2 | p3) & fire=1 --")
print(f"    Matches: {human_fire_mask.sum():>10,} / {n_fire:,}  = {human_fire_mask.sum()/n_fire:.2%}")
print()

# =============================================================
# 4. OVERLAP BETWEEN PATHWAYS (fire=1)
# =============================================================

print("=" * 60)
print("OVERLAP BETWEEN PATHWAYS (fire=1 samples)")
print("=" * 60)

p1_fire = p1 & (df["fire"] == 1)
p2_fire = p2 & (df["fire"] == 1)
p3_fire = p3 & (df["fire"] == 1)

pw_fire = {"P1": p1_fire, "P2": p2_fire, "P3": p3_fire}

for (a_name, a_mask), (b_name, b_mask) in combinations(pw_fire.items(), 2):
    overlap = (a_mask & b_mask).sum()
    union   = (a_mask | b_mask).sum()
    jaccard = overlap / union if union else 0
    print(f"  {a_name} n {b_name} : {overlap:>8,}  (Jaccard = {jaccard:.2%})")

all_three = (p1_fire & p2_fire & p3_fire).sum()
print(f"  P1 n P2 n P3    : {all_three:>8,}")
print()

# =============================================================
# 5. GRADUATED WEIGHTING (no rain bonus)
# =============================================================

print("=" * 60)
print("GRADUATED WEIGHTING (fire=1 only, no rain bonus)")
print("=" * 60)

fire_mask_bool = (df["fire"] == 1).values

score = (
    p1[fire_mask_bool].astype(int).values
    + p2[fire_mask_bool].astype(int).values
    + p3[fire_mask_bool].astype(int).values
)

weights_map = {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.5}

for s in range(4):
    cnt = (score == s).sum()
    pct = cnt / n_fire if n_fire else 0
    w   = weights_map[s]
    print(f"  score={s}  ->  {cnt:>10,}  ({pct:.2%})  weight={w:.1f}x")

print()

# =============================================================
# 6. MATCH RATE BY SPLIT
# =============================================================

print("=" * 60)
print("MATCH RATE BY SPLIT")
print("=" * 60)

df["_date"] = pd.to_datetime(df["date"])

splits = {
    "train": df["_date"] <= pd.Timestamp(xgb_config.TRAIN_END_DATE),
    "val":  (df["_date"] > pd.Timestamp(xgb_config.TRAIN_END_DATE))
            & (df["_date"] <= pd.Timestamp(xgb_config.VAL_END_DATE)),
    "test":  df["_date"] > pd.Timestamp(xgb_config.VAL_END_DATE),
}

for split_name, split_mask in splits.items():
    s_fire  = (split_mask & (df["fire"] == 1)).sum()
    s_human = (split_mask & human_fire_mask).sum()
    rate    = s_human / s_fire if s_fire else 0
    s_p1 = (split_mask & p1_fire).sum()
    s_p2 = (split_mask & p2_fire).sum()
    s_p3 = (split_mask & p3_fire).sum()
    print(f"  {split_name:<6}  fire={s_fire:>8,}  human={s_human:>8,}  "
          f"rate={rate:.2%}  |  P1={s_p1:,}  P2={s_p2:,}  P3={s_p3:,}")

print()

# =============================================================
# 7. EFFECTIVE WEIGHT (train set)
# =============================================================

print("=" * 60)
print("EFFECTIVE WEIGHT (train set)")
print("=" * 60)

train_mask = df["_date"] <= pd.Timestamp(xgb_config.TRAIN_END_DATE)
train_fire = (train_mask & (df["fire"] == 1)).values

score_train = (
    p1[train_fire].astype(int).values
    + p2[train_fire].astype(int).values
    + p3[train_fire].astype(int).values
)

w = np.array([weights_map.get(s, 3.5) for s in score_train], dtype="float32")

n_train_fire  = train_fire.sum()
n_train_total = train_mask.sum()

print(f"  Train fire samples    : {n_train_fire:>10,}")
print(f"  Mean weight (fire=1)  : {w.mean():.4f}")
print(f"  Median weight         : {np.median(w):.4f}")
print(f"  Total weight (fire=1) : {w.sum():>12,.1f}")
print(f"  Effective fire ratio  : "
      f"{w.sum() / (n_train_total - n_train_fire + w.sum()):.6f}"
      f"  (vs raw {n_train_fire / n_train_total:.6f})")

print(f"\n  Weight distribution (train fire samples):")
for wval in sorted(set(weights_map.values())):
    cnt = (w == wval).sum()
    print(f"    weight={wval:.1f}  ->  {cnt:>8,}  ({cnt/n_train_fire:.2%})")

print()

# =============================================================
# 8. COMPARISON: OLD vs V3 vs V4
# =============================================================

print("=" * 60)
print("COMPARISON: OLD (AND) vs V3 vs V4")
print("=" * 60)

for label, mask in [("OLD (AND)", old_mask), ("V3", v3_mask), ("V4", human_fire_mask)]:
    cnt  = mask.sum()
    rate = cnt / n_fire if n_fire else 0
    # selectivity: what fraction of matched samples are fire=1
    precision = cnt / (mask.sum() + (mask & (df["fire"] == 0)).sum()) if mask.sum() > 0 else 0
    n_matched_nofire = (mask & (df["fire"] == 0)).sum()
    sel_ratio = (cnt / n_fire) / (n_matched_nofire / n_nofire) if n_matched_nofire > 0 else float("inf")
    print(f"  {label:<10}  matched={cnt:>8,} / {n_fire:,}  = {rate:.2%}  "
          f"selectivity={sel_ratio:.2f}x  precision={precision:.4%}")

print()
print("=" * 60)
print("DONE -- V4")
print("=" * 60)
