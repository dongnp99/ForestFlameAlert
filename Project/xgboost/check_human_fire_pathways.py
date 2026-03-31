"""
check_human_fire_pathways_v3.py
================================
Kiểm tra tỷ lệ match của 3 pathway nhận diện cháy nhân tạo — V3.

Thay đổi so với v2:
  - P2: thay tree_cover_pct bằng fire_count_prev_year (tiền sử cháy)
  - P3: bỏ logic "mưa" → chuyển sang "repeat burning near infrastructure"
  - Rain trở thành bonus modifier (không phải pathway riêng)
  - Graduated weighting tính cả rain bonus

Chạy: python check_human_fire_pathways_v3.py
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
    "dist_forest_edge_km",
    "rain_14d_sum",
    "dist_settlement_km",
    "dist_road_km",
    "tree_cover_pct",
    "fire_count_prev_year",
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
# 2. PHÂN PHỐI CÁC FEATURE (chỉ fire=1)
# =============================================================

print("=" * 60)
print("PHÂN PHỐI FEATURE (chỉ fire=1 samples)")
print("=" * 60)

# Tất cả feature dùng trong pathway + modifier
stats_config = {
    "burn_season_flag":     {"thresholds": [("== 1", lambda s: s == 1)]},
    "days_since_harvest":   {"thresholds": [(f"< {t}", lambda s, t=t: s < t) for t in [14, 21, 30, 45]]},
    "cropland_frac_1km":    {"thresholds": [(f"> {t}", lambda s, t=t: s > t) for t in [0.1, 0.2, 0.3, 0.5]]},
    "deforestation_lag_1y": {"thresholds": [(f"> {t}", lambda s, t=t: s > t) for t in [1.0, 1.5, 2.0, 3.0]]},
    "fire_count_prev_year": {"thresholds": [(f"> {t}", lambda s, t=t: s > t) for t in [0, 1, 3, 5]]},
    "dist_road_km":         {"thresholds": [(f"< {t}", lambda s, t=t: s < t) for t in [0.5, 1.0, 2.0, 3.0]]},
    "dist_settlement_km":   {"thresholds": [(f"< {t}", lambda s, t=t: s < t) for t in [1.0, 2.0, 3.0, 5.0]]},
    "rain_14d_sum":         {"thresholds": [(f"> {t}", lambda s, t=t: s > t) for t in [200, 300, 400, 500]]},
    "dist_forest_edge_km":  {"thresholds": [(f"< {t}", lambda s, t=t: s < t) for t in [0.5, 1.0, 2.0]]},
    "tree_cover_pct":       {"thresholds": [(f"> {t}", lambda s, t=t: s > t) for t in [15, 30, 50, 70]]},
}

for col, cfg in stats_config.items():
    s_fire   = fire_df[col]
    s_all    = df[col]
    print(f"\n  {col}:")
    print(f"    [fire=1]  mean={s_fire.mean():.4f}  median={s_fire.median():.4f}  "
          f"std={s_fire.std():.4f}")
    print(f"    [fire=1]  min={s_fire.min():.4f}  25%={s_fire.quantile(0.25):.4f}  "
          f"75%={s_fire.quantile(0.75):.4f}  max={s_fire.max():.4f}")

    # So sánh fire vs non-fire ở từng ngưỡng
    for label, fn in cfg["thresholds"]:
        pct_fire   = fn(s_fire).mean()
        pct_nofire = fn(df.loc[df["fire"] == 0, col]).mean()
        ratio      = pct_fire / pct_nofire if pct_nofire > 0 else float("inf")
        marker     = "✓" if ratio > 1.5 else "~" if ratio > 1.0 else "✗"
        print(f"    → {label:<8}  fire={pct_fire:.2%}  nofire={pct_nofire:.2%}  "
              f"ratio={ratio:.2f}x {marker}")

print()

# =============================================================
# 3. PATHWAY DEFINITIONS — V3
# =============================================================

print("=" * 60)
print("PATHWAY MATCH RATES  (V3)")
print("=" * 60)

# ── Pathway 1: post-harvest agricultural burning ──
p1 = (
    (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 30)
    & (df["cropland_frac_1km"] > 0.2)
)

# ── Pathway 2: deforestation / land clearing with fire history ──
p2 = (
    (df["deforestation_lag_1y"] > 1.5)
    & (df["fire_count_prev_year"] > 0)
)

# ── Pathway 3: repeat burning near road infrastructure ──
p3 = (
    (df["fire_count_prev_year"] > 0)
    & (df["dist_road_km"] < 1.0)
    & (df["burn_season_flag"] == 1)
)

# ── Rain bonus modifier (NOT a standalone pathway) ──
rain_bonus = (df["rain_14d_sum"] > 300)

# ── Combined ──
human_fire_mask = (df["fire"] == 1) & (p1 | p2 | p3)

# ── Logic cũ (AND) ──
old_mask = (
    (df["fire"] == 1)
    & (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 14)
    & (df["deforestation_lag_1y"] > 1.5)
    & (df["rain_14d_sum"] > 500)
)

# ── V2 logic để so sánh ──
p1_v2 = (
    (df["burn_season_flag"] == 1)
    & (df["days_since_harvest"] < 30)
    & (df["cropland_frac_1km"] > 0.2)
)
p2_v2 = (
    (df["deforestation_lag_1y"] > 1.5)
    & (df["tree_cover_pct"].between(15, 70))
)
p3_v2 = (
    (df["rain_14d_sum"] > 300)
    & (df["cropland_frac_1km"] > 0.2)
    & (df["burn_season_flag"] == 1)
)
v2_mask = (df["fire"] == 1) & (p1_v2 | p2_v2 | p3_v2)

# ── Report ──
pathways = {
    "P1 (post-harvest)":       p1,
    "P2 (deforest+history)":   p2,
    "P3 (repeat+road+season)": p3,
}

for name, mask in pathways.items():
    on_fire   = (mask & (df["fire"] == 1)).sum()
    on_nofire = (mask & (df["fire"] == 0)).sum()
    fire_rate  = on_fire / n_fire if n_fire else 0
    nofire_rate = on_nofire / n_nofire if n_nofire else 0
    sel = fire_rate / nofire_rate if nofire_rate > 0 else float("inf")
    flag = "✓" if sel >= 1.5 else "✗"
    print(f"\n  {name}:")
    print(f"    Matches fire=1   : {on_fire:>10,} / {n_fire:,}  = {fire_rate:.2%}")
    print(f"    Matches fire=0   : {on_nofire:>10,} / {n_nofire:,}  = {nofire_rate:.2%}")
    print(f"    Selectivity ratio: {sel:.2f}x  {flag}")

# Rain bonus stats (chỉ trên fire samples đã match pathway)
rain_on_matched   = (human_fire_mask & rain_bonus).sum()
rain_on_unmatched = ((df["fire"] == 1) & ~(p1 | p2 | p3) & rain_bonus).sum()
print(f"\n  ── RAIN BONUS (modifier, not pathway) ──")
print(f"    Matched pathway + rain>300  : {rain_on_matched:>8,}")
print(f"    No pathway + rain>300       : {rain_on_unmatched:>8,}  (ignored)")

print(f"\n  ── COMBINED V3 (p1 | p2 | p3) & fire=1 ──")
print(f"    Matches: {human_fire_mask.sum():>10,} / {n_fire:,}  = {human_fire_mask.sum()/n_fire:.2%}")

print(f"\n  ── V2 LOGIC (previous iteration) & fire=1 ──")
print(f"    Matches: {v2_mask.sum():>10,} / {n_fire:,}  = {v2_mask.sum()/n_fire:.2%}")

print(f"\n  ── OLD LOGIC (AND) & fire=1 ──")
print(f"    Matches: {old_mask.sum():>10,} / {n_fire:,}  = {old_mask.sum()/n_fire:.2%}")

print()

# =============================================================
# 4. OVERLAP GIỮA CÁC PATHWAY (fire=1)
# =============================================================

print("=" * 60)
print("OVERLAP GIỮA CÁC PATHWAY (trên fire=1 samples)")
print("=" * 60)

p1_fire = p1 & (df["fire"] == 1)
p2_fire = p2 & (df["fire"] == 1)
p3_fire = p3 & (df["fire"] == 1)

pw_fire = {"P1": p1_fire, "P2": p2_fire, "P3": p3_fire}

for (a_name, a_mask), (b_name, b_mask) in combinations(pw_fire.items(), 2):
    overlap = (a_mask & b_mask).sum()
    union   = (a_mask | b_mask).sum()
    jaccard = overlap / union if union else 0
    print(f"  {a_name} ∩ {b_name} : {overlap:>8,}  "
          f"(Jaccard = {jaccard:.2%})")

all_three = (p1_fire & p2_fire & p3_fire).sum()
print(f"  P1 ∩ P2 ∩ P3    : {all_three:>8,}")
print()

# =============================================================
# 5. GRADUATED WEIGHTING + RAIN BONUS
# =============================================================

print("=" * 60)
print("GRADUATED WEIGHTING + RAIN BONUS (fire=1 only)")
print("=" * 60)

fire_mask_bool = (df["fire"] == 1).values

# Base score = number of pathways matched
score = (
    p1[fire_mask_bool].astype(int).values
    + p2[fire_mask_bool].astype(int).values
    + p3[fire_mask_bool].astype(int).values
)

# Rain bonus: +1 if already matched ≥1 pathway AND rain>300
rain_eligible = (score > 0) & rain_bonus[fire_mask_bool].values
score_with_rain = score + rain_eligible.astype(int)

print("  --- Without rain bonus ---")
for s in range(4):
    cnt = (score == s).sum()
    pct = cnt / n_fire if n_fire else 0
    print(f"    score={s}  →  {cnt:>10,}  ({pct:.2%})")

print()
print("  --- With rain bonus ---")
weights_map = {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.0}
for s in range(5):
    cnt = (score_with_rain == s).sum()
    pct = cnt / n_fire if n_fire else 0
    w   = weights_map.get(s, 4.0)
    print(f"    score={s}  →  {cnt:>10,}  ({pct:.2%})  weight={w:.1f}x")

print()

# =============================================================
# 6. MATCH RATE THEO SPLIT
# =============================================================

print("=" * 60)
print("MATCH RATE THEO SPLIT")
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

    # Per-pathway breakdown within this split
    s_p1 = (split_mask & p1_fire).sum()
    s_p2 = (split_mask & p2_fire).sum()
    s_p3 = (split_mask & p3_fire).sum()

    print(f"  {split_name:<6}  fire={s_fire:>8,}  human={s_human:>8,}  "
          f"rate={rate:.2%}  |  P1={s_p1:,}  P2={s_p2:,}  P3={s_p3:,}")

print()

# =============================================================
# 7. EFFECTIVE WEIGHT (graduated + rain bonus, train set only)
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
rain_elig_train = (score_train > 0) & rain_bonus[train_fire].values
score_train_full = score_train + rain_elig_train.astype(int)

w = np.array([weights_map.get(s, 4.0) for s in score_train_full], dtype="float32")

n_train_fire  = train_fire.sum()
n_train_total = train_mask.sum()

print(f"  Train fire samples    : {n_train_fire:>10,}")
print(f"  Mean weight (fire=1)  : {w.mean():.4f}")
print(f"  Median weight         : {np.median(w):.4f}")
print(f"  Total weight (fire=1) : {w.sum():>12,.1f}")
print(f"  Effective fire ratio  : "
      f"{w.sum() / (n_train_total - n_train_fire + w.sum()):.6f}"
      f"  (vs raw {n_train_fire / n_train_total:.6f})")

# Weight distribution
print(f"\n  Weight distribution (train fire samples):")
for wval in sorted(set(weights_map.values())):
    cnt = (w == wval).sum()
    print(f"    weight={wval:.1f}  →  {cnt:>8,}  ({cnt/n_train_fire:.2%})")

print()

# =============================================================
# 8. SO SÁNH 3 PHIÊN BẢN
# =============================================================

print("=" * 60)
print("SO SÁNH: OLD (AND) vs V2 vs V3")
print("=" * 60)

for label, mask in [("OLD (AND)", old_mask), ("V2", v2_mask), ("V3", human_fire_mask)]:
    cnt = mask.sum()
    rate = cnt / n_fire if n_fire else 0
    print(f"  {label:<10}  matched={cnt:>8,} / {n_fire:,}  = {rate:.2%}")

print()
print("=" * 60)
print("DONE — V3")
print("=" * 60)