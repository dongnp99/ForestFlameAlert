import pandas as pd

"""
data_sanity_check.py
====================
Kiểm tra bất thường cho dataset cháy rừng Đắk Lắk.

Các loại check:
  1. Missing values (% NaN)
  2. Range check (so với expected min/max vật lý)
  3. Distribution check (mean, median, std, percentiles)
  4. Outlier detection (% giá trị nằm ngoài range hợp lý)
  5. Constant columns (std = 0)
  6. Negative values cho biến không được âm
  7. Cardinality cho categorical
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

#  Path 
DATA_PATH = "daklak_final_dataset_v2_human.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED RANGES — based on Daklak climate, geography, and physics
# ══════════════════════════════════════════════════════════════════════════════
# Format: column → (min_expected, max_expected, unit, notes)

EXPECTED_RANGES = {
    #  ID & coords 
    "grid_id": (0, 1_000_000, "int", "Grid identifier"),
    "lon": (107.0, 109.5, "deg", "Daklak longitude bbox"),
    "lat": (12.0, 13.8, "deg", "Daklak latitude bbox"),

    #  Weather (daily) 
    "tmean": (10, 38, "°C", "Daily mean temperature"),
    "rh": (10, 100, "%", "Relative humidity"),
    "wind": (0, 15, "m/s", "Wind speed"),
    "rain": (0, 200, "mm/day", "Daily rainfall — > 200 is suspicious"),
    "vpd": (0, 6, "kPa", "Vapor pressure deficit"),

    #  Rolling weather 
    "rain_14d_sum": (0, 600, "mm", "14-day rain sum"),
    "rain_30d_sum": (0, 1200, "mm", "30-day rain sum"),
    "vpd_14d_mean": (0, 6, "kPa", "14-day VPD mean"),
    "vpd_30d_mean": (0, 6, "kPa", "30-day VPD mean"),

    #  Target 
    "fire": (0, 1, "binary", "Fire occurrence"),
    "fire_lag_1": (0, 1, "binary", "Fire 1 day ago"),
    "fire_lag_3": (0, 1, "binary", "Fire 3 days ago"),

    #  Terrain (DEM) 
    "dem_mean": (0, 2500, "m", "Elevation"),
    "dem_stdev": (0, 500, "m", "Elevation std within cell"),
    "dem_min": (0, 2500, "m", "Min elevation"),
    "dem_max": (0, 2500, "m", "Max elevation"),
    "slp_mean": (0, 90, "deg", "Slope angle"),
    "slp_stdev": (0, 45, "deg", "Slope std"),

    #  Vegetation indices 
    "ndvi": (-1, 1, "ratio", "Normalized Difference Vegetation Index"),
    "ndvi_std": (0, 0.5, "ratio", "NDVI std within cell"),
    "ndwi": (-1, 1, "ratio", "Water index"),
    "nbr": (-1, 1, "ratio", "Burn ratio"),
    "ndii": (-1, 1, "ratio", "Infrared index"),
    "delta_ndvi_14d": (-1, 1, "ratio", "Change in NDVI over 14d"),
    "delta_nbr_7d": (-1, 1, "ratio", "Change in NBR over 7d"),
    "has_s2": (0, 1, "binary", "Sentinel-2 availability flag"),

    #  Distances 
    "dist_road_km": (0, 50, "km", "Distance to road"),
    "dist_settlement_km": (0, 50, "km", "Distance to settlement"),
    "dist_forest_edge_km": (0, 50, "km", "Distance to forest edge"),
    "dist_powerline_km": (0, 100, "km", "Distance to powerline"),

    #  Land use 
    "lulc_class": (0, 100, "int", "ESA WorldCover class"),
    "cropland_frac_1km": (0, 1, "frac", "Cropland fraction"),
    "tree_cover_pct": (0, 100, "%", "Tree cover percent"),
    "deforestation_lag_1y": (0, 100, "frac/area", "Deforestation 1y ago"),

    #  Human 
    "nightlight_mean": (0, 100, "nW/cm²/sr", "VIIRS nightlight"),
    "pop_density": (0, 10000, "ppl/km²", "Population density"),

    #  Fire history 
    "fire_count_prev_year": (0, 100, "count", "Fires in prev year"),
    "fire_count_prev_3y": (0, 300, "count", "Fires in prev 3 years"),
    "days_since_last_fire": (0, 5000, "days", "Days since last fire"),

    #  Seasonality 
    "burn_season_flag": (0, 1, "binary", "Burn season indicator"),
    "days_since_harvest": (0, 365, "days", "Days since harvest"),
    "sin_doy": (-1, 1, "ratio", "Sine of day of year"),
    "cos_doy": (-1, 1, "ratio", "Cosine of day of year"),

    #  Neighbors 
    "neighbor_count": (0, 50, "count", "Number of neighbor cells"),
    "neighbor_fire_1d": (0, 50, "count", "Neighbor fires 1d"),
    "neighbor_fire_3d": (0, 50, "count", "Neighbor fires 3d"),
    "neighbor_fire_7d": (0, 50, "count", "Neighbor fires 7d"),
    "vpd_neighbor_1d": (0, 6, "kPa", "Neighbor VPD"),
    "vpd_fire_lag_1": (0, 6, "kPa", "VPD with fire lag"),
}

# Cột không được phép âm
NON_NEGATIVE_COLS = {
    "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    "dem_mean", "dem_stdev", "dem_min", "dem_max",
    "slp_mean", "slp_stdev", "ndvi_std",
    "dist_road_km", "dist_settlement_km", "dist_forest_edge_km", "dist_powerline_km",
    "cropland_frac_1km", "tree_cover_pct", "deforestation_lag_1y",
    "nightlight_mean", "pop_density",
    "fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y",
    "days_since_last_fire", "days_since_harvest",
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
}

# Cột binary
BINARY_COLS = {"fire", "fire_lag_1", "fire_lag_3", "has_s2", "burn_season_flag"}


# ══════════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_column(s: pd.Series, col: str) -> dict:
    """Kiểm tra một cột, trả về dict với các flag bất thường."""
    issues = []
    n = len(s)

    # Skip date column
    if col == "date":
        return {"issues": [], "stats": f"min={s.min()}, max={s.max()}"}

    # Missing values
    n_nan = s.isna().sum()
    pct_nan = 100 * n_nan / n
    if pct_nan > 0:
        severity = "❌" if pct_nan > 5 else "⚠️" if pct_nan > 1 else "ℹ️"
        issues.append(f"{severity} NaN: {n_nan:,} ({pct_nan:.2f}%)")

    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {"issues": issues + ["❌ ALL NaN"], "stats": "N/A"}

    # Numeric checks
    if pd.api.types.is_numeric_dtype(s):
        col_min = float(s_clean.min())
        col_max = float(s_clean.max())
        col_mean = float(s_clean.mean())
        col_median = float(s_clean.median())
        col_std = float(s_clean.std())
        p01 = float(s_clean.quantile(0.01))
        p99 = float(s_clean.quantile(0.99))

        # Constant column check
        if col_std == 0:
            issues.append(f"❌ CONSTANT (all = {col_min})")

        # Non-negative check
        if col in NON_NEGATIVE_COLS and col_min < 0:
            n_neg = (s_clean < 0).sum()
            issues.append(f"❌ Negative values: {n_neg:,} (min={col_min:.4f})")

        # Binary check
        if col in BINARY_COLS:
            unique_vals = set(s_clean.unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                issues.append(f"❌ Non-binary values: {sorted(unique_vals)[:5]}")

        # Range check
        if col in EXPECTED_RANGES:
            exp_min, exp_max, unit, _ = EXPECTED_RANGES[col]

            # Check if min < expected min
            if col_min < exp_min:
                n_below = (s_clean < exp_min).sum()
                pct = 100 * n_below / len(s_clean)
                sev = "❌" if pct > 1 else "⚠️"
                issues.append(f"{sev} Below expected: {n_below:,} ({pct:.2f}%) "
                              f"min={col_min:.2f} < {exp_min} {unit}")

            # Check if max > expected max
            if col_max > exp_max:
                n_above = (s_clean > exp_max).sum()
                pct = 100 * n_above / len(s_clean)
                sev = "❌" if pct > 1 else "⚠️"
                issues.append(f"{sev} Above expected: {n_above:,} ({pct:.2f}%) "
                              f"max={col_max:.2f} > {exp_max} {unit}")

        # Suspicious distribution: high skew (mean >> median)
        if col_median > 0 and col_mean / col_median > 10:
            issues.append(f"⚠️ Highly skewed: mean={col_mean:.2f}, median={col_median:.2f}")

        stats = (f"min={col_min:.3f}  p01={p01:.3f}  median={col_median:.3f}  "
                 f"mean={col_mean:.3f}  p99={p99:.3f}  max={col_max:.3f}")
    else:
        # Categorical
        n_unique = s_clean.nunique()
        if n_unique == 1:
            issues.append(f"❌ CONSTANT (only value: {s_clean.iloc[0]})")
        stats = f"unique={n_unique}, top5={s_clean.value_counts().head(5).to_dict()}"

    return {"issues": issues, "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    log.info("Loaded: %d rows × %d cols", len(df), len(df.columns))
    log.info("Memory: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)

    print("\n" + "=" * 100)
    print(f"{'COLUMN':<28}{'STATUS':<10}{'STATS':<60}")
    print("=" * 100)

    summary = {"ok": 0, "info": 0, "warn": 0, "error": 0, "missing": 0}
    error_cols = []
    warn_cols = []

    columns_to_check = [
        'grid_id', 'date', 'lon', 'lat', 'tmean', 'rh', 'wind', 'rain', 'vpd',
        'fire', 'dem_mean', 'dem_stdev', 'dem_min', 'dem_max', 'slp_mean',
        'slp_stdev', 'ndvi', 'ndvi_std', 'ndwi', 'nbr', 'ndii', 'dist_road_km',
        'dist_settlement_km', 'dist_forest_edge_km', 'dist_powerline_km',
        'lulc_class', 'cropland_frac_1km', 'tree_cover_pct', 'rain_14d_sum',
        'vpd_14d_mean', 'rain_30d_sum', 'vpd_30d_mean', 'fire_lag_1', 'fire_lag_3',
        'neighbor_count', 'neighbor_fire_1d', 'neighbor_fire_3d', 'neighbor_fire_7d',
        'vpd_neighbor_1d', 'vpd_fire_lag_1', 'sin_doy', 'cos_doy', 'has_s2',
        'delta_ndvi_14d', 'delta_nbr_7d', 'deforestation_lag_1y', 'nightlight_mean',
        'pop_density', 'burn_season_flag', 'days_since_harvest',
        'fire_count_prev_year', 'fire_count_prev_3y', 'fire_freq_5y',
        'days_since_last_fire',
    ]

    for col in columns_to_check:
        if col not in df.columns:
            print(f"{col:<28}{'MISSING':<10}{'Column not in dataset':<60}")
            summary["missing"] += 1
            continue

        result = check_column(df[col], col)
        issues = result["issues"]
        stats = result["stats"]

        if not issues:
            status = "✅ OK"
            summary["ok"] += 1
        else:
            has_error = any("❌" in i for i in issues)
            has_warn = any("⚠️" in i for i in issues)
            if has_error:
                status = "❌ ERROR"
                summary["error"] += 1
                error_cols.append(col)
            elif has_warn:
                status = "⚠️ WARN"
                summary["warn"] += 1
                warn_cols.append(col)
            else:
                status = "ℹ️ INFO"
                summary["info"] += 1

        print(f"{col:<28}{status:<10}{stats[:60]:<60}")
        for issue in issues:
            print(f"{'':<38}{issue}")

    #  Summary 
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  ✅ OK     : {summary['ok']:>3}")
    print(f"  ℹ️ INFO   : {summary['info']:>3}")
    print(f"  ⚠️ WARN   : {summary['warn']:>3}")
    print(f"  ❌ ERROR  : {summary['error']:>3}")
    print(f"  ❓ MISSING: {summary['missing']:>3}")
    print(f"  TOTAL    : {len(columns_to_check):>3}")

    if error_cols:
        print(f"\n❌ Columns with ERRORS ({len(error_cols)}):")
        for c in error_cols:
            print(f"   - {c}")

    if warn_cols:
        print(f"\n⚠️ Columns with WARNINGS ({len(warn_cols)}):")
        for c in warn_cols:
            print(f"   - {c}")

    #  Cross-checks 
    print("\n" + "=" * 100)
    print("CROSS-CHECKS")
    print("=" * 100)

    # Date range
    if "date" in df.columns:
        print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"  Unique dates: {df['date'].nunique():,}")
        print(f"  Unique grids: {df['grid_id'].nunique():,}")

    # DEM consistency
    if all(c in df.columns for c in ["dem_min", "dem_mean", "dem_max"]):
        bad = ((df["dem_min"] > df["dem_mean"]) | (df["dem_mean"] > df["dem_max"])).sum()
        if bad > 0:
            print(f"  ❌ DEM inconsistency: {bad:,} rows where min > mean or mean > max")
        else:
            print(f"  ✅ DEM consistency: min ≤ mean ≤ max")

    # rain_30d_sum vs rain_14d_sum
    if all(c in df.columns for c in ["rain_14d_sum", "rain_30d_sum"]):
        bad = (df["rain_14d_sum"] > df["rain_30d_sum"]).sum()
        if bad > 0:
            print(f"  ❌ Rain rolling: {bad:,} rows where 14d_sum > 30d_sum")
        else:
            print(f"  ✅ Rain rolling: 14d_sum ≤ 30d_sum")

    # Fire rate
    if "fire" in df.columns:
        fire_rate = df["fire"].mean()
        print(f"  Fire rate: {fire_rate:.6f} ({df['fire'].sum():,} / {len(df):,})")
        if fire_rate > 0.05:
            print(f"  ⚠️ Unusually high fire rate (expected < 1%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
df = pd.read_parquet(
    "daklak_final_dataset_v3_pathways.parquet"
)
print(df.columns.tolist())
print(df.tail(3))
print("grid_id range:", df["grid_id"].min(), "→", df["grid_id"].max())
print("Số grid unique:", df["grid_id"].nunique())