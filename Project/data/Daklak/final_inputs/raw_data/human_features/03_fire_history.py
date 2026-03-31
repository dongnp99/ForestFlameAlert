"""
03_fire_history.py - Fire History Features (Phase 3)

Computes 5 features directly from the existing dataset. No external data needed.

    fire_count_prev_year  (int16)   - fire=1 count in the previous year for this grid
    fire_count_prev_3y    (int16)   - fire=1 count over the 3 preceding years
    fire_freq_5y          (float32) - mean annual fire count over the 5 preceding years
    days_since_last_fire  (int16)   - days since the last fire at this grid (-1 if never)
    burn_season_flag      (int8)    - 1 if month in [1,2,3,4] (Dak Lak burn season)

IMPORTANT: All features are computed on the FULL dataset before any train/val/test
split to avoid data leakage.

Output:
    features/fire_history.parquet   - grid_id, date, + 5 fire history features

Usage:
    python 03_fire_history.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -- Paths ---------------------------------------------------------------------
HERE         = Path(__file__).parent
FEATURES_DIR = HERE / "features"
FEATURES_DIR.mkdir(exist_ok=True)

DATASET_PATH = HERE.parent.parent / "daklak_final_dataset.parquet"
OUT_PARQUET  = FEATURES_DIR / "fire_history.parquet"


def days_since_harvest(date: pd.Timestamp) -> int:
    """
    Days since end of coffee harvest (Dec 31).
    Jan 1 -> 1, Feb 1 -> 32, Dec 31 -> 365.
    """
    harvest_end_doy = 365
    doy = date.dayofyear
    if doy >= harvest_end_doy:
        return int(doy - harvest_end_doy)
    else:
        return int(365 - harvest_end_doy + doy)


def compute_days_since_last_fire(group: pd.DataFrame) -> pd.Series:
    """Row-by-row: days since most recent prior fire=-1 if none yet."""
    last_fire = pd.NaT
    result = []
    for row in group.itertuples():
        if last_fire is pd.NaT:
            result.append(-1)
        else:
            result.append((row.date - last_fire).days)
        if row.fire == 1:
            last_fire = row.date
    return pd.Series(result, index=group.index, dtype="int16")


def main() -> None:
    # ── 1. Load only needed columns ───────────────────────────────────────────
    print("[1/6] Loading dataset ...")
    df = pd.read_parquet(
        DATASET_PATH,
        columns=["grid_id", "date", "fire"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    print(f"  Rows: {len(df):,}  |  grids: {df['grid_id'].nunique():,}  |  "
          f"years: {df['year'].min()}-{df['year'].max()}")

    # ── 2. burn_season_flag ───────────────────────────────────────────────────
    print("[2/6] burn_season_flag ...")
    df["burn_season_flag"] = df["date"].dt.month.isin([1, 2, 3, 4]).astype("int8")

    # ── 3. days_since_harvest ─────────────────────────────────────────────────
    print("[3/6] days_since_harvest ...")
    df["days_since_harvest"] = df["date"].apply(days_since_harvest).astype("int16")

    # ── 4. Annual fire counts per grid ────────────────────────────────────────
    print("[4/6] Annual fire count features ...")

    annual = (
        df.groupby(["grid_id", "year"])["fire"]
        .sum()
        .rename("fire_count")
        .reset_index()
    )

    # fire_count_prev_year: shift by 1 year
    annual_shifted = annual.copy()
    annual_shifted["year"] = annual_shifted["year"] + 1
    annual_shifted = annual_shifted.rename(columns={"fire_count": "fire_count_prev_year"})
    df = df.merge(annual_shifted[["grid_id", "year", "fire_count_prev_year"]],
                  on=["grid_id", "year"], how="left")
    df["fire_count_prev_year"] = df["fire_count_prev_year"].fillna(0).astype("int16")

    # fire_count_prev_3y: rolling 3-year sum, shifted to exclude current year
    annual_sorted = annual.sort_values(["grid_id", "year"])
    annual_sorted["fire_count_prev_3y"] = (
        annual_sorted.groupby("grid_id")["fire_count"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
        .astype("int16")
    )
    df = df.merge(annual_sorted[["grid_id", "year", "fire_count_prev_3y"]],
                  on=["grid_id", "year"], how="left")
    df["fire_count_prev_3y"] = df["fire_count_prev_3y"].fillna(0).astype("int16")

    # fire_freq_5y: rolling 5-year mean, shifted
    annual_sorted["fire_freq_5y"] = (
        annual_sorted.groupby("grid_id")["fire_count"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0)
        .astype("float32")
    )
    df = df.merge(annual_sorted[["grid_id", "year", "fire_freq_5y"]],
                  on=["grid_id", "year"], how="left")
    df["fire_freq_5y"] = df["fire_freq_5y"].fillna(0).astype("float32")

    # ── 5. days_since_last_fire ───────────────────────────────────────────────
    print("[5/6] days_since_last_fire (row-by-row per grid, may take a minute) ...")
    df["days_since_last_fire"] = (
        df.groupby("grid_id", group_keys=False)
        .apply(compute_days_since_last_fire, include_groups=False)
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print("[6/6] Saving ...")
    FEATURE_COLS = [
        "grid_id", "date",
        "fire_count_prev_year",
        "fire_count_prev_3y",
        "fire_freq_5y",
        "days_since_last_fire",
        "burn_season_flag",
        "days_since_harvest",
    ]
    out = df[FEATURE_COLS].copy()
    out.to_parquet(OUT_PARQUET, index=False)

    print(f"\nSaved -> {OUT_PARQUET}")
    print(f"Shape: {out.shape}")
    print("\nNull counts:")
    print(out.isnull().sum().to_string())
    print("\nStats:")
    print(out.drop(columns=["grid_id", "date"]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
