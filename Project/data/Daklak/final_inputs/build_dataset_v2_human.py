"""
build_dataset_v2_human.py — Full data pipeline + Human Activity Features
                             (memory-optimised version)

Memory strategy vs naive version:
  - Static human features applied via dict.map() per chunk (no 68M-row merge)
  - Year-varying features applied via MultiIndex.reindex() (no merge copy)
  - Fire history features recomputed inline (avoids loading 68M-row parquet)
  - Intermediates deleted and gc.collect() called after every major step

Output: daklak_final_dataset_v2_human.parquet  (47 columns)
"""

import gc
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import lil_matrix
from tqdm import tqdm, trange

#  Paths 
BASE      = Path(__file__).resolve().parent
RAW       = BASE / "raw_data"
VEG_DIR   = RAW / "veg_indices"
ADJ_PATH  = RAW / "grid_adjacency.pkl"
HUMAN_DIR = RAW / "human_features" / "features"
OUT_PATH  = BASE / "daklak_final_dataset_v2_human.parquet"

CHUNKSIZE  = 500_000
SAVE_CHUNK = 1_000_000

VEG_USECOLS = ["date", "grid_id", "NDVI_mean", "NDVI_stdDev", "NDWI_mean", "NBR_mean", "NDII_mean"]
VEG_RENAME  = {
    "NDVI_mean":   "ndvi",
    "NDVI_stdDev": "ndvi_std",
    "NDWI_mean":   "ndwi",
    "NBR_mean":    "nbr",
    "NDII_mean":   "ndii",
}
VEG_FILL_COLS = ["ndvi", "ndvi_std", "ndwi", "nbr", "ndii"]

#  1. Fire label lookup ─
print("1. Loading FIRMS fire labels...")
fire_df  = pd.read_csv(RAW / "daklak_firms.csv", parse_dates=["date"])
fire_key = set(zip(fire_df.grid_id, fire_df.date))
del fire_df
gc.collect()
print(f"   {len(fire_key):,} fire events")

#  2. DEM ─
print("2. Loading DEM...")
dem = pd.read_csv(RAW / "daklak_dem.csv")[
    ["grid_id", "dem_mean", "dem_stdev", "dem_min", "dem_max", "slp_mean", "slp_stdev"]
]
dem["grid_id"] = dem["grid_id"].astype("int32")
# Resolve dem_mean, dem_min negative

# Fix sentinel values (-32767, -6423...) trong dem_mean và dem_min
n_bad_mean = (dem["dem_mean"] < 0).sum()
n_bad_min  = (dem["dem_min"]  < 0).sum()

# Set giá trị âm về NaN
dem.loc[dem["dem_mean"] < 0, "dem_mean"] = np.nan
dem.loc[dem["dem_min"]  < 0, "dem_min"]  = np.nan

mask_mean_nan = dem["dem_mean"].isna() & dem["dem_max"].notna() & (dem["dem_max"] >= 0)
dem.loc[mask_mean_nan, "dem_mean"] = dem.loc[mask_mean_nan, "dem_max"]

mask_min_nan = dem["dem_min"].isna() & dem["dem_max"].notna() & (dem["dem_max"] >= 0)
dem.loc[mask_min_nan, "dem_min"] = dem.loc[mask_min_nan, "dem_max"]
if dem["dem_mean"].isna().any():
    dem["dem_mean"] = dem[("dem_"
                           "mean")].fillna(dem["dem_mean"].median())
if dem["dem_min"].isna().any():
    dem["dem_min"] = dem["dem_min"].fillna(dem["dem_min"].median())
# Đảm bảo dem_min ≤ dem_mean ≤ dem_max (consistency)
dem["dem_min"]  = np.minimum(dem["dem_min"],  dem["dem_mean"])
dem["dem_mean"] = np.minimum(dem["dem_mean"], dem["dem_max"])
print(f"   {len(dem):,} grid cells")

#  3. Veg indices ─
print("3. Loading veg indices (2015–2024)...")
veg_dfs = []
for year in range(2015, 2025):
    for prefix in ["veg_v2_", "veg_"]:
        f = VEG_DIR / f"{prefix}{year}.csv"
        if f.exists():
            df_v = pd.read_csv(f, usecols=VEG_USECOLS, parse_dates=["date"])
            df_v["grid_id"] = df_v["grid_id"].astype("int32")
            for col in VEG_USECOLS[2:]:
                df_v[col] = df_v[col].astype("float32")
            veg_dfs.append(df_v)
            print(f"   {f.name}: {len(df_v):,} rows")
            break
    else:
        print(f"   WARNING: no veg file for {year}")

veg = pd.concat(veg_dfs, ignore_index=True)
veg = veg.rename(columns=VEG_RENAME)
veg = veg.dropna(subset=["ndvi", "nbr"])
veg = veg.drop_duplicates(subset=["date", "grid_id"])
veg = veg.set_index(["date", "grid_id"])
del veg_dfs, df_v
gc.collect()
print(f"   Total: {len(veg):,} veg rows")

#  4. Adjacency matrix 
print("4. Building adjacency matrix...")
with open(ADJ_PATH, "rb") as f:
    adjacency = pickle.load(f)

grid_ids_sorted = sorted(adjacency.keys())
id_to_idx       = {gid: i for i, gid in enumerate(grid_ids_sorted)}
n               = len(grid_ids_sorted)

A = lil_matrix((n, n), dtype=np.float32)
for gid, neighbors in adjacency.items():
    i = id_to_idx[gid]
    for nb in neighbors:
        if nb in id_to_idx:
            A[i, id_to_idx[nb]] = 1.0
A = A.tocsr()
print(f"   {n} grid cells in adjacency")

#  4b. Pre-load human static features as dicts (grid_id → value) ─
# dict.map() on a Series allocates only the output column, no merge copy.
print("4b. Pre-loading human static feature dicts...")

osm  = pd.read_parquet(HUMAN_DIR / "osm_distances.parquet")
lulc = pd.read_parquet(HUMAN_DIR / "lulc_hansen.parquet")

def _col_dict(df_src, col, dtype):
    return df_src.set_index("grid_id")[col].astype(dtype).to_dict()

STATIC_MAPS = {
    "dist_road_km":        _col_dict(osm,  "dist_road_km",        "float32"),
    "dist_settlement_km":  _col_dict(osm,  "dist_settlement_km",  "float32"),
    "dist_forest_edge_km": _col_dict(osm,  "dist_forest_edge_km", "float32"),
    "dist_powerline_km":   _col_dict(osm,  "dist_powerline_km",   "float32"),
    "lulc_class":          _col_dict(lulc, "lulc_class",          "int8"),
    "cropland_frac_1km":   _col_dict(lulc, "cropland_frac_1km",   "float32"),
    "tree_cover_pct":      _col_dict(lulc, "tree_cover_pct",      "float32"),
}
del osm, lulc
gc.collect()
print(f"   {len(STATIC_MAPS)} static feature dicts ready")

#  4c. Pre-load year-varying features as MultiIndex Series ─
# reindex() on a MultiIndex avoids creating a merge copy of the 68M df.
print("4c. Pre-loading year-varying feature Series...")

defor = pd.read_parquet(HUMAN_DIR / "deforestation_by_year.parquet")
ntl   = pd.read_parquet(HUMAN_DIR / "nightlight_pop_by_year.parquet")

defor_series = (defor.set_index(["grid_id", "year"])["deforestation_lag_1y"]
                .astype("float32"))
ntl_mean_series = (ntl.set_index(["grid_id", "year"])["nightlight_mean"]
                   .astype("float32"))
ntl_pop_series  = (ntl.set_index(["grid_id", "year"])["pop_density"]
                   .astype("float32"))
del defor, ntl
gc.collect()
print("   deforestation + nightlight + pop series ready")

#  5. Chunked read ERA5 → fire + DEM + veg + static human 
print("\n5. Reading ERA5 and merging fire / DEM / veg + static human features...")
chunks = []

for i, chunk in enumerate(pd.read_csv(
        RAW / "daklak_era5.csv", chunksize=CHUNKSIZE, parse_dates=["date"])):

    chunk["grid_id"] = chunk["grid_id"].astype("int32")

    chunk["fire"] = [
        1 if (gid, d) in fire_key else 0
        for gid, d in zip(chunk.grid_id, chunk.date)
    ]

    chunk = chunk.merge(dem, on="grid_id", how="left")
    chunk = chunk.join(veg, on=["date", "grid_id"], how="left")

    # Apply static human features per-chunk via dict.map (no copy overhead)
    for col, mapping in STATIC_MAPS.items():
        chunk[col] = chunk["grid_id"].map(mapping)

    for col in chunk.columns:
        if chunk[col].dtype == "float64":
            chunk[col] = chunk[col].astype("float32")
    for c in ("number", "mask"):
        if c in chunk.columns:
            del chunk[c]

    chunks.append(chunk)
    print(f"   chunk {i + 1:>3d}: {len(chunk):,} rows")

del fire_key, dem, veg, STATIC_MAPS
gc.collect()

#  6. Concat & sort ─
print("\n6. Concatenating and sorting...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

df["fire"]    = df["fire"].astype("int8")
df["grid_id"] = df["grid_id"].astype("int32")

df.sort_values(["grid_id", "date"], inplace=True)
df.index = range(len(df))
gc.collect()
print(f"   {len(df):,} rows total  |  {len(df.columns)} columns")

g = df.groupby("grid_id", group_keys=False)

#  7. Rolling features 
print("7. Rolling features...")
for window in [14, 30]:
    df[f"rain_{window}d_sum"] = (
        g["rain"].transform(lambda x: x.rolling(window, min_periods=1).sum())
        .astype("float32")
    )
    df[f"vpd_{window}d_mean"] = (
        g["vpd"].transform(lambda x: x.rolling(window, min_periods=1).mean())
        .astype("float32")
    )

#  8. Lag features 
print("8. Fire lag features...")
for lag in [1, 3, 7]:
    df[f"fire_lag_{lag}"] = g["fire"].shift(lag).fillna(0).astype("int8")

#  9. Neighbor fire features 
print("9. Neighbor fire features (slow step)...")
df["neighbor_count"] = (
    df["grid_id"].map({gid: len(nb) for gid, nb in adjacency.items()})
    .astype("int8")
)
df["neighbor_fire_1d"] = np.float32(0)
df["neighbor_fire_3d"] = np.float32(0)
df["neighbor_fire_7d"] = np.float32(0)

for date in tqdm(df["date"].unique(), desc="   dates"):
    mask   = df["date"] == date
    day_df = df.loc[mask]

    fire1 = day_df.set_index("grid_id")["fire_lag_1"].reindex(grid_ids_sorted).fillna(0).values
    fire3 = day_df.set_index("grid_id")["fire_lag_3"].reindex(grid_ids_sorted).fillna(0).values
    fire7 = day_df.set_index("grid_id")["fire_lag_7"].reindex(grid_ids_sorted).fillna(0).values

    nb1 = pd.Series(A @ fire1, index=grid_ids_sorted)
    nb3 = pd.Series(A @ fire3, index=grid_ids_sorted)
    nb7 = pd.Series(A @ fire7, index=grid_ids_sorted)

    df.loc[mask, "neighbor_fire_1d"] = day_df["grid_id"].map(nb1).values
    df.loc[mask, "neighbor_fire_3d"] = day_df["grid_id"].map(nb3).values
    df.loc[mask, "neighbor_fire_7d"] = day_df["grid_id"].map(nb7).values

df["neighbor_fire_1d"] = df["neighbor_fire_1d"].astype("float32")
df["neighbor_fire_3d"] = df["neighbor_fire_3d"].astype("float32")
df["neighbor_fire_7d"] = df["neighbor_fire_7d"].astype("float32")

# fire_lag_7 no longer needed after neighbor computation
del df["fire_lag_7"]
gc.collect()

#  10. Interaction & seasonal features ─
print("10. Interaction & seasonal features...")
df["vpd_neighbor_1d"] = (df["vpd"] * df["neighbor_fire_1d"]).astype("float32")
df["vpd_fire_lag_1"]  = (df["vpd"] * df["fire_lag_1"]).astype("float32")

doy           = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * doy / 365).astype("float32")
df["cos_doy"] = np.cos(2 * np.pi * doy / 365).astype("float32")
del doy
gc.collect()

#  11. Veg forward-fill & delta features ─
print("11. Veg forward-fill & delta features...")
df["has_s2"] = df["ndvi"].notna().astype("int8")

dry_mask = ~df["date"].dt.month.isin([5, 6, 7, 8, 9])
for col in VEG_FILL_COLS:
    filled_long  = df.groupby("grid_id")[col].ffill(limit=20)
    filled_short = df.groupby("grid_id")[col].ffill(limit=6)
    filled_long[~dry_mask] = filled_short[~dry_mask]
    df[col] = filled_long
    del filled_long, filled_short
del dry_mask
gc.collect()

# Tính delta indices
df["delta_ndvi_14d"] = (
    df.groupby("grid_id")["ndvi"].transform(lambda s: s.shift(14) - s)
).astype("float32")
df["delta_nbr_7d"] = (
    df.groupby("grid_id")["nbr"].transform(lambda s: s.shift(7) - s)
).astype("float32")

# Clip delta về [-1, 1] (loại bỏ outlier do làm tròn float)
df["delta_ndvi_14d"] = df["delta_ndvi_14d"].clip(-1, 1)
df["delta_nbr_7d"]   = df["delta_nbr_7d"].clip(-1, 1)

# Fill NaN
for col in VEG_FILL_COLS + ["delta_ndvi_14d", "delta_nbr_7d"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median()).astype("float32")

print(f"   NDVI coverage: {(df['has_s2'] == 1).mean():.1%}")
#  12. Final dtype pass ─
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")

# ══════════════════════════════════════════════════════════════════════════════
#  12b. Human Activity Features — year-varying + fire history ─
# ══════════════════════════════════════════════════════════════════════════════
print("\n12b. Human activity features — year-varying + fire history...")

#  Year-varying via MultiIndex reindex (no merge copy) ─
print("   [a] deforestation_lag_1y / nightlight_mean / pop_density ...")
_year = df["date"].dt.year.astype("int32")
_idx  = pd.MultiIndex.from_arrays([df["grid_id"].values, _year.values],
                                   names=["grid_id", "year"])

df["deforestation_lag_1y"] = defor_series.reindex(_idx).fillna(0).values.astype("float32")
df["nightlight_mean"]      = ntl_mean_series.reindex(_idx).fillna(0).values.astype("float32")
df["nightlight_mean"]      = df["nightlight_mean"].clip(lower=0)
df["pop_density"]          = ntl_pop_series.reindex(_idx).fillna(0).values.astype("float32")

del _year, _idx, defor_series, ntl_mean_series, ntl_pop_series
gc.collect()

#  Fire history features recomputed from fire column ─
# Avoids loading the 68M-row fire_history.parquet entirely.
print("   [b] burn_season_flag / days_since_harvest (date arithmetic) ...")

df["burn_season_flag"] = df["date"].dt.month.isin([1, 2, 3, 4]).astype("int8")

harvest_end_doy = 365
_doy = df["date"].dt.dayofyear
df["days_since_harvest"] = np.where(
    _doy >= harvest_end_doy,
    _doy - harvest_end_doy,
    365 - harvest_end_doy + _doy,
).astype("int16")
del _doy
gc.collect()

print("   [c] fire_count_prev_year / fire_count_prev_3y / fire_freq_5y ...")
_year_col = df["date"].dt.year.rename("year")

# Annual fire count per grid (already sorted by grid_id, date)
annual = (
    df.groupby(["grid_id", _year_col])["fire"]
    .sum()
    .rename("fire_count")
    .reset_index()
    .sort_values(["grid_id", "year"])
)

# Shift and rolling on the annual series grouped by grid
annual["fire_count_prev_year"] = (
    annual.groupby("grid_id")["fire_count"]
    .transform(lambda x: x.shift(1))
    .fillna(0).astype("int16")
)
annual["fire_count_prev_3y"] = (
    annual.groupby("grid_id")["fire_count"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    .fillna(0).astype("int16")
)
annual["fire_freq_5y"] = (
    annual.groupby("grid_id")["fire_count"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    .fillna(0).astype("float32")
)

# Build MultiIndex lookup Series for year-keyed join (no merge copy)
_idx2 = pd.MultiIndex.from_arrays([annual["grid_id"].values, annual["year"].values])
_lookup_idx = pd.MultiIndex.from_arrays(
    [df["grid_id"].values, df["date"].dt.year.values]
)
for col in ["fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y"]:
    s = pd.Series(annual[col].values, index=_idx2)
    df[col] = s.reindex(_lookup_idx).fillna(0).values

df["fire_count_prev_year"] = df["fire_count_prev_year"].astype("int16")
df["fire_count_prev_3y"]   = df["fire_count_prev_3y"].astype("int16")
df["fire_freq_5y"]         = df["fire_freq_5y"].astype("float32")
del annual, _year_col, _idx2, _lookup_idx
gc.collect()

print("   [d] days_since_last_fire (vectorized shift+ffill) ...")
# df is already sorted by (grid_id, date).
# For each row: days since the previous fire in the same grid (-1 if none yet).
# Vectorized: mark fire dates, shift by 1 within grid (so the current row's
# own fire doesn't count), then forward-fill — all built-in C code, no full-df copy.
_fire_dates = df["date"].where(df["fire"] == 1)
_shifted    = _fire_dates.groupby(df["grid_id"], sort=False).shift(1)
_last_fire  = _shifted.groupby(df["grid_id"], sort=False).ffill()
df["days_since_last_fire"] = (
    (df["date"] - _last_fire).dt.days.fillna(-1).astype("int16")
)
df.loc[df["days_since_last_fire"] == -1, "days_since_last_fire"] = 9999
del _fire_dates, _shifted, _last_fire

gc.collect()

#  Null check for all human columns 
HUMAN_COLS = [
    "dist_road_km", "dist_settlement_km", "dist_forest_edge_km", "dist_powerline_km",
    "lulc_class", "cropland_frac_1km", "tree_cover_pct",
    "deforestation_lag_1y", "nightlight_mean", "pop_density",
    "fire_count_prev_year", "fire_count_prev_3y", "fire_freq_5y",
    "days_since_last_fire", "burn_season_flag", "days_since_harvest",
]
null_counts = pd.Series({col: int(df[col].isnull().sum()) for col in HUMAN_COLS})
if null_counts.any():
    print(f"   WARNING nulls — filling 0:\n{null_counts[null_counts > 0]}")
    for col in HUMAN_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)

print(f"   Total nulls in human cols: {null_counts.sum()}")
print(f"   Total columns: {len(df.columns)}")

#  13. Incremental parquet write ─
print(f"\n13. Saving to {OUT_PATH} ...")
total_rows   = len(df)
total_chunks = math.ceil(total_rows / SAVE_CHUNK)
writer       = None

for i in trange(total_chunks, desc="    writing"):
    chunk = df.iloc[i * SAVE_CHUNK : (i + 1) * SAVE_CHUNK]
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT_PATH, table.schema, compression="snappy")
    writer.write_table(table)
    del chunk, table
    gc.collect()

writer.close()
print(f"\nDone!  {total_rows:,} rows")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
