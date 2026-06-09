"""
build_dataset.py — Full data pipeline

Merges: ERA5 + FIRMS fire label + DEM + Veg Indices
Feature engineering: rolling, lag, neighbor fire, seasonal, veg ffill + delta
Output: daklak_final_dataset.parquet

Replaces:
  merge_era5_fire_label.py
  merge_era5_veg_indices.py
  build_xgb_dataset.py
  merge_veg_indices.py  (veg ffill + delta features baked into parquet)
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
BASE     = Path(__file__).resolve().parent / ".."
RAW      = BASE / "raw_data"
VEG_DIR  = BASE / "xgb_veg_indices"
ADJ_PATH = VEG_DIR / "grid_adjacency.pkl"
OUT_PATH = BASE / "daklak_final_dataset.parquet"

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
print(f"   {len(dem):,} grid cells")

#  3. Veg indices ─
print("3. Loading veg indices (2015–2024)...")
veg_dfs = []
for year in range(2015, 2025):
    for prefix in ["veg_v2_", "veg_"]:       # prefer Cloud Score+ v2
        f = VEG_DIR / f"{prefix}{year}.csv"
        if f.exists():
            df = pd.read_csv(f, usecols=VEG_USECOLS, parse_dates=["date"])
            df["grid_id"] = df["grid_id"].astype("int32")
            for col in VEG_USECOLS[2:]:
                df[col] = df[col].astype("float32")
            veg_dfs.append(df)
            print(f"   {f.name}: {len(df):,} rows")
            break
    else:
        print(f"   WARNING: no veg file for {year}")

veg = pd.concat(veg_dfs, ignore_index=True)
veg = veg.rename(columns=VEG_RENAME)
veg = veg.dropna(subset=["ndvi", "nbr"])
veg = veg.drop_duplicates(subset=["date", "grid_id"])
veg = veg.set_index(["date", "grid_id"])
del veg_dfs
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

#  5. Chunked read ERA5 → fire + DEM + veg ─
print("\n5. Reading ERA5 and merging fire / DEM / veg...")
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

    # Cast float64 → float32 before accumulating to halve concat memory
    for col in chunk.columns:
        if chunk[col].dtype == "float64":
            chunk[col] = chunk[col].astype("float32")
    chunk = chunk.drop(columns=[c for c in ("number", "mask") if c in chunk.columns])

    chunks.append(chunk)
    print(f"   chunk {i + 1:>3d}: {len(chunk):,} rows")

del fire_key, dem, veg
gc.collect()

#  6. Concat & sort ─
print("\n6. Concatenating and sorting...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

df["fire"]    = df["fire"].astype("int8")
df["grid_id"] = df["grid_id"].astype("int32")
df = df.sort_values(["grid_id", "date"]).reset_index(drop=True)
print(f"   {len(df):,} rows total")

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

#  10. Interaction & seasonal features ─
print("10. Interaction & seasonal features...")
df["vpd_neighbor_1d"] = (df["vpd"] * df["neighbor_fire_1d"]).astype("float32")
df["vpd_fire_lag_1"]  = (df["vpd"] * df["fire_lag_1"]).astype("float32")

doy           = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * doy / 365).astype("float32")
df["cos_doy"] = np.cos(2 * np.pi * doy / 365).astype("float32")

#  11. Veg: has_s2 + seasonal ffill + delta features ─
print("11. Veg forward-fill & delta features...")
df["has_s2"] = df["ndvi"].notna().astype("int8")

# Dry season (Oct–Apr): ffill up to 20 days; wet (May–Sep): up to 6 days
dry_mask     = ~df["date"].dt.month.isin([5, 6, 7, 8, 9])
filled_long  = df.groupby("grid_id")[VEG_FILL_COLS].ffill(limit=20)
filled_short = df.groupby("grid_id")[VEG_FILL_COLS].ffill(limit=6)
filled_long[~dry_mask] = filled_short[~dry_mask]
df[VEG_FILL_COLS] = filled_long
del filled_long, filled_short
gc.collect()

df["delta_ndvi_14d"] = (
    df.groupby("grid_id")["ndvi"].transform(lambda s: s.shift(14) - s)
).astype("float32")
df["delta_nbr_7d"] = (
    df.groupby("grid_id")["nbr"].transform(lambda s: s.shift(7) - s)
).astype("float32")

for col in VEG_FILL_COLS + ["delta_ndvi_14d", "delta_nbr_7d"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median()).astype("float32")

print(f"   NDVI coverage: {(df['has_s2'] == 1).mean():.1%}")

#  12. Final dtype pass ─
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")

#  13. Save parquet ─
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