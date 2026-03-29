import pandas as pd
import numpy as np
import pickle
import math
import gc
from tqdm import tqdm, trange
from scipy.sparse import lil_matrix
import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_PATH = "../daklak_final_dataset.parquet"
ADJ_PATH = "../xgb_veg_indices/grid_adjacency.pkl"
CHUNK_SIZE = 1_000_000

# ===== 1. LOAD DEM =====
print("Loading DEM...")
dem = pd.read_csv("../raw_data/daklak_dem.csv")[
    ["grid_id", "dem_mean", "dem_stdev", "dem_min", "dem_max",
     "slp_mean", "slp_stdev"]
]

# ===== 2. MERGE ERA5 + FIRMS + DEM =====
print("Merging ERA5+FIRMS with DEM...")
chunks = []
drop_cols = ["number", "mask"]

for chunk in pd.read_csv("../daklak_era5_firms.csv", chunksize=500_000, parse_dates=["date"]):
    chunk = chunk.merge(dem, on="grid_id", how="left")
    chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])
    chunks.append(chunk)
    print(f"  chunk {len(chunks)} done")

print("Concatenating chunks...")
df = pd.concat(chunks, ignore_index=True)
del chunks, dem
gc.collect()

# Apply dtypes
df["fire"] = df["fire"].astype("int8")
df["grid_id"] = df["grid_id"].astype("int32")

# ===== 3. FEATURE ENGINEERING =====
print("Sorting...")
df = df.sort_values(["grid_id", "date"])
g = df.groupby("grid_id", group_keys=False)

# Rolling features
print("Creating rolling features...")
for window in [14, 30]:
    df[f"rain_{window}d_sum"] = (
        g["rain"].transform(lambda x: x.rolling(window, 1).sum()).astype("float32")
    )
    df[f"vpd_{window}d_mean"] = (
        g["vpd"].transform(lambda x: x.rolling(window, 1).mean()).astype("float32")
    )

# Lag features
print("Creating lag features...")
for lag in [1, 3, 7]:
    df[f"fire_lag_{lag}"] = (
        g["fire"].shift(lag).fillna(0).astype("int8")
    )

# Neighbor fire features
print("Loading adjacency...")
with open(ADJ_PATH, "rb") as f:
    adjacency = pickle.load(f)

df["neighbor_count"] = df["grid_id"].map(
    {gid: len(nb) for gid, nb in adjacency.items()}
).astype("int8")

print("Building sparse adjacency matrix...")
grid_ids_sorted = sorted(adjacency.keys())
id_to_idx = {gid: i for i, gid in enumerate(grid_ids_sorted)}
n = len(grid_ids_sorted)
A = lil_matrix((n, n), dtype=np.float32)
for gid, neighbors in adjacency.items():
    i = id_to_idx[gid]
    for nb in neighbors:
        if nb in id_to_idx:
            A[i, id_to_idx[nb]] = 1.0
A = A.tocsr()

print("Creating neighbor fire features...")
df["neighbor_fire_1d"] = 0.0
df["neighbor_fire_3d"] = 0.0
df["neighbor_fire_7d"] = 0.0

for date in tqdm(df["date"].unique()):
    mask = df["date"] == date
    day_df = df.loc[mask]
    fire1 = day_df.set_index("grid_id")["fire_lag_1"].reindex(grid_ids_sorted).fillna(0).values
    fire3 = day_df.set_index("grid_id")["fire_lag_3"].reindex(grid_ids_sorted).fillna(0).values
    fire7 = day_df.set_index("grid_id")["fire_lag_7"].reindex(grid_ids_sorted).fillna(0).values
    df.loc[mask, "neighbor_fire_1d"] = A @ fire1
    df.loc[mask, "neighbor_fire_3d"] = A @ fire3
    df.loc[mask, "neighbor_fire_7d"] = A @ fire7

df["neighbor_fire_1d"] = df["neighbor_fire_1d"].astype("float32")
df["neighbor_fire_3d"] = df["neighbor_fire_3d"].astype("float32")
df["neighbor_fire_7d"] = df["neighbor_fire_7d"].astype("float32")

# Interaction features
print("Creating interaction features...")
df["vpd_neighbor_1d"] = (df["vpd"] * df["neighbor_fire_1d"]).astype("float32")
df["vpd_fire_lag_1"] = (df["vpd"] * df["fire_lag_1"]).astype("float32")

# Seasonal features
print("Creating seasonal features...")
doy = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * doy / 365).astype("float32")
df["cos_doy"] = np.cos(2 * np.pi * doy / 365).astype("float32")

# Optimize dtypes
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")

# ===== 4. SAVE =====
print("Saving final dataset...")
total_rows = len(df)
total_chunks = math.ceil(total_rows / CHUNK_SIZE)
print(f"Saving {total_rows} rows in {total_chunks} chunks...")

writer = None
for i in trange(total_chunks):
    chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="snappy")
    writer.write_table(table)
    del chunk, table
    gc.collect()

writer.close()
print("Done. Saved to", OUTPUT_PATH)