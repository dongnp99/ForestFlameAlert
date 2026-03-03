import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.sparse import lil_matrix
import math
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from tqdm import trange
from scipy.sparse import csr_matrix

INPUT_PATH = "dataset_fire_final.csv"
OUTPUT_PATH = "daklak_fire_xgb_additional_features.parquet"
ADJ_PATH = "grid_adjacency.pkl"
CHUNK_SIZE = 1_000_000  # 1 triệu dòng

print("Loading data...")
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

# ===============================
# SORT (BẮT BUỘC)
# ===============================
df = df.sort_values(["grid_id", "date"])

# ===============================
# GROUP BY GRID
# ===============================
g = df.groupby("grid_id", group_keys=False)

# =========================================================
# 1️⃣ ROLLING FEATURES
# =========================================================
print("Creating rolling features...")

for window in [7, 14, 30]:
    df[f"rain_{window}d_sum"] = (
        g["rain"].transform(lambda x: x.rolling(window, 1).sum())
        .astype("float32")
    )

    df[f"vpd_{window}d_mean"] = (
        g["vpd"].transform(lambda x: x.rolling(window, 1).mean())
        .astype("float32")
    )

    df[f"wind_{window}d_mean"] = (
        g["wind"].transform(lambda x: x.rolling(window, 1).mean())
        .astype("float32")
    )

# =========================================================
# 2️⃣ DRYNESS INDEX
# =========================================================
print("Creating dryness index...")

df["dryness_14d"] = (
    df["vpd_14d_mean"] - 0.1 * df["rain_14d_sum"]
).astype("float32")

df["dryness_30d"] = (
    df["vpd_30d_mean"] - 0.05 * df["rain_30d_sum"]
).astype("float32")

# =========================================================
# 3️⃣ CONSECUTIVE DRY DAYS
# =========================================================
print("Creating consecutive dry days...")

def consecutive_dry(x):
    dry = (x < 1).astype(int)
    return dry * (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1)

df["consecutive_dry_days"] = (
    g["rain"].transform(consecutive_dry)
    .astype("int16")
)

# =========================================================
# 4️⃣ LAG FEATURES
# =========================================================
print("Creating lag features...")

for lag in [1, 3, 7]:
    df[f"vpd_lag_{lag}"] = (
        g["vpd"].shift(lag).fillna(0).astype("float32")
    )

    df[f"rain_lag_{lag}"] = (
        g["rain"].shift(lag).fillna(0).astype("float32")
    )

    df[f"fire_lag_{lag}"] = (
        g["fire"].shift(lag).fillna(0).astype("int8")
    )

# =========================================================
# 5️⃣ NEIGHBOR FIRE FEATURES
# =========================================================
print("Loading adjacency...")
with open(ADJ_PATH, "rb") as f:
    adjacency = pickle.load(f)

# Tạo dict đếm số neighbor
neighbor_count_dict = {
    grid_id: len(neighbors)
    for grid_id, neighbors in adjacency.items()
}

# Map vào dataframe
df["neighbor_count"] = df["grid_id"].map(neighbor_count_dict).astype("int8")

print("Building sparse adjacency matrix...")

grid_ids_sorted = sorted(adjacency.keys())
id_to_idx = {gid: i for i, gid in enumerate(grid_ids_sorted)}

n = len(grid_ids_sorted)
A = lil_matrix((n, n), dtype=np.float32)

for gid, neighbors in adjacency.items():
    i = id_to_idx[gid]
    for nb in neighbors:
        if nb in id_to_idx:
            j = id_to_idx[nb]
            A[i, j] = 1.0

A = A.tocsr()

print("Creating neighbor fire features (fast)...")

df["neighbor_fire_1d"] = 0.0
df["neighbor_fire_3d"] = 0.0
df["neighbor_fire_7d"] = 0.0

unique_dates = df["date"].unique()

for date in tqdm(unique_dates):

    mask = df["date"] == date
    day_df = df.loc[mask]

    fire1 = day_df.set_index("grid_id")["fire_lag_1"].reindex(grid_ids_sorted).fillna(0).values
    fire3 = day_df.set_index("grid_id")["fire_lag_3"].reindex(grid_ids_sorted).fillna(0).values
    fire7 = day_df.set_index("grid_id")["fire_lag_7"].reindex(grid_ids_sorted).fillna(0).values

    nf1 = A @ fire1
    nf3 = A @ fire3
    nf7 = A @ fire7

    df.loc[mask, "neighbor_fire_1d"] = nf1
    df.loc[mask, "neighbor_fire_3d"] = nf3
    df.loc[mask, "neighbor_fire_7d"] = nf7
# =========================================================
# 6️⃣ INTERACTION FEATURES
# =========================================================
print("Creating interaction features...")

df["wind_vpd"] = (df["wind"] * df["vpd"]).astype("float32")
df["slope_wind"] = (df["slp_mean"] * df["wind"]).astype("float32")
df["aspect_wind"] = (df["aspect_sin"] * df["wind"]).astype("float32")

# =========================================================
# 7️⃣ SEASONAL FEATURE
# =========================================================
print("Creating seasonal features...")

df["doy"] = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365).astype("float32")
df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365).astype("float32")
df = df.drop(columns=["doy"])

# =========================================================
# 8️⃣ OPTIMIZE DTYPES
# =========================================================
print("Optimizing dtypes...")

float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")

df["fire"] = df["fire"].astype("int8")

# =========================================================
# SAVE
# =========================================================
print("Saving new dataset...")
total_rows = len(df)
total_chunks = math.ceil(total_rows / CHUNK_SIZE)

print(f"Saving {total_rows} rows in {total_chunks} chunks...")

writer = None

for i in trange(total_chunks):
    start = i * CHUNK_SIZE
    end = min((i + 1) * CHUNK_SIZE, total_rows)

    chunk = df.iloc[start:end]

    table = pa.Table.from_pandas(
        chunk,
        preserve_index=False
    )

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_PATH,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    del chunk, table
    gc.collect()

writer.close()

print("Parquet file saved successfully.")