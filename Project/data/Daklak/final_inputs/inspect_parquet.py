import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import numpy as np

PARQUET_PATH = "clstm_data/clstm_clean_data.parquet"

print("========================================")
print("Opening parquet file...")
print("========================================")

# ======================================================
# 1️⃣ READ METADATA (NO FULL LOAD)
# ======================================================

pq_file = pq.ParquetFile(PARQUET_PATH)

print("\nNumber of row groups:", pq_file.num_row_groups)
print("Number of rows:", pq_file.metadata.num_rows)
print("Number of columns:", pq_file.metadata.num_columns)

print("\nSchema:")
print(pq_file.schema)

# ======================================================
# 2️⃣ LOAD SMALL SAMPLE
# ======================================================

print("\nLoading first 5 rows...")

sample = pq_file.read_row_group(0).to_pandas().head(5)
print(sample)

# ======================================================
# 3️⃣ CHECK DTYPES (sample-based)
# ======================================================

print("\nColumn dtypes (sample):")
print(sample.dtypes)

# ======================================================
# 4️⃣ QUICK FIRE RATE CHECK
# ======================================================

dataset = ds.dataset(PARQUET_PATH)

fire_col = dataset.to_table(columns=["fire"]).column("fire").to_numpy()

fire_rate = fire_col.mean()

print("\nFire rate:", fire_rate)
print("Positive samples:", fire_col.sum())
print("Total samples:", len(fire_col))

# ======================================================
# 5️⃣ NULL CHECK (SAFE METHOD)
# ======================================================

print("\nChecking null counts per column...")

null_counts = {}

for col in pq_file.schema.names:
    arr = dataset.to_table(columns=[col]).column(col)
    null_counts[col] = arr.null_count

null_df = pd.DataFrame.from_dict(null_counts, orient="index", columns=["null_count"])
print(null_df.sort_values("null_count", ascending=False).head(20))

# ======================================================
# 6️⃣ MEMORY ESTIMATE
# ======================================================

print("\nEstimating memory usage (approx)...")

total_bytes = 0

for col in pq_file.schema.names:
    arr = dataset.to_table(columns=[col]).column(col)
    total_bytes += arr.nbytes

print("Estimated memory if fully loaded (GB):", round(total_bytes / 1e9, 2))

print("\nDone.")