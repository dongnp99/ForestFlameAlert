import pandas as pd
import pyarrow.parquet as pq

PATH = "veg_indices_daily.parquet"

# --- Schema & file metadata (no full load needed) ---
pf = pq.read_metadata(PATH)
print(f"Rows: {pf.num_rows:,}  |  Row groups: {pf.num_row_groups}  |  Size: {pf.serialized_size / 1e6:.1f} MB")

schema = pq.read_schema(PATH)
print("\nSchema:")
print(schema)

# --- Load full dataframe ---
df = pd.read_parquet(PATH)

print("\nDtypes:")
print(df.dtypes)

print("\nHead:")
print(df.head(3))

print("\nBasic stats:")
print(df.describe())

print(f"\ngrid_id range : {df['grid_id'].min()} → {df['grid_id'].max()}")
print(f"Unique grids  : {df['grid_id'].nunique():,}")
print(f"Date range    : {df['date'].min()} → {df['date'].max()}")
print(f"\nFire label distribution:")
print(df["fire"].value_counts())

print(f"\nMissing values:")
missing = df.isna().sum()
print(missing[missing > 0] if missing.any() else "None")