import pandas as pd
import pyarrow.parquet as pq

PATH = "features/nightlight_pop_by_year.parquet"

# --- Schema & file metadata (no full load needed) ---
pf = pq.read_metadata(PATH)
print(f"Rows: {pf.num_rows:,} |  Size: {pf.serialized_size / 1e6:.1f} MB")

# --- Load full dataframe ---
df = pd.read_parquet(PATH)

print("\nDtypes:")
print(df.dtypes)

# print(df[df["deforestation_lag_1y"] != 0].head(100))
print(df.head(100))
