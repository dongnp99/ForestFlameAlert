import pandas as pd
import numpy as np

FILE = "../data/Daklak/final_inputs/clstm_data/clstm_clean_data.parquet"

GRID_ID_COL = "grid_id"
DATE_COL = "date"
FIRE_COL = "fire"

CHUNK = 2_000_000

print("Opening parquet...")

df_iter = pd.read_parquet(FILE)

rows = len(df_iter)
print("Total rows:", rows)

print("\n===== BASIC CHECK =====")

nan_count = 0
inf_count = 0

numeric_cols = df_iter.select_dtypes(include=[np.number]).columns

for start in range(0, rows, CHUNK):

    end = min(start + CHUNK, rows)

    chunk = df_iter.iloc[start:end]

    nan_count += chunk.isna().sum().sum()

    numeric = chunk[numeric_cols]

    inf_count += np.isinf(numeric).sum().sum()

    print("checked rows:", end)

print("Total NaN:", nan_count)
print("Total Inf:", inf_count)


print("\n===== FIRE CHECK =====")

fire = df_iter[FIRE_COL]

print("Fire min:", fire.min())
print("Fire max:", fire.max())

fire_ratio = (fire > 0).mean()

print("Fire ratio:", fire_ratio)

if fire.max() > 1:
    print("WARNING: fire label not binary")


print("\n===== GRID CHECK =====")

grid_per_day = df_iter.groupby(DATE_COL)[GRID_ID_COL].nunique()

print("Grid min/day:", grid_per_day.min())
print("Grid max/day:", grid_per_day.max())


print("\n===== EXTREME VALUE CHECK =====")

for col in numeric_cols:

    s = df_iter[col]

    max_val = s.max()
    min_val = s.min()

    if abs(max_val) > 1e6 or abs(min_val) > 1e6:

        print("Extreme values:", col, min_val, max_val)


print("\nCHECK DONE")
