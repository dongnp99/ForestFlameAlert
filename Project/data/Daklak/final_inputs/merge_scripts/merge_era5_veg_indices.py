import pandas as pd
import gc
from pathlib import Path

VEG_DIR   = Path("../xgb_veg_indices")
IN_FILE   = "../daklak_era5_firms.csv"
OUT_FILE  = "../daklak_era5_firms_veg.csv"
CHUNKSIZE = 500_000

VEG_COLS  = ["date", "grid_id", "NDVI_mean", "NDVI_stdDev", "NDWI_mean", "NBR_mean", "NDII_mean"]
RENAME    = {
    "NDVI_mean":   "ndvi",
    "NDVI_stdDev": "ndvi_std",
    "NDWI_mean":   "ndwi",
    "NBR_mean":    "nbr",
    "NDII_mean":   "ndii",
}

#  1. Load all veg CSVs into memory ─
print("Loading veg indices...")
dfs = []
for year in range(2015, 2025):
    # Prefer v2 (Cloud Score+ masking), fall back to v1
    for prefix in ["veg_v2_", "veg_"]:
        f = VEG_DIR / f"{prefix}{year}.csv"
        if f.exists():
            df = pd.read_csv(f, usecols=VEG_COLS, parse_dates=["date"])
            for col in ["NDVI_mean", "NDVI_stdDev", "NDWI_mean", "NBR_mean", "NDII_mean"]:
                df[col] = df[col].astype("float32")
            df["grid_id"] = df["grid_id"].astype("int32")
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df):,} rows")
            break
    else:
        print(f"  WARNING: no veg file found for {year}, skipping")

veg = pd.concat(dfs, ignore_index=True)
veg = veg.rename(columns=RENAME)
veg = veg.drop_duplicates(subset=["date", "grid_id"])
veg = veg.set_index(["date", "grid_id"])
del dfs
gc.collect()
print(f"Total veg rows loaded: {len(veg):,}\n")

#  2. Chunked merge with daklak_era5_firms.csv ─
print("Merging with ERA5+FIRMS...")
first = True

for i, chunk in enumerate(pd.read_csv(IN_FILE, chunksize=CHUNKSIZE, parse_dates=["date"])):
    chunk["grid_id"] = chunk["grid_id"].astype("int32")

    # Left join: all ERA5 rows kept, NaN where no S2 observation on that day
    chunk = chunk.join(veg, on=["date", "grid_id"], how="left")

    chunk.to_csv(
        OUT_FILE,
        mode="w" if first else "a",
        header=first,
        index=False,
    )
    first = False

    if (i + 1) % 10 == 0:
        print(f"  chunk {i + 1} done")

print(f"\nDone. Output: {OUT_FILE}")