import pandas as pd

# ===== 1. LOAD DEM (NHẸ) =====
dem = pd.read_csv("daklak_dem.csv")

# chỉ giữ cột cần thiết
dem_cols = [
    "grid_id",
    "date",
    "dem_mean",
    "dem_stdev",
    "dem_min",
    "dem_max",
    "slp_mean",
    "slp_stdev",
    "aspect_sin",
    "aspect_cos",
]
dem = dem[dem_cols]

# ===== 2. MERGE ERA5 + FIRE + DEM THEO CHUNK =====
chunksize = 500_000
out_file = "dataset_fire_final.csv"
first = True

print(f"Processing...")

for chunk in pd.read_csv(
        "daklak_era5_firms.csv",
    chunksize=chunksize,
    parse_dates=["date"]
):
    print(f"Chunking...")
    # merge DEM
    chunk = chunk.merge(dem, on="grid_id", how="left")

    # --- dọn cột (TÙY CHỌN, KHUYẾN NGHỊ) ---
    drop_cols = [
        "lon", "lat", "number", "mask"
    ]
    drop_cols = [c for c in drop_cols if c in chunk.columns]
    chunk = chunk.drop(columns=drop_cols)

    # ghi ra file
    chunk.to_csv(
        out_file,
        mode="w" if first else "a",
        header=first,
        index=False
    )
    first = False
    
    print(f"Chunked complete!")
