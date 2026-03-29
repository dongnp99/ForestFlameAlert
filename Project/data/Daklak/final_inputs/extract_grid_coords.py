import pandas as pd

df = pd.read_csv("raw_data/daklak_era5.csv", usecols=["grid_id", "lon", "lat"])
grid_coords = df.drop_duplicates(subset="grid_id").sort_values("grid_id").reset_index(drop=True)
grid_coords.to_csv("raw_data/daklak_grid_lon_lat.csv", index=False)

print(f"Extracted {len(grid_coords)} unique grids")
print(grid_coords.head())