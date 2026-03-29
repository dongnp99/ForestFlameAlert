# fix_grid_bounds.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

DATA_PATH = "../daklak_final_dataset.parquet"
OUT_PATH  = "grid_bounds.parquet"

one_day = pd.read_parquet(DATA_PATH, filters=[
    ("date", "=", pd.Timestamp("2020-01-01"))
])[["grid_id", "lon", "lat"]].drop_duplicates("grid_id")

lon_min, lon_max = one_day["lon"].min(), one_day["lon"].max()
lat_min, lat_max = one_day["lat"].min(), one_day["lat"].max()

# Grid layout từ s2_build_grid_maps.py
H, W = 137, 138

# Kích thước cell thực = toàn bộ range / số bước
cell_lon = (lon_max - lon_min) / (W - 1)
cell_lat = (lat_max - lat_min) / (H - 1)

print(f"Cell size đúng: {cell_lon:.6f}° × {cell_lat:.6f}°")
print(f"  ≈ {cell_lon * 111:.2f} km × {cell_lat * 111:.2f} km")

# Tạo polygon đúng kích thước
one_day["geometry"] = one_day.apply(
    lambda r: box(
        r.lon - cell_lon / 2, r.lat - cell_lat / 2,
        r.lon + cell_lon / 2, r.lat + cell_lat / 2,
    ), axis=1
)

gdf = gpd.GeoDataFrame(one_day, geometry="geometry", crs="EPSG:4326")
gdf.to_parquet(OUT_PATH, index=False)

# Kiểm tra nhanh
sample_geom = gdf.iloc[0]["geometry"]
bounds = sample_geom.bounds  # (minx, miny, maxx, maxy)
width_km  = (bounds[2] - bounds[0]) * 111
height_km = (bounds[3] - bounds[1]) * 111
print(f"\nKiểm tra polygon đầu tiên:")
print(f"  Width:  {width_km:.3f} km")
print(f"  Height: {height_km:.3f} km")
print(f"\nSaved {len(gdf)} cells → {OUT_PATH}")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 9))

# Vẽ tất cả polygon (outline)
gdf.plot(ax=ax, facecolor="none", edgecolor="#378ADD", linewidth=0.2, alpha=0.6)

# Highlight vài cell đầu để kiểm tra
gdf.head(10).plot(ax=ax, facecolor="#E24B4A", alpha=0.5, edgecolor="darkred", linewidth=0.5)

ax.set_title(f"Grid Daklak — {len(gdf)} cells (~1.5km)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("grid_check_v2.png", dpi=150)
print("Saved grid_check_v2.png")