"""
extract_terrain.py - Trích xuất độ dốc và hướng sườn từ raster terrain cho từng grid.

Đầu ra: daklak_terrain.csv
    grid_id       - ID của grid cell
    slope_mean    - Độ dốc trung bình (độ)
    slope_std     - Độ lệch chuẩn độ dốc
    aspect_sin    - Mean sin(aspect) — biểu diễn hướng sườn tránh discontinuity tại 0/360°
    aspect_cos    - Mean cos(aspect)

Raster đầu vào (EPSG:32648):
    terrain/daklak_slope_utm.tif
    terrain/daklak_asp_sin.tif
    terrain/daklak_asp_cos.tif

Usage:
    python extract_terrain.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

# -- Paths --------------------------------------------------------------------
HERE         = Path(__file__).parent
TERRAIN_DIR  = HERE.parent.parent.parent / "terrain"  # data/Daklak/terrain/
OUT_CSV      = HERE / "daklak_terrain.csv"

SLOPE_TIF      = TERRAIN_DIR / "daklak_slope_utm.tif"
ASP_SIN_TIF    = TERRAIN_DIR / "daklak_asp_sin.tif"
ASP_COS_TIF    = TERRAIN_DIR / "daklak_asp_cos.tif"

# -- Grid ---------------------------------------------------------------------
def load_grid_polygons_utm() -> gpd.GeoDataFrame:
    """Grid polygons 1km x 1km trong EPSG:32648 (khớp với CRS của raster terrain)."""
    grid_csv = HERE / "daklak_grid_lon_lat.csv"
    meta = pd.read_csv(grid_csv)
    gdf = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["lon"], meta["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32648")
    gdf["geometry"] = gdf.geometry.buffer(500.0, cap_style=3)  # ô vuông 1km
    return gdf


# -- Main ---------------------------------------------------------------------
def main() -> None:
    print("Đang tải grid polygons (UTM 48N)...")
    grid = load_grid_polygons_utm()
    print(f"  {len(grid):,} grid cells")

    #  Slope ─
    print("\nTính zonal stats - slope...")
    slope_stats = zonal_stats(
        grid,
        str(SLOPE_TIF),
        stats=["mean", "std"],
        nodata=-9999.0,
    )
    slope_mean = np.array([s["mean"] if s["mean"] is not None else np.nan for s in slope_stats], dtype="float32")
    slope_std  = np.array([s["std"]  if s["std"]  is not None else np.nan for s in slope_stats], dtype="float32")
    print(f"  slope_mean: min={np.nanmin(slope_mean):.2f}°, max={np.nanmax(slope_mean):.2f}°, mean={np.nanmean(slope_mean):.2f}°")

    #  Aspect sin/cos 
    # Dùng sin/cos để tránh discontinuity tại 0°/360° (North-facing)
    print("\nTính zonal stats - aspect sin...")
    asp_sin_stats = zonal_stats(
        grid,
        str(ASP_SIN_TIF),
        stats=["mean"],
        nodata=-3.4028234663852886e+38,
    )
    aspect_sin = np.array([s["mean"] if s["mean"] is not None else np.nan for s in asp_sin_stats], dtype="float32")

    print("Tính zonal stats - aspect cos...")
    asp_cos_stats = zonal_stats(
        grid,
        str(ASP_COS_TIF),
        stats=["mean"],
        nodata=-3.4028234663852886e+38,
    )
    aspect_cos = np.array([s["mean"] if s["mean"] is not None else np.nan for s in asp_cos_stats], dtype="float32")

    print(f"  aspect_sin: mean={np.nanmean(aspect_sin):.4f}")
    print(f"  aspect_cos: mean={np.nanmean(aspect_cos):.4f}")

    #  Lưu kết quả ─
    out_df = pd.DataFrame({
        "grid_id":    grid["grid_id"].values,
        "slope_mean": slope_mean,
        "slope_std":  slope_std,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
    })

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nĐã lưu -> {OUT_CSV.name}  ({len(out_df):,} rows)")
    print(out_df.describe().to_string())


if __name__ == "__main__":
    main()