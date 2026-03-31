"""
04_nightlight_pop.py - Night-time Lights + Population Density (Phase 4)

Computes 2 year-varying features for every grid cell x year:
    nightlight_mean  (float32) - mean annual VIIRS NTL radiance per grid cell
    pop_density      (float32) - mean population density (people/km2)

Input files (already present, no download needed):
    nightlight/viirs_ntl_{year}.tif   for years 2015-2024
    worldpop/vnm_pd_{year}_1km.tif    for years 2018-2020
        -> years 2015-2017 use 2018, years 2021-2024 use 2020

Output:
    features/nightlight_pop_by_year.parquet
        columns: grid_id, year, nightlight_mean, pop_density

Note: VIIRS tiles cover bbox (107.40-108.90 lon, 11.40-13.10 lat).
      Grid cells outside this extent (eastern/northern edge) will have
      nightlight_mean = 0.0 (remote/forested, acceptable approximation).

Usage:
    python 04_nightlight_pop.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio

# -- Paths ---------------------------------------------------------------------
HERE         = Path(__file__).parent
NIGHTLIGHT_DIR = HERE / "nightlight"
WORLDPOP_DIR   = HERE / "worldpop"
FEATURES_DIR   = HERE / "features"
FEATURES_DIR.mkdir(exist_ok=True)

OUT_PARQUET = FEATURES_DIR / "nightlight_pop_by_year.parquet"

DATASET_YEARS = list(range(2015, 2025))

# WorldPop year mapping (only 2018-2020 available)
WORLDPOP_YEAR_MAP = {y: 2018 for y in range(2015, 2018)}
WORLDPOP_YEAR_MAP.update({2018: 2018, 2019: 2019, 2020: 2020})
WORLDPOP_YEAR_MAP.update({y: 2020 for y in range(2021, 2025)})


# -- Helpers -------------------------------------------------------------------
def zonal_mean_from_file(grid_poly: gpd.GeoDataFrame,
                         raster_path: Path, col: str) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
    stats = zonal_stats(
        grid_poly, str(raster_path), stats=["mean"], nodata=nodata,
    )
    vals = np.array([s.get("mean") or 0.0 for s in stats], dtype="float32")
    print(f"    {col}: mean={vals.mean():.4f}  non-zero={np.sum(vals > 0):,}/{len(vals):,}")
    return vals


# -- Main ----------------------------------------------------------------------
def main() -> None:
    sys.path.insert(0, str(HERE))
    from grid_utils import load_grid_polygons, load_grid_points

    grid_poly = load_grid_polygons(crs="EPSG:4326")
    grid_pts  = load_grid_points(crs="EPSG:4326")
    grid_ids  = grid_pts["grid_id"].values
    print(f"Grid: {len(grid_poly):,} cells\n")

    records = []

    for year in DATASET_YEARS:
        print(f"[{year}]")

        # -- VIIRS nightlight --------------------------------------------------
        viirs_path = NIGHTLIGHT_DIR / f"viirs_ntl_{year}.tif"
        if not viirs_path.exists():
            print(f"  WARNING: {viirs_path.name} not found, filling nightlight_mean=0")
            nightlight = np.zeros(len(grid_poly), dtype="float32")
        else:
            nightlight = zonal_mean_from_file(grid_poly, viirs_path, "nightlight_mean")

        # -- WorldPop population density ---------------------------------------
        wp_year = WORLDPOP_YEAR_MAP[year]
        wp_path = WORLDPOP_DIR / f"vnm_pd_{wp_year}_1km.tif"
        if not wp_path.exists():
            print(f"  WARNING: {wp_path.name} not found, filling pop_density=0")
            pop = np.zeros(len(grid_poly), dtype="float32")
        else:
            pop = zonal_mean_from_file(grid_poly, wp_path, f"pop_density (wp{wp_year})")

        records.append(pd.DataFrame({
            "grid_id":         grid_ids,
            "year":            year,
            "nightlight_mean": nightlight,
            "pop_density":     pop,
        }))

    # -- Save ------------------------------------------------------------------
    out = pd.concat(records, ignore_index=True)
    out.to_parquet(OUT_PARQUET, index=False)

    print(f"\nSaved -> {OUT_PARQUET}")
    print(f"Shape: {out.shape}")
    print("\nNull counts:", out.isnull().sum().to_dict())
    print("\nPer-year nightlight_mean summary:")
    print(out.groupby("year")["nightlight_mean"].mean().round(4).to_string())
    print("\nPop density summary (unique WorldPop years):")
    print(out.groupby("year")["pop_density"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
