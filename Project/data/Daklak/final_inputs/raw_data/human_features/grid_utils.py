"""
grid_utils.py — Shared grid GeoDataFrame builder for human_features pipeline.

Loads daklak_grid_lon_lat.csv and returns point and polygon GeoDataFrames
in both WGS84 (EPSG:4326) and UTM 48N (EPSG:32648).

Usage:
    from grid_utils import load_grid_points, load_grid_polygons
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

# Paths relative to this file
_HERE = Path(__file__).parent
_GRID_CSV = _HERE.parent / "daklak_grid_lon_lat.csv"

# Approximate cell half-width in metres (cells are ~1 km²)
CELL_HALF_M = 500.0


def load_grid_points(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with one point per grid centroid.

    Columns: grid_id, lon, lat, geometry
    """
    meta = pd.read_csv(_GRID_CSV)
    gdf = gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["lon"], meta["lat"]),
        crs="EPSG:4326",
    )
    if crs != "EPSG:4326":
        gdf = gdf.to_crs(crs)
    return gdf


def load_grid_polygons(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with one square polygon per grid cell.
    Polygons are built by buffering each centroid by CELL_HALF_M metres
    in UTM 48N, then optionally reprojected to `crs`.

    Columns: grid_id, lon, lat, geometry
    """
    gdf_utm = load_grid_points(crs="EPSG:32648")
    gdf_utm["geometry"] = gdf_utm.geometry.buffer(CELL_HALF_M, cap_style=3)  # square buffer
    if crs != "EPSG:32648":
        gdf_utm = gdf_utm.to_crs(crs)
    return gdf_utm
