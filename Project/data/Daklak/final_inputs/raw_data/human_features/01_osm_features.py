"""
01_osm_features.py — OSM Proximity Features (Phase 1)

Computes 3 distance features for every grid cell in Đắk Lắk:
    dist_road_km        — distance to nearest road (any highway tag)
    dist_settlement_km  — distance to nearest village/town/city/hamlet node
    dist_powerline_km   — distance to nearest power line

Note: dist_forest_edge_km is computed in 02_lulc_hansen.py (requires ESA raster).

Output: features/osm_distances.parquet
        columns: grid_id, dist_road_km, dist_settlement_km, dist_powerline_km

Usage:
    python 01_osm_features.py [--pbf path/to/vietnam-latest.osm.pbf]

If --pbf is not provided the script looks for vietnam-latest.osm.pbf in the
rasters/ subfolder and downloads it from Geofabrik if missing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import osmium
import osmium.geom
import pandas as pd
import geopandas as gpd
import shapely
import shapely.wkb as wkblib
from shapely.geometry import Point, box

#  Paths 
HERE        = Path(__file__).parent
RASTERS_DIR = HERE / "rasters"
FEATURES_DIR = HERE / "features"
RASTERS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

PBF_DEFAULT  = RASTERS_DIR / "vietnam-latest.osm.pbf"
PBF_URL      = "https://download.geofabrik.de/asia/vietnam-latest.osm.pbf"
OUT_PARQUET  = FEATURES_DIR / "osm_distances.parquet"

#  Đắk Lắk bounding box 
BBOX = (107.4, 11.4, 108.9, 13.1)   # minlon, minlat, maxlon, maxlat
BBOX_GEOM = box(*BBOX)

#  Settlement place types to include 
PLACE_TAGS = {"village", "town", "city", "hamlet", "suburb", "quarter"}


#  OSM Handler
class DakLakExtractor(osmium.SimpleHandler):
    """Single-pass handler that collects road, settlement, and powerline geometries."""

    def __init__(self):
        super().__init__()
        self._wkb = osmium.geom.WKBFactory()
        self.roads       = []   # shapely LineStrings / MultiLineStrings
        self.settlements = []   # shapely Points
        self.powerlines  = []   # shapely LineStrings

    # ways → roads and power lines
    def way(self, w):
        tags = w.tags
        try:
            if "highway" in tags:
                geom = wkblib.loads(self._wkb.create_linestring(w), hex=True)
                if geom.intersects(BBOX_GEOM):
                    self.roads.append(geom)
            elif tags.get("power") == "line":
                geom = wkblib.loads(self._wkb.create_linestring(w), hex=True)
                if geom.intersects(BBOX_GEOM):
                    self.powerlines.append(geom)
        except Exception:
            pass

    # nodes → settlement centroids
    def node(self, n):
        if n.tags.get("place", "") in PLACE_TAGS:
            loc = n.location
            if (BBOX[0] <= loc.lon <= BBOX[2] and BBOX[1] <= loc.lat <= BBOX[3]):
                self.settlements.append(Point(loc.lon, loc.lat))


#  Distance helper (vectorized via shapely 2.x STRtree) 
def nearest_distance_km(grid_pts_utm: np.ndarray, target_geoms: list) -> np.ndarray:
    """
    For each point in grid_pts_utm, return the distance in km to the
    nearest geometry in target_geoms.  Uses shapely 2.x STRtree for speed.
    """
    if not target_geoms:
        print("  WARNING: no target geometries found — returning NaN column")
        return np.full(len(grid_pts_utm), np.nan)

    target_arr = np.array(target_geoms)
    tree       = shapely.STRtree(target_arr)
    nearest_idx = tree.nearest(grid_pts_utm)           # vectorized
    distances_m = shapely.distance(grid_pts_utm, target_arr[nearest_idx])
    return (distances_m / 1000).astype("float32")


#  Download PBF if missing
def ensure_pbf(pbf_path: Path) -> None:
    if pbf_path.exists():
        size_mb = pbf_path.stat().st_size / 1e6
        print(f"  Found {pbf_path.name} ({size_mb:.0f} MB)")
        return

    print(f"  {pbf_path.name} not found — downloading from Geofabrik (~300 MB) …")
    import requests
    r = requests.get(PBF_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(pbf_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%", end="", flush=True)
    print(f"\n  Saved to {pbf_path}")


#  Main 
def main(pbf_path: Path) -> None:
    # 1. Load grid centroids
    print("\n[1/5] Loading grid centroids …")
    sys.path.insert(0, str(HERE))
    from grid_utils import load_grid_points
    grid_gdf = load_grid_points(crs="EPSG:4326")
    grid_utm = grid_gdf.to_crs("EPSG:32648")
    grid_pts_utm = np.array(grid_utm.geometry)
    print(f"  {len(grid_gdf):,} grid cells")

    # 2. Download PBF if needed
    print("\n[2/5] Checking OSM PBF …")
    ensure_pbf(pbf_path)

    # 3. Parse PBF
    print("\n[3/5] Parsing OSM PBF (single pass, locations=True) …")
    handler = DakLakExtractor()
    handler.apply_file(str(pbf_path), locations=True)
    print(f"  Roads:       {len(handler.roads):,}")
    print(f"  Settlements: {len(handler.settlements):,}")
    print(f"  Powerlines:  {len(handler.powerlines):,}")

    # 4. Project to UTM 48N for metric distances
    print("\n[4/5] Projecting OSM geometries to UTM 48N …")
    def to_utm(geoms):
        if not geoms:
            return []
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        return np.array(gdf.to_crs("EPSG:32648").geometry)

    roads_utm      = to_utm(handler.roads)
    settlements_utm = to_utm(handler.settlements)
    powerlines_utm = to_utm(handler.powerlines)

    # 5. Compute distances
    print("\n[5/5] Computing nearest distances …")

    print("  dist_road_km …")
    dist_road = nearest_distance_km(grid_pts_utm, list(roads_utm))

    print("  dist_settlement_km …")
    dist_settlement = nearest_distance_km(grid_pts_utm, list(settlements_utm))

    print("  dist_powerline_km …")
    dist_powerline = nearest_distance_km(grid_pts_utm, list(powerlines_utm))

    # 6. Save
    out = pd.DataFrame({
        "grid_id":            grid_gdf["grid_id"].values,
        "dist_road_km":       dist_road,
        "dist_settlement_km": dist_settlement,
        "dist_powerline_km":  dist_powerline,
    })
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved -> {OUT_PARQUET}")
    print(out.describe().to_string())

    # Note on dist_forest_edge_km
    print("\nNOTE: dist_forest_edge_km will be added by 02_lulc_hansen.py")
    print("      (requires ESA WorldCover raster)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pbf", type=Path, default=PBF_DEFAULT,
        help="Path to vietnam-latest.osm.pbf (default: rasters/vietnam-latest.osm.pbf)"
    )
    args = parser.parse_args()
    main(args.pbf)
