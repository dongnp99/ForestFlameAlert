"""
02_lulc_hansen.py - Land Cover + Forest Change Features (Phase 2)

Computes 5 features:
    lulc_class          (int8)    - majority land cover class per grid cell (ESA WorldCover)
    cropland_frac_1km   (float32) - fraction of cropland (class 40) within grid cell
    tree_cover_pct      (float32) - mean tree cover % in 2000 (Hansen GFC)
    dist_forest_edge_km (float32) - distance to nearest forest pixel (ESA forest mask)
    deforestation_lag_1y (float32, year-varying) - forest loss area (ha) in previous year

Outputs:
    features/lulc_hansen.parquet         - grid_id + 4 static features
    features/deforestation_by_year.parquet - grid_id + year + deforestation_lag_1y

ESA WorldCover tiles needed (auto-downloaded):
    N09E105, N09E108, N12E105, N12E108

Hansen GFC tile needed (auto-downloaded):
    10N_100E  (treecover2000 + lossyear)

Usage:
    python 02_lulc_hansen.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import rasterio.warp
from rasterstats import zonal_stats
from scipy.ndimage import distance_transform_edt
import requests

# -- Paths ---------------------------------------------------------------------
HERE         = Path(__file__).parent
RASTERS_DIR  = HERE / "rasters"
FEATURES_DIR = HERE / "features"
RASTERS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

OUT_STATIC  = FEATURES_DIR / "lulc_hansen.parquet"
OUT_DEFOR   = FEATURES_DIR / "deforestation_by_year.parquet"

# -- Bbox and CRS --------------------------------------------------------------
BBOX = (107.39, 12.06, 109.56, 13.79)  # actual grid extent + 0.1 deg buffer
BBOX_WGS84 = {
    "type": "Polygon",
    "coordinates": [[
        [BBOX[0], BBOX[1]], [BBOX[2], BBOX[1]],
        [BBOX[2], BBOX[3]], [BBOX[0], BBOX[3]],
        [BBOX[0], BBOX[1]],
    ]]
}

# -- ESA WorldCover ------------------------------------------------------------
ESA_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
# Grid lat range 12.06-13.79 is fully covered by N12 tiles (12N-15N)
# Grid lon range 107.39-109.56 covered by N12E105 (105-108E) + N12E108 (108-111E)
ESA_TILES = ["N12E105", "N12E108"]

# ESA class codes
ESA_TREE_COVER = 10   # Forest class used for forest mask

# -- Hansen GFC ----------------------------------------------------------------
HANSEN_BASE = (
    "https://storage.googleapis.com/earthenginepartners-hansen"
    "/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_{layer}_20N_100E.tif"
)
# Dataset years 2015-2024 -> deforestation_lag_1y uses loss from 2014-2023
# Hansen lossyear encodes: 0=no loss, 1=2001, ..., 23=2023
DATASET_YEARS = list(range(2015, 2025))


# -- Download helper -----------------------------------------------------------
def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  Downloading {dest.name} ...", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done/total*100:.1f}%", end="", flush=True)
    print(f"\r    Done ({done/1e6:.0f} MB)")


# -- Raster helpers ------------------------------------------------------------
def merge_and_clip_esa(tile_paths: list[Path], out_path: Path) -> None:
    """Merge ESA tiles and clip to Đak Lak bbox. Saves a GeoTIFF."""
    if out_path.exists():
        print(f"  Merged ESA raster already exists: {out_path.name}")
        return

    print("  Merging and clipping ESA WorldCover tiles ...")
    srcs = [rasterio.open(p) for p in tile_paths]
    merged, merged_transform = rio_merge(srcs)
    merged_crs = srcs[0].crs
    for s in srcs:
        s.close()

    # Clip to bbox
    from shapely.geometry import box as sbox
    import geopandas as gpd
    clip_geom = gpd.GeoDataFrame(
        geometry=[sbox(*BBOX)], crs="EPSG:4326"
    )
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=merged.shape[1], width=merged.shape[2],
            count=1, dtype=merged.dtype, crs=merged_crs, transform=merged_transform
        ) as tmp:
            tmp.write(merged)
        with memfile.open() as tmp:
            out_arr, out_transform = rio_mask(
                tmp, clip_geom.geometry, crop=True, nodata=0
            )

    profile = {
        "driver": "GTiff", "dtype": merged.dtype,
        "width": out_arr.shape[2], "height": out_arr.shape[1],
        "count": 1, "crs": merged_crs, "transform": out_transform,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr[0], 1)
    print(f"  Saved: {out_path.name}  shape={out_arr.shape[1:]}")


def clip_hansen(src_path: Path, out_path: Path) -> None:
    """Clip a Hansen tile to Đak Lak bbox."""
    if out_path.exists():
        print(f"  Clipped Hansen raster already exists: {out_path.name}")
        return
    print(f"  Clipping {src_path.name} to bbox ...")
    from shapely.geometry import box as sbox
    clip_geom = gpd.GeoDataFrame(geometry=[sbox(*BBOX)], crs="EPSG:4326")
    with rasterio.open(src_path) as src:
        out_arr, out_transform = rio_mask(src, clip_geom.geometry, crop=True, nodata=0)
        profile = src.profile.copy()
        profile.update(
            width=out_arr.shape[2], height=out_arr.shape[1],
            transform=out_transform, compress="lzw"
        )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr[0], 1)
    print(f"  Saved: {out_path.name}  shape={out_arr.shape[1:]}")


# -- Main ----------------------------------------------------------------------
def main() -> None:
    sys.path.insert(0, str(HERE))
    from grid_utils import load_grid_polygons, load_grid_points

    # ── 1. Download ESA WorldCover tiles ──────────────────────────────────────
    print("\n[1/7] Downloading ESA WorldCover tiles ...")
    esa_tile_paths = []
    for tile in ESA_TILES:
        dest = RASTERS_DIR / f"ESA_WorldCover_2021_{tile}.tif"
        download(ESA_BASE.format(tile=tile), dest)
        esa_tile_paths.append(dest)

    # ── 2. Merge + clip ESA ───────────────────────────────────────────────────
    print("\n[2/7] Merging ESA tiles ...")
    esa_merged_path = RASTERS_DIR / "esa_worldcover_daklak.tif"
    merge_and_clip_esa(esa_tile_paths, esa_merged_path)

    # ── 3. Download Hansen GFC tiles ──────────────────────────────────────────
    print("\n[3/7] Downloading Hansen GFC tiles ...")
    hansen_tc_raw   = RASTERS_DIR / "hansen_treecover2000_20N_100E.tif"
    hansen_loss_raw = RASTERS_DIR / "hansen_lossyear_20N_100E.tif"
    download(HANSEN_BASE.format(layer="treecover2000"), hansen_tc_raw)
    download(HANSEN_BASE.format(layer="lossyear"),      hansen_loss_raw)

    # Clip Hansen to bbox
    hansen_tc_clip   = RASTERS_DIR / "hansen_treecover2000_daklak.tif"
    hansen_loss_clip = RASTERS_DIR / "hansen_lossyear_daklak.tif"
    clip_hansen(hansen_tc_raw,  hansen_tc_clip)
    clip_hansen(hansen_loss_raw, hansen_loss_clip)
    # Delete wrong 10N tiles if they exist
    for old in ["hansen_treecover2000_10N_100E.tif", "hansen_lossyear_10N_100E.tif"]:
        p = RASTERS_DIR / old
        if p.exists():
            p.unlink()
            print(f"  Removed wrong tile: {old}")

    # ── 4. Build grid polygons ────────────────────────────────────────────────
    print("\n[4/7] Building grid polygons ...")
    grid_poly = load_grid_polygons(crs="EPSG:4326")
    grid_pts  = load_grid_points(crs="EPSG:4326")
    print(f"  {len(grid_poly):,} grid polygons")

    # ── 5. Zonal stats: ESA WorldCover ────────────────────────────────────────
    print("\n[5/7] ESA WorldCover zonal stats (majority + cropland fraction) ...")
    if OUT_STATIC.exists():
        print("  lulc_hansen.parquet already exists, loading cached values ...")
        _cached = pd.read_parquet(OUT_STATIC)
        lulc_class    = _cached["lulc_class"].values
        cropland_frac = _cached["cropland_frac_1km"].values
        tree_cover_pct = _cached["tree_cover_pct"].values
        print(f"  Loaded {len(lulc_class):,} rows from cache")
    else:
        print("  Running zonal_stats on ESA raster (10m, may take a few minutes) ...")

        # rasterstats 0.20+ passes (array, properties) to add_stats functions
        def cropland_frac_fn(x, props):
            valid = x.data[~x.mask] if hasattr(x, 'mask') else x.data.ravel()
            return float(np.sum(valid == 40) / valid.size) if valid.size > 0 else 0.0

        esa_stats = zonal_stats(
            grid_poly,
            str(esa_merged_path),
            stats=["majority"],
            add_stats={"cropland_frac": cropland_frac_fn},
            nodata=0,
        )

        lulc_class      = np.array([s.get("majority") or 0 for s in esa_stats], dtype="int8")
        cropland_frac   = np.array([s.get("cropland_frac") or 0.0 for s in esa_stats], dtype="float32")
        print(f"  lulc_class: {np.unique(lulc_class)}")
        print(f"  cropland_frac mean: {cropland_frac.mean():.3f}")

        # ── Hansen tree cover (only if not cached) ─────────────────────────────
        print("  tree_cover_pct ...")
        tc_stats = zonal_stats(
            grid_poly,
            str(hansen_tc_clip),
            stats=["mean"],
            nodata=0,
        )
        tree_cover_pct = np.array(
            [s.get("mean") or 0.0 for s in tc_stats], dtype="float32"
        )
        print(f"  tree_cover_pct mean: {tree_cover_pct.mean():.1f}%")

    # ── 6. Zonal stats: Hansen tree cover (header kept for step numbering) ────
    print("\n[6/7] Hansen GFC deforestation stats ...")

    # deforestation_lag_1y per year
    # loss_year encoding: 1=2001, ..., 14=2014, ..., 23=2023
    print("  deforestation_lag_1y per year ...")
    # Factory to avoid rasterstats 0.20+ positional-arg collision
    def make_defor_fn(loss_code: int):
        def defor_ha(x, props):
            valid = x.data[~x.mask] if hasattr(x, 'mask') else x.data.ravel()
            return float(np.sum(valid == np.uint8(loss_code)) * 0.09)
        return defor_ha

    defor_records = []
    for year in DATASET_YEARS:
        prev_year = year - 1
        loss_code = prev_year - 2000          # e.g. 2022 -> 22
        if loss_code < 1 or loss_code > 23:
            continue
        # zonal stats: count pixels where lossyear == loss_code
        # each 30m pixel = 0.09 ha
        target_stats = zonal_stats(
            grid_poly,
            str(hansen_loss_clip),
            add_stats={"defor_ha": make_defor_fn(loss_code)},
            nodata=None,
        )
        vals = np.array([s.get("defor_ha") or 0.0 for s in target_stats], dtype="float32")
        defor_records.append(
            pd.DataFrame({
                "grid_id":             grid_pts["grid_id"].values,
                "year":                year,
                "deforestation_lag_1y": vals,
            })
        )
        print(f"    year {year}: mean defor_ha={vals.mean():.4f}")

    defor_df = pd.concat(defor_records, ignore_index=True)
    defor_df.to_parquet(OUT_DEFOR, index=False)
    print(f"  Saved -> {OUT_DEFOR.name}  shape={defor_df.shape}")

    # ── 7. dist_forest_edge_km via EDT on ESA forest mask ─────────────────────
    print("\n[7/7] Computing dist_forest_edge_km ...")

    with rasterio.open(esa_merged_path) as src:
        esa_arr    = src.read(1)
        esa_tf     = src.transform
        esa_crs    = src.crs
        # pixel size in degrees (WGS84) -> convert to metres for EDT scaling
        px_lon_deg = abs(esa_tf.a)   # ~0.000083 deg at 10m
        px_lat_deg = abs(esa_tf.e)

    # Approximate pixel size in km at Đak Lak centre lat (~12.9N)
    import math
    lat_centre = (BBOX[1] + BBOX[3]) / 2
    px_x_km = px_lon_deg * math.cos(math.radians(lat_centre)) * 111.32
    px_y_km = px_lat_deg * 110.574
    print(f"  ESA pixel size: {px_x_km*1000:.1f} m x {px_y_km*1000:.1f} m")

    # Forest mask: 1 = forest (class 10), 0 = non-forest
    forest_mask = (esa_arr == ESA_TREE_COVER).astype("uint8")
    non_forest  = ~forest_mask.astype(bool)

    # EDT: distance (in pixels) from each non-forest pixel to nearest forest pixel
    # sampling = (row_scale, col_scale) i.e. (px_y_km, px_x_km)
    dist_px = distance_transform_edt(non_forest, sampling=(px_y_km, px_x_km))
    dist_km = dist_px.astype("float32")   # already in km due to sampling

    # Save distance raster for zonal stats sampling
    dist_raster_path = RASTERS_DIR / "dist_forest_edge_daklak.tif"
    with rasterio.open(esa_merged_path) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", compress="lzw")
    with rasterio.open(dist_raster_path, "w", **profile) as dst:
        dst.write(dist_km, 1)
    print(f"  Saved distance raster -> {dist_raster_path.name}")

    # Sample at grid centroids (nearest pixel, much faster than zonal mean)
    grid_pts_wgs = load_grid_points(crs="EPSG:4326")
    coords = list(zip(grid_pts_wgs["lon"], grid_pts_wgs["lat"]))
    with rasterio.open(dist_raster_path) as src:
        dist_vals = np.array(
            [v[0] for v in src.sample(coords)], dtype="float32"
        )
    print(f"  dist_forest_edge_km mean: {dist_vals.mean():.3f} km")

    # ── 8. Assemble static feature table ──────────────────────────────────────
    static_df = pd.DataFrame({
        "grid_id":            grid_pts["grid_id"].values,
        "lulc_class":         lulc_class,
        "cropland_frac_1km":  cropland_frac,
        "tree_cover_pct":     tree_cover_pct,
        "dist_forest_edge_km": dist_vals,
    })

    # Add dist_forest_edge_km into osm_distances.parquet (drop old column if present)
    osm_path = FEATURES_DIR / "osm_distances.parquet"
    if osm_path.exists():
        osm_df = pd.read_parquet(osm_path)
        osm_df = osm_df.drop(columns=[c for c in osm_df.columns if "dist_forest_edge" in c])
        osm_df = osm_df.merge(
            static_df[["grid_id", "dist_forest_edge_km"]], on="grid_id", how="left"
        )
        osm_df.to_parquet(osm_path, index=False)
        print(f"  Updated osm_distances.parquet with dist_forest_edge_km")

    static_df = static_df.drop(columns=["dist_forest_edge_km"])
    static_df.to_parquet(OUT_STATIC, index=False)
    print(f"\nSaved -> {OUT_STATIC.name}  shape={static_df.shape}")
    print(static_df.describe().to_string())

    print("\n[DONE] Phase 2 complete.")
    print(f"  {OUT_STATIC}")
    print(f"  {OUT_DEFOR}")


if __name__ == "__main__":
    main()
