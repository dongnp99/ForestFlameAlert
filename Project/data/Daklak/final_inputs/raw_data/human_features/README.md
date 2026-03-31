# human_features — Human Activity Features for ForestFlameAlert

## Overview

This folder contains all scripts, raw raster downloads, and intermediate outputs
needed to compute **16 human-activity features** added to the XGBoost wildfire
prediction model for Đắk Lắk province.

These features complement the existing weather, terrain, seasonality, and
vegetation features in `daklak_final_dataset.parquet`.

---

## Directory Structure

```
human_features/
├── README.md                   ← This file
├── grid_utils.py               ← Shared helper: loads grid centroids & polygons
│
├── 01_osm_features.py          ← Source 1: OSM proximity distances
├── 02_lulc_hansen.py           ← Source 2: ESA WorldCover + Hansen GFC
├── 03_fire_history.py          ← Source 3: Fire history from existing dataset
├── 04_nightlight_pop.py        ← Source 4: VIIRS night-time lights + WorldPop
├── 05_assemble.py              ← Final assembly → daklak_final_dataset_v2_human.parquet
│
├── features/                   ← Intermediate output parquets (git-ignored)
│   ├── osm_distances.parquet
│   ├── lulc_hansen.parquet
│   └── nightlight_pop.parquet
│
└── rasters/                    ← Downloaded raster files (git-ignored, large)
    ├── esa_worldcover_daklak.tif
    ├── hansen_treecover2000_daklak.tif
    ├── hansen_lossyear_daklak.tif
    ├── viirs_nightlight_2022.tif
    └── worldpop_vietnam_2020.tif
```

---

## Feature Inventory (16 total)

| # | Feature | Type | Source | Script |
|---|---------|------|--------|--------|
| 1 | `dist_road_km` | float32 | OpenStreetMap | 01 |
| 2 | `dist_settlement_km` | float32 | OpenStreetMap | 01 |
| 3 | `dist_forest_edge_km` | float32 | OSM + ESA WorldCover | 01 + 02 |
| 4 | `dist_powerline_km` | float32 | OpenStreetMap | 01 |
| 5 | `lulc_class` | int8 | ESA WorldCover 2021 | 02 |
| 6 | `cropland_frac_1km` | float32 | ESA WorldCover 2021 | 02 |
| 7 | `tree_cover_pct` | float32 | Hansen GFC | 02 |
| 8 | `deforestation_lag_1y` | float32 | Hansen GFC | 02 |
| 9 | `fire_count_prev_year` | int16 | Existing dataset | 03 |
| 10 | `fire_count_prev_3y` | int16 | Existing dataset | 03 |
| 11 | `fire_freq_5y` | float32 | Existing dataset | 03 |
| 12 | `days_since_last_fire` | int16 | Existing dataset | 03 |
| 13 | `burn_season_flag` | int8 | Derived from date | 03 |
| 14 | `nightlight_mean` | float32 | VIIRS DNB (NASA) | 04 |
| 15 | `pop_density` | float32 | WorldPop 2020 | 04 |
| 16 | `days_since_harvest` | int16 | Derived from DOY | 05 |

---

## Data Sources

| Source | URL | License | Update frequency |
|--------|-----|---------|-----------------|
| OpenStreetMap | https://download.geofabrik.de/asia/vietnam-latest.osm.pbf | ODbL | ~weekly |
| ESA WorldCover 2021 | https://esa-worldcover.org/en/data | CC BY 4.0 | Annual |
| Hansen GFC 2023 | https://storage.googleapis.com/earthenginepartners-hansen/ | CC BY 4.0 | Annual |
| VIIRS Night-time Lights | https://eogdata.mines.edu/nighttime_light/annual/v22/ | Public domain | Annual |
| WorldPop Vietnam 2020 | https://data.worldpop.org/ | CC BY 4.0 | Annual |

---

## How to Run (in order)

```bash
# 0. Install dependencies (once)
pip install osmnx geopandas rasterio rasterstats scipy earthengine-api

# 1. OSM distances (static, run once)
python 01_osm_features.py
# → features/osm_distances.parquet

# 2. Land cover + forest change (static, run once per year)
python 02_lulc_hansen.py
# → features/lulc_hansen.parquet

# 3. Fire history (re-run when dataset updates)
python 03_fire_history.py
# → features/fire_history.parquet

# 4. Night-time lights + population (static, run once per year)
python 04_nightlight_pop.py
# → features/nightlight_pop.parquet

# 5. Assemble final dataset
python 05_assemble.py
# → ../../daklak_final_dataset_v2_human.parquet
```

---

## Grid Reference

- **Source file:** `raw_data/daklak_grid_lon_lat.csv`
- **Total cells:** 18,803
- **Approximate cell size:** ~1 km²
- **CRS:** WGS84 (EPSG:4326) for centroids; UTM 48N (EPSG:32648) for distance calculations
- **Bounding box:** lon [107.49, 109.46], lat [12.16, 13.69]

The shared `grid_utils.py` module provides `load_grid_points()` and
`load_grid_polygons()` — all scripts import from here to ensure consistency.

---

## Notes

- `features/` and `rasters/` contain large files and are not committed to git.
- OSM powerline coverage in rural Đắk Lắk may be incomplete; consider supplementing
  with EVN shapefiles if available.
- All fire history features must be computed on the **full dataset before any
  train/val/test split** to avoid data leakage.
- `deforestation_lag_1y` is year-varying: for each observation year Y, it counts
  Hansen loss pixels from year Y−1.
