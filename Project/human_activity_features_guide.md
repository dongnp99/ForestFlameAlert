# Human Activity Features — Hướng dẫn lấy dữ liệu
**Dự án:** Dự báo cháy rừng Đắk Lắk · XGBoost Model  
**Cập nhật:** 2024 · 16 features · 5 nguồn dữ liệu

---

## Tổng quan

| Nhóm nguồn | Số features | Độ khó | Cập nhật |
|---|---|---|---|
| OpenStreetMap (OSM) | 4 | Dễ | 1 lần (tĩnh) |
| ESA WorldCover / Hansen GFC | 4 | Dễ–Trung bình | Hàng năm |
| Tính từ dataset hiện có | 5 | Dễ | Tự động theo ngày |
| VIIRS Night-time Lights / WorldPop | 2 | Trung bình | Hàng năm |
| Lịch nông nghiệp (tính từ DOY) | 1 | Dễ | Tự động |

---

## Nguồn 1 — OpenStreetMap (OSM)

**4 features · Tĩnh · Join 1 lần theo grid_id**

| Feature | Dtype | Đơn vị | Mô tả |
|---|---|---|---|
| `dist_road_km` | float32 | km | Khoảng cách tới đường gần nhất (mọi loại) |
| `dist_settlement_km` | float32 | km | Khoảng cách tới điểm dân cư gần nhất |
| `dist_forest_edge_km` | float32 | km | Khoảng cách tới ranh giới rừng |
| `dist_powerline_km` | float32 | km | Khoảng cách tới đường dây điện |

### Cách lấy dữ liệu

**Bước 1 — Tải OSM extract cho Việt Nam:**
```bash
# Tải từ Geofabrik (~300MB)
wget https://download.geofabrik.de/asia/vietnam-latest.osm.pbf

# Hoặc chỉ lấy tỉnh Đắk Lắk (nhỏ hơn)
# Dùng Osmium để crop theo bounding box Đắk Lắk
pip install osmium
osmium extract --bbox=107.4,11.4,108.9,13.1 vietnam-latest.osm.pbf -o daklak.osm.pbf
```

**Bước 2 — Tính khoảng cách bằng GeoPandas:**
```python
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import osmnx as ox

# Tải roads, settlements, powerlines từ OSM
roads       = ox.features_from_bbox(13.1, 11.4, 108.9, 107.4, tags={"highway": True})
settlements = ox.features_from_bbox(13.1, 11.4, 108.9, 107.4, tags={"place": ["village","town","city","hamlet"]})
powerlines  = ox.features_from_bbox(13.1, 11.4, 108.9, 107.4, tags={"power": "line"})

# Tải grid centroids từ meta parquet
meta = pd.read_parquet("daklak_final_dataset.parquet", columns=["grid_id","lat","lon"]).drop_duplicates("grid_id")
grid_gdf = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta.lon, meta.lat), crs="EPSG:4326")

# Project sang UTM 48N để tính khoảng cách theo km
grid_utm   = grid_gdf.to_crs("EPSG:32648")
roads_utm  = roads.to_crs("EPSG:32648")
settl_utm  = settlements.to_crs("EPSG:32648")
power_utm  = powerlines.to_crs("EPSG:32648")

# Tính distance (vectorized)
grid_utm["dist_road_km"]       = grid_utm.geometry.apply(lambda g: roads_utm.distance(g).min() / 1000)
grid_utm["dist_settlement_km"] = grid_utm.geometry.apply(lambda g: settl_utm.distance(g).min() / 1000)
grid_utm["dist_powerline_km"]  = grid_utm.geometry.apply(lambda g: power_utm.distance(g).min() / 1000)

# Lưu lookup table
osm_features = grid_utm[["grid_id","dist_road_km","dist_settlement_km","dist_powerline_km"]].copy()
osm_features.to_parquet("features/osm_distances.parquet", index=False)
```

**Bước 3 — Tính dist_forest_edge_km từ land cover (xem Nguồn 2):**
```python
# Sau khi có lulc_class, tính edge
from scipy.ndimage import distance_transform_edt
# forest_mask = raster nhị phân (1=rừng, 0=không)
# dist_from_forest = distance_transform_edt(~forest_mask) * pixel_size_km
```

**Bước 4 — Join vào dataset chính:**
```python
df = pd.read_parquet("daklak_final_dataset.parquet")
df = df.merge(osm_features, on="grid_id", how="left")
```

> **Lưu ý:** OSM data Vietnam khá đầy đủ cho đường và khu dân cư. Đường dây điện có thể thiếu ở vùng nông thôn — nên dùng EVN shapefile nếu có thêm.

---

## Nguồn 2 — ESA WorldCover & Hansen GFC

**4 features · Hàng năm · Cần xử lý raster**

| Feature | Dtype | Đơn vị | Nguồn | Mô tả |
|---|---|---|---|---|
| `lulc_class` | int8 | category | ESA WorldCover 2021 | Loại đất: rừng/nông nghiệp/cỏ/đất trống/khác |
| `cropland_frac_1km` | float32 | 0–1 | ESA WorldCover | Tỷ lệ đất nông nghiệp trong buffer 1km |
| `tree_cover_pct` | float32 | % | Hansen GFC | Phần trăm độ che phủ tán cây |
| `deforestation_lag_1y` | float32 | ha | Hansen GFC | Diện tích mất rừng năm trước |

### ESA WorldCover 2021 (10m resolution, miễn phí)

```python
# Tải qua API (không cần đăng ký)
import requests, zipfile, io

# Tile Đắk Lắk: S12E108 (kiểm tra tại https://esa-worldcover.org/en/data)
url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_S12E108_Map.tif"
r = requests.get(url)
with open("esa_worldcover_daklak.tif", "wb") as f:
    f.write(r.content)
```

```python
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats

# ESA WorldCover class mapping
# 10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland,
# 50=Built-up, 60=Bare/sparse, 70=Snow, 80=Water, 90=Wetland, 95=Mangrove

with rasterio.open("esa_worldcover_daklak.tif") as src:
    lulc_data = src.read(1)
    transform = src.transform

# Tính lulc_class và cropland_frac cho từng grid
# grid_polygons: GeoDataFrame với polygon của mỗi grid cell
stats = zonal_stats(
    grid_polygons,
    "esa_worldcover_daklak.tif",
    stats=["majority"],
    add_stats={"cropland_frac": lambda x: np.sum(x == 40) / x.size}
)

grid_polygons["lulc_class"]       = [s["majority"] for s in stats]
grid_polygons["cropland_frac_1km"] = [s["cropland_frac"] for s in stats]

# Map class sang simplified label
lulc_map = {10:"forest", 20:"shrub", 30:"grass", 40:"cropland",
            50:"urban", 60:"bare", 80:"water", 90:"wetland"}
grid_polygons["lulc_label"] = grid_polygons["lulc_class"].map(lulc_map)
```

### Hansen Global Forest Change (GFC)

```python
# Tải qua Google Earth Engine (cần tài khoản GEE miễn phí)
import ee
ee.Initialize()

hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

# Tree cover năm 2000 (baseline)
tree_cover = hansen.select("treecover2000")

# Loss layer: 1 = mất rừng, giá trị = năm mất (1=2001, ..., 23=2023)
loss_year = hansen.select("lossyear")

# Export cho Đắk Lắk
daklak_bbox = ee.Geometry.Rectangle([107.4, 11.4, 108.9, 13.1])

task = ee.batch.Export.image.toDrive(
    image=tree_cover.addBands(loss_year),
    description="hansen_daklak",
    region=daklak_bbox,
    scale=30,
    crs="EPSG:4326"
)
task.start()
```

```python
# Sau khi tải về, tính deforestation_lag_1y
# loss_year == 22 → mất rừng năm 2022 → deforestation_lag_1y cho năm 2023

def compute_deforestation_lag(grid_polygons, loss_raster_path, current_year):
    target_loss_val = current_year - 2000  # vd: 2023 → 23
    stats = zonal_stats(
        grid_polygons, loss_raster_path,
        add_stats={"defor_ha": lambda x: np.sum(x == target_loss_val) * 0.09}  # 30m pixel = 0.09 ha
    )
    return [s["defor_ha"] for s in stats]
```

> **Lưu ý:** Nếu không có GEE, tải Hansen tiles trực tiếp tại:  
> `https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/`  
> File cần: `Hansen_GFC-2023-v1.11_treecover2000_10N_110E.tif` và `lossyear_10N_110E.tif`

---

## Nguồn 3 — Tính từ dataset hiện có (fire column)

**5 features · Không cần dữ liệu ngoài · Tính trực tiếp từ parquet**

| Feature | Dtype | Mô tả |
|---|---|---|
| `fire_count_prev_year` | int16 | Số lần fire=1 trong năm trước tại grid đó |
| `fire_count_prev_3y` | int16 | Số lần fire=1 trong 3 năm trước |
| `fire_freq_5y` | float32 | Tần suất cháy / năm trong 5 năm (= count / 5) |
| `days_since_last_fire` | int16 | Số ngày từ lần fire=1 gần nhất (-1 nếu chưa từng cháy) |
| `burn_season_flag` | int8 | 1 nếu tháng 1–4 (mùa đốt rẫy Đắk Lắk), 0 nếu không |

### Code tính toán

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("daklak_final_dataset.parquet",
                     columns=["grid_id","date","fire"])
df = df.sort_values(["grid_id","date"])
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# ── burn_season_flag (đơn giản nhất) ─────────────────────────────────
df["burn_season_flag"] = df["date"].dt.month.isin([1,2,3,4]).astype("int8")

# ── fire_count_prev_year ──────────────────────────────────────────────
annual = (df.groupby(["grid_id","year"])["fire"]
            .sum()
            .reset_index()
            .rename(columns={"fire":"fire_count","year":"join_year"}))
annual["year_key"] = annual["join_year"] + 1  # shift để join với năm sau
df = df.merge(
    annual[["grid_id","year_key","fire_count"]].rename(columns={"year_key":"year","fire_count":"fire_count_prev_year"}),
    on=["grid_id","year"], how="left"
)
df["fire_count_prev_year"] = df["fire_count_prev_year"].fillna(0).astype("int16")

# ── fire_count_prev_3y ────────────────────────────────────────────────
annual3 = (df.groupby(["grid_id","year"])["fire"]
             .sum()
             .groupby("grid_id")
             .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()))
df["fire_count_prev_3y"] = annual3.fillna(0).astype("int16")

# ── fire_freq_5y ──────────────────────────────────────────────────────
annual5 = (df.groupby(["grid_id","year"])["fire"]
             .sum()
             .groupby("grid_id")
             .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
df["fire_freq_5y"] = annual5.fillna(0).astype("float32")

# ── days_since_last_fire ──────────────────────────────────────────────
def days_since(group):
    last_fire = pd.NaT
    result = []
    for row in group.itertuples():
        if last_fire is pd.NaT:
            result.append(-1)
        else:
            result.append((row.date - last_fire).days)
        if row.fire == 1:
            last_fire = row.date
    return result

df["days_since_last_fire"] = (
    df.groupby("grid_id", group_keys=False)
      .apply(lambda g: pd.Series(days_since(g), index=g.index))
      .astype("int16")
)

print("Fire history features done.")
print(df[["fire_count_prev_year","fire_freq_5y","days_since_last_fire","burn_season_flag"]].describe())
```

> **Lưu ý quan trọng:** `days_since_last_fire` và các rolling features phải được tính **trước khi split train/val/test** để tránh data leakage. Sau đó mới split theo date.

---

## Nguồn 4 — VIIRS Night-time Lights & WorldPop

**2 features · Hàng năm · Cần tải raster bên ngoài**

| Feature | Dtype | Đơn vị | Nguồn | Mô tả |
|---|---|---|---|---|
| `nightlight_mean` | float32 | nW/cm²/sr | VIIRS DNB (NASA) | Độ sáng đêm trung bình — proxy hoạt động kinh tế |
| `pop_density` | float32 | người/km² | WorldPop | Mật độ dân số |

### VIIRS Night-time Lights

```python
# Tải Annual VNL V2.2 (miễn phí, không cần đăng ký)
# URL: https://eogdata.mines.edu/nighttime_light/annual/v22/

import requests

year = 2022
url = f"https://eogdata.mines.edu/nighttime_light/annual/v22/{year}/VNL_v22_npp_{year}_global_vcmslcfg_c202303062300.average_masked.dat.tif.gz"

# Hoặc dùng Google Earth Engine (dễ hơn):
import ee
ee.Initialize()

vnl = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
         .filterDate(f"{year}-01-01", f"{year}-12-31")
         .select("avg_rad")
         .mean())

daklak_bbox = ee.Geometry.Rectangle([107.4, 11.4, 108.9, 13.1])
task = ee.batch.Export.image.toDrive(
    image=vnl, description=f"viirs_nightlight_{year}",
    region=daklak_bbox, scale=500, crs="EPSG:4326"
)
task.start()
```

```python
# Sau khi tải về, tính zonal stats cho từng grid
from rasterstats import zonal_stats

stats = zonal_stats(grid_polygons, f"viirs_nightlight_{year}.tif", stats=["mean"])
grid_polygons["nightlight_mean"] = [s["mean"] or 0.0 for s in stats]
grid_polygons["nightlight_mean"] = grid_polygons["nightlight_mean"].astype("float32")
```

### WorldPop (mật độ dân số)

```python
# Tải trực tiếp không cần đăng ký
# Vietnam 2020, 100m resolution
url = "https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km/2020/VNM/vnm_pd_2020_1km.tif"

import requests
r = requests.get(url, stream=True)
with open("worldpop_vietnam_2020.tif", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
```

```python
# Tính cho từng grid cell
stats = zonal_stats(grid_polygons, "worldpop_vietnam_2020.tif", stats=["mean"])
grid_polygons["pop_density"] = [s["mean"] or 0.0 for s in stats]
grid_polygons["pop_density"] = grid_polygons["pop_density"].astype("float32")
```

> **Lưu ý:** WorldPop có resolution 100m và 1km. Dùng 1km cho tốc độ, 100m nếu cần chính xác hơn.

---

## Nguồn 5 — Lịch nông nghiệp (tính từ DOY)

**1 feature · Tự động · Không cần dữ liệu ngoài**

| Feature | Dtype | Mô tả |
|---|---|---|
| `days_since_harvest` | int16 | Số ngày kể từ sau thu hoạch cà phê (ước lượng theo DOY) |

```python
# Lịch mùa vụ Đắk Lắk (ước lượng theo kiến thức địa phương)
# Cà phê: thu hoạch tháng 11–12, đốt dọn tháng 1–3
# Lúa nương: thu hoạch tháng 10–11, đốt dọn tháng 12–2

def days_since_harvest(date):
    """
    Tính số ngày kể từ sau thu hoạch cà phê (kết thúc 31/12).
    Nếu chưa qua mùa thu hoạch trong năm đó → tính từ thu hoạch năm trước.
    """
    harvest_end_doy = 365  # 31 tháng 12
    doy = date.dayofyear
    if doy >= harvest_end_doy:
        return doy - harvest_end_doy
    else:
        # Tính từ harvest_end năm trước (365 - harvest_end_doy + doy)
        return 365 - harvest_end_doy + doy

df["days_since_harvest"] = df["date"].apply(days_since_harvest).astype("int16")

# Validate: tháng 1 → ~1–31 ngày, tháng 2 → ~32–59, tháng 6 → ~180
```

---

## Pipeline tổng hợp

```python
import pandas as pd

# 1. Load dataset gốc
df = pd.read_parquet("daklak_final_dataset.parquet")

# 2. Join các feature tĩnh (tính 1 lần)
osm      = pd.read_parquet("features/osm_distances.parquet")        # Nguồn 1
lulc     = pd.read_parquet("features/lulc_hansen.parquet")          # Nguồn 2
ntl_pop  = pd.read_parquet("features/nightlight_pop.parquet")       # Nguồn 4

df = (df
      .merge(osm,     on="grid_id", how="left")
      .merge(lulc,    on="grid_id", how="left")
      .merge(ntl_pop, on="grid_id", how="left"))

# 3. Tính fire history features (Nguồn 3)
# [chạy đoạn code Nguồn 3 ở trên]

# 4. Tính agricultural calendar (Nguồn 5)
df["burn_season_flag"]   = df["date"].dt.month.isin([1,2,3,4]).astype("int8")
df["days_since_harvest"] = df["date"].apply(days_since_harvest).astype("int16")

# 5. Kiểm tra missing values
print(df[NEW_HUMAN_FEATURES].isnull().sum())

# 6. Lưu
df.to_parquet("daklak_final_dataset_v2_human.parquet", index=False)
print("Done. Shape:", df.shape)
```

---

## Cập nhật xgb_config.py

```python
# Thêm vào FEATURE_COLS:
HUMAN_FEATURE_COLS = [
    # === OSM Proximity ===
    "dist_road_km",
    "dist_settlement_km",
    "dist_forest_edge_km",
    "dist_powerline_km",

    # === Land Use / Land Cover ===
    "lulc_class",
    "cropland_frac_1km",
    "tree_cover_pct",
    "deforestation_lag_1y",

    # === Fire History ===
    "fire_count_prev_year",
    "fire_count_prev_3y",
    "fire_freq_5y",
    "days_since_last_fire",

    # === Human Presence ===
    "nightlight_mean",
    "pop_density",

    # === Agricultural Calendar ===
    "burn_season_flag",
    "days_since_harvest",
]

FEATURE_COLS = FEATURE_COLS + HUMAN_FEATURE_COLS
```

---

## Checklist triển khai

- [ ] Tải OSM extract Đắk Lắk từ Geofabrik
- [ ] Tính 4 distance features, lưu `features/osm_distances.parquet`
- [ ] Tải ESA WorldCover tile S12E108
- [ ] Tải Hansen GFC tiles cho Đắk Lắk (treecover2000 + lossyear)
- [ ] Tính lulc_class, cropland_frac_1km, tree_cover_pct, deforestation_lag_1y
- [ ] Tính 5 fire history features từ dataset hiện có
- [ ] Tải VIIRS NTL và WorldPop, tính zonal stats
- [ ] Tính burn_season_flag và days_since_harvest
- [ ] Kiểm tra null values cho tất cả 16 features
- [ ] Retrain model với FEATURE_COLS mới
- [ ] So sánh AUC-PR trước/sau trên val set
