# gee_export_veg_indices.py — phiên bản fix lỗi null date
import ee
import pandas as pd

ee.Initialize(project="forestflameprediction")

ASSET  = "projects/forestflameprediction/assets/grid_bounds"
FOLDER = "daklak_fire_project"

grid_fc = ee.FeatureCollection(ASSET)

def mask_s2(img):
    qa   = img.select("QA60")
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
              .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return (img.updateMask(mask).divide(10000)
              .copyProperties(img)
              .set("system:time_start", img.get("system:time_start")))

def add_indices(img):
    ndvi = img.normalizedDifference(["B8",  "B4" ]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3",  "B8" ]).rename("NDWI")
    nbr  = img.normalizedDifference(["B8",  "B12"]).rename("NBR")
    ndii = img.normalizedDifference(["B8",  "B11"]).rename("NDII")
    return img.addBands([ndvi, ndwi, nbr, ndii])

def stats_per_image(img):
    ts = img.get("system:time_start")
    date_str = ee.Algorithms.If(
        ts,
        ee.Date(ts).format("YYYY-MM-dd"),
        ee.String("unknown")
    )
    def per_cell(feat):
        s = img.select(["NDVI","NDWI","NBR","NDII"]).reduceRegion(
            reducer    = ee.Reducer.mean().combine(
                             ee.Reducer.stdDev(), sharedInputs=True),
            geometry   = feat.geometry(),
            scale      = 20,
            maxPixels  = 1e8,
            bestEffort = True,
        )
        return feat.set(s).set("date", date_str)
    return grid_fc.map(per_cell)

def export_year(year):
    start = f"{year}-10-01"
    end   = f"{year}-12-31" if year == 2024 else f"{year+1}-06-30"

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(grid_fc)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        # ── Fix: lọc bỏ ảnh có system:time_start bị null ──────────
        .filter(ee.Filter.notNull(["system:time_start"]))
        # ── Fix: lọc bỏ ảnh thiếu band cần thiết ──────────────────
        .filter(ee.Filter.listContains("system:band_names", "B8"))
        .filter(ee.Filter.listContains("system:band_names", "B11"))
        .filter(ee.Filter.listContains("system:band_names", "B12"))
        .map(mask_s2)
        .map(add_indices)
    )

    task = ee.batch.Export.table.toDrive(
        collection     = col.map(stats_per_image).flatten(),
        description    = f"daklak_veg_{year}",
        folder         = FOLDER,
        fileNamePrefix = f"veg_{year}",
        fileFormat     = "CSV",
        selectors      = [
            "date", "grid_id",
            "NDVI_mean", "NDVI_stdDev",
            "NDWI_mean", "NBR_mean", "NDII_mean",
        ]
    )
    task.start()
    print(f"Started: veg_{year}  ({start} → {end})")

for year in range(2015, 2025):
    export_year(year)

print("\nTất cả task đã start.")
print("Theo dõi: code.earthengine.google.com/tasks")