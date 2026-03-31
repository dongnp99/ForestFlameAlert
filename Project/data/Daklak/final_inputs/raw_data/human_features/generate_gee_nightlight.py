import ee

ee.Initialize(project="forestflameprediction")

DAKLAK_BBOX = ee.Geometry.Rectangle([107.4, 11.4, 108.9, 13.1])

def export_annual_ntl(year):
    col = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(f"{year}-01-01", f"{year}-12-31")
             .filterBounds(DAKLAK_BBOX)
             .select("avg_rad"))   # chỉ có band này

    # VCMSLCFG đã lọc cloud và stray light từ phía NOAA
    # Không cần mask thêm — bỏ hàm mask_clouds đi
    annual = col.median()

    task = ee.batch.Export.image.toDrive(
        image=annual,
        description=f"viirs_ntl_{year}_daklak",
        folder="fire_model_data",
        fileNamePrefix=f"viirs_ntl_{year}",
        region=DAKLAK_BBOX,
        scale=500,
        crs="EPSG:4326",
        maxPixels=1e9
    )
    task.start()
    print(f"Export started — year={year}  task_id={task.id}")
    return task

for year in range(2015, 2025):
    export_annual_ntl(year)