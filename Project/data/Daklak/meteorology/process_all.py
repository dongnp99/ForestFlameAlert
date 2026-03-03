import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import regionmask
import glob, os, sys

# =========================import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import glob, os, sys

# =========================
# CONFIG
# =========================
NC_DIR = "raw_nc"
GRID_SHP = "daklak_grid_1km_clip.shp"
OUT_DIR = "out_csv"
INTERP_METHOD = "nearest"   # "nearest" hoặc "linear"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== ERA5 → DEM GRID 1KM PIPELINE ===")

# =========================
# LOAD GRID DEM
# =========================
print("[1] Loading DEM grid...")
grid = gpd.read_file(GRID_SHP)

if grid.crs is None or grid.crs.to_epsg() != 4326:
    print("  → Reprojecting grid to EPSG:4326")
    grid = grid.to_crs(4326)

ID_COL = "id"   # đổi đúng tên cột DEM
grid["grid_id"] = grid[ID_COL].astype(int)

# lấy tâm ô
grid["lon"] = grid.geometry.centroid.x
grid["lat"] = grid.geometry.centroid.y

grid_df = grid[["grid_id", "lon", "lat"]].reset_index(drop=True)

print(f"  → Grid cells: {len(grid_df)} (expected ~36901)")

# =========================
# LOOP ERA5 FILES
# =========================
nc_files = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))

if len(nc_files) == 0:
    print("❌ No NetCDF files found")
    sys.exit(1)

print(f"[2] Found {len(nc_files)} ERA5 files")

for f in nc_files:
    print("\n========================================")
    print(">>> Processing:", os.path.basename(f))

    ds = xr.open_dataset(f)

    # chuẩn hóa time
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # =========================
    # MET VARIABLES
    # =========================
    print("  [2.1] Calculating meteorological variables...")

    t = ds["t2m"] - 273.15
    td = ds["d2m"] - 273.15

    rh = 100 * np.exp((17.625 * td) / (243.04 + td)) / \
               np.exp((17.625 * t) / (243.04 + t))

    wind = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
    rain = ds["tp"] * 1000  # mm

    vpd = (1 - rh / 100) * (
        0.6108 * np.exp((17.27 * t) / (t + 237.3))
    )

    # =========================
    # DAILY AGGREGATION
    # =========================
    print("  [2.2] Aggregating to daily values...")

    daily = xr.Dataset({
        "tmean": t.resample(time="1D").mean(),
        "rh": rh.resample(time="1D").mean(),
        "wind": wind.resample(time="1D").mean(),
        "rain": rain.resample(time="1D").sum(),
        "vpd": vpd.resample(time="1D").mean()
    })

    daily = daily.rename({"latitude": "lat", "longitude": "lon"})
    daily = daily.transpose("time", "lat", "lon")

    # =========================
    # 🔥 INTERPOLATE TO DEM GRID
    # =========================
    print("  [2.3] Interpolating ERA5 to DEM grid...")

    era5_on_grid = daily.interp(
    lon=xr.DataArray(grid_df["lon"].values, dims="grid"),
    lat=xr.DataArray(grid_df["lat"].values, dims="grid"),
    method=INTERP_METHOD
)

    # 🔥 FIX NaN ở biên (bắt buộc)
    era5_on_grid = era5_on_grid.interpolate_na(
        dim="grid",
        method="nearest",
        fill_value="extrapolate"
    )

    # =========================
    # EXPORT CSV
    # =========================
    print("  [2.4] Exporting CSV...")

    df = era5_on_grid.to_dataframe().reset_index()

    df["grid_id"] = grid_df["grid_id"].values[df["grid"]]
    df["date"] = pd.to_datetime(df["time"])

    df = df.drop(columns=["grid", "time"])

    out_name = os.path.basename(f).replace(".nc", "_grid1km.csv")
    out_path = os.path.join(OUT_DIR, out_name)
    ORDERED_COLS = [
        "grid_id",
        "date",
        "lon",
        "lat",
        "tmean",
        "rh",
        "wind",
        "rain",
        "vpd"
    ]
    df = df[ORDERED_COLS]
    df.to_csv(out_path, index=False)

    print(f"  ✔ Saved: {out_path}")
    print("  ✔ Rows per day:",
          df.groupby("date").size().iloc[0])

print("\n🔥 PIPELINE FINISHED SUCCESSFULLY")
