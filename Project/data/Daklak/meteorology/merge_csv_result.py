import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
ERA5_DIR = Path("out_csv")      # thư mục chứa các file theo tháng
OUT_FILE = "daklak_era5_firms.csv"

# =========================
# LOAD ALL CSV
# =========================
csv_files = sorted(ERA5_DIR.glob("*.csv"))
assert len(csv_files) > 0, "❌ Không tìm thấy file ERA5 CSV"

dfs = []
for f in csv_files:
    print("Loading:", f.name)
    df = pd.read_csv(f)
    dfs.append(df)

# =========================
# CONCAT
# =========================
era5_all = pd.concat(dfs, ignore_index=True)

# =========================
# STANDARDIZE
# =========================
era5_all["date"] = pd.to_datetime(era5_all["date"])
era5_all = era5_all.sort_values(["date", "grid_id"]).reset_index(drop=True)

# =========================
# EXPORT
# =========================
era5_all.to_csv(OUT_FILE, index=False)

print("✔ ERA5 merged:", OUT_FILE)
print("✔ Rows:", len(era5_all))
print("✔ Days:", era5_all["date"].nunique())
print("✔ Grids per day:", era5_all.groupby("date")["grid_id"].nunique().median())
