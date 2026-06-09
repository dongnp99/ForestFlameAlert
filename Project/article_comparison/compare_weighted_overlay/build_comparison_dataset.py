"""
Tạo dataset cho bài báo so sánh
Giai đoạn: 2023–2024.
Features đầu ra
---------------
Động (thay đổi theo ngày):
    tmean           - Nhiệt độ trung bình ngày (°C)
    rain            - Lượng mưa ngày (mm)
    wind            - Tốc độ gió ngày (m/s)

Tĩnh (bất biến theo thời gian):
    lon, lat            - Tọa độ tâm grid (WGS84)
    lulc_class          - Kiểu thảm thực vật (ESA WorldCover):
                            10=Rừng cây,  30=Cây bụi,  40=Đất canh tác,
                            50=Khu dân cư, 60=Đất trống, 80=Mặt nước
    tree_cover_pct      - Độ che phủ tán rừng năm 2000 (%)
    cropland_frac_1km   - Tỷ lệ đất canh tác trong ô 1km²
    slope_mean          - Độ dốc trung bình (°)
    slope_std           - Độ lệch chuẩn độ dốc
    aspect_sin          - sin(hướng sườn) — biểu diễn circular
    aspect_cos          - cos(hướng sườn)
    aspect_deg          - Hướng sườn theo độ (0–360°)
    dist_settlement_km  - Khoảng cách từ khu dân cư đến ô rừng (km)
    dist_forest_edge_km - Khoảng cách từ ô đến rìa rừng gần nhất (km)
                          (≈ khoảng cách từ đất canh tác/nương rẫy đến rừng)

Đầu ra:
    article_comparison/daklak_comparison_dataset.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data" / "Daklak"
RAW_DIR  = DATA_DIR / "final_inputs" / "raw_data"
FEAT_DIR = RAW_DIR / "human_features" / "features"

OUT_PARQUET = HERE / "daklak_comparison_dataset.parquet"
GRID_CSV    = RAW_DIR / "daklak_grid_lon_lat.csv"

YEARS = [2023, 2024]
DRY_MONTHS = [11, 12, 1, 2, 3, 4]  # mùa khô: tháng 11 → tháng 4

# ESA WorldCover class labels (để tham khảo)
LULC_LABELS = {
    10: "Tree cover (Rừng cây)",
    20: "Shrubland (Cây bụi thưa)",
    30: "Grassland (Đồng cỏ)",
    40: "Cropland (Đất canh tác)",
    50: "Built-up (Khu dân cư)",
    60: "Bare/sparse vegetation (Đất trống)",
    70: "Snow and ice",
    80: "Permanent water (Mặt nước)",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


# ---------------------------------------------------------------------------
# 1. ERA5 — lọc 2023-2024, giữ theo ngày
# ---------------------------------------------------------------------------
def load_era5_daily(years: list[int]) -> pd.DataFrame:
    print("[1/4] Đọc ERA5 (daily)...")
    era5 = pd.read_csv(
        RAW_DIR / "daklak_era5.csv",
        usecols=["grid_id", "date", "tmean", "wind", "rain"],
        parse_dates=["date"],
    )
    # Lọc theo năm và chỉ giữ mùa khô (tháng 11 → tháng 4)
    # Mùa khô vắt qua 2 năm: Nov-Dec 2022, Jan-Apr 2023, Nov-Dec 2023, Jan-Apr 2024
    # → mở rộng lấy thêm năm trước để có Nov-Dec 2022
    all_years = sorted(set(years) | {min(years) - 1})
    era5 = era5[era5["date"].dt.year.isin(all_years)].copy()
    era5 = era5[era5["date"].dt.month.isin(DRY_MONTHS)].copy()
    era5 = era5.sort_values(["grid_id", "date"]).reset_index(drop=True)
    print(f"      ERA5 mùa khô: {len(era5):,} rows  ({era5['date'].min().date()} – {era5['date'].max().date()})")
    return era5


# ---------------------------------------------------------------------------
# 2. Static features
# ---------------------------------------------------------------------------
def load_static() -> pd.DataFrame:
    print("[2/4] Đọc các feature tĩnh...")

    # Lon/lat từ daklak_grid_lon_lat.csv
    grid_ll = pd.read_csv(GRID_CSV)[["grid_id", "lon", "lat"]]
    print(f"      Grid lon/lat: {len(grid_ll):,} grids")

    # Slope & aspect từ daklak_dem.csv
    dem = pd.read_csv(RAW_DIR / "daklak_dem.csv")[
        ["grid_id", "slp_mean", "slp_stdev", "aspect_sin", "aspect_cos"]
    ].rename(columns={"slp_mean": "slope_mean", "slp_stdev": "slope_std"})
    print(f"      DEM: {len(dem):,} grids")

    # LULC (kiểu thảm thực vật + cropland fraction + tree cover)
    lulc = pd.read_parquet(FEAT_DIR / "lulc_hansen.parquet")[
        ["grid_id", "lulc_class", "tree_cover_pct", "cropland_frac_1km"]
    ]
    print(f"      LULC: {len(lulc):,} grids")

    # OSM distances (khoảng cách khu dân cư, rìa rừng)
    osm = pd.read_parquet(FEAT_DIR / "osm_distances.parquet")[
        ["grid_id", "dist_settlement_km", "dist_forest_edge_km"]
    ]
    print(f"      OSM distances: {len(osm):,} grids")

    # Merge tất cả theo grid_id
    static = grid_ll.merge(dem,  on="grid_id", how="inner") \
                    .merge(lulc, on="grid_id", how="inner") \
                    .merge(osm,  on="grid_id", how="left")

    print(f"      Static tổng hợp: {len(static):,} grids, cols={static.columns.tolist()}")
    return static


# ---------------------------------------------------------------------------
# 3. Fire label từ FIRMS
# ---------------------------------------------------------------------------
def load_fire_labels() -> pd.DataFrame:
    print("[3/4] Đọc fire labels từ FIRMS...")
    firms = pd.read_csv(
        RAW_DIR / "daklak_firms.csv",
        usecols=["grid_id", "date", "fire"],
        parse_dates=["date"],
    )
    # FIRMS chỉ chứa các ngày có cháy (fire=1); merge left sẽ điền NaN → 0
    print(f"      FIRMS: {len(firms):,} fire records")
    return firms


# ---------------------------------------------------------------------------
# 4. Ghép dynamic × static × fire label
# ---------------------------------------------------------------------------
def build_dataset(daily: pd.DataFrame, static: pd.DataFrame, firms: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Ghép daily ERA5 + static + fire label...")
    df = daily.merge(static, on="grid_id", how="left")
    df = df.merge(firms, on=["grid_id", "date"], how="left")
    df["fire"] = df["fire"].fillna(0).astype("int8")

    # Hướng sườn theo độ (0–360°)
    df["aspect_deg"] = np.degrees(np.arctan2(df["aspect_sin"], df["aspect_cos"])) % 360

    # Sắp xếp cột theo logic
    cols = [
        "grid_id", "lon", "lat", "date",
        # Label
        "fire",
        # Khí hậu
        "tmean", "rain", "wind",
        # Thảm thực vật rừng
        "lulc_class", "tree_cover_pct", "cropland_frac_1km",
        # Địa hình
        "slope_mean", "slope_std", "aspect_sin", "aspect_cos", "aspect_deg",
        # Khoảng cách nhân sinh
        "dist_settlement_km", "dist_forest_edge_km",
    ]
    df = df[cols].sort_values(["grid_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 5. Lưu
# ---------------------------------------------------------------------------
def main() -> None:
    daily  = load_era5_daily(YEARS)
    static = load_static()
    firms  = load_fire_labels()
    df     = build_dataset(daily, static, firms)

    fire_rate = df["fire"].mean() * 100
    print(f"\n[5/5] Lưu dataset: {df.shape[0]:,} rows × {df.shape[1]} cols  (fire rate: {fire_rate:.3f}%)")

    df.to_parquet(OUT_PARQUET, index=False)

    print(f"  -> {OUT_PARQUET.name}")
    print()


if __name__ == "__main__":
    main()