# export_for_gee.py
import geopandas as gpd

gdf = gpd.read_parquet("grid_bounds.parquet")

print("CRS:", gdf.crs)
print("Columns:", gdf.columns.tolist())

# Export GeoJSON
gdf[["grid_id", "geometry"]].to_file(
    "grid_bounds.geojson",
    driver="GeoJSON"
)
print("Saved grid_bounds.geojson")

gdf[["grid_id", "geometry"]].to_file(
    "grid_bounds.shp"
)
print("Saved grid_bounds.shp")