import geopandas as gpd
import pickle
from tqdm import tqdm

# =========================
# LOAD GRID SHAPEFILE
# =========================
GRID_SHP = "../meteorology/daklak_grid_1km_clip.shp"

print("Loading grid shapefile...")
gdf = gpd.read_file(GRID_SHP)

print("Total grids:", len(gdf))

# đảm bảo grid_id tồn tại
assert "id" in gdf.columns

# =========================
# SPATIAL INDEX (QUAN TRỌNG)
# =========================
print("Building spatial index...")
sindex = gdf.sindex

adjacency = {}

print("Computing adjacency...")
for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):

    grid_id = row["id"]
    geom = row.geometry

    # tìm candidate polygons bằng bbox
    possible_matches_index = list(sindex.intersection(geom.bounds))
    possible_matches = gdf.iloc[possible_matches_index]

    # giữ polygon chạm biên (touches)
    neighbors = possible_matches[
        possible_matches.geometry.touches(geom)
    ]

    neighbor_ids = neighbors["id"].tolist()

    adjacency[grid_id] = neighbor_ids

print("Adjacency computed.")

# =========================
# SAVE
# =========================
with open("grid_adjacency.pkl", "wb") as f:
    pickle.dump(adjacency, f)

print("Saved grid_adjacency.pkl")