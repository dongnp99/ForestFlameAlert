# Data validate 1: read, get head and tail with all column
import pandas as pd

df = pd.read_csv("daklak_fire_xgb_additional_features.csv", nrows=500_000)

# test = df[df["grid_id"] == 75].copy()
# print(df[["date", "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d"]].head(10))
# print(
#     df.loc[df["neighbor_count"] > 7,
#              ["date", "grid_id",
#               "neighbor_count",
#               "neighbor_fire_1d",]]
#     .head(10)
# )
# print(df.head())
# print(df.tail())
print(df.columns.tolist())
print(df["fire"].value_counts())

##Data validate 2: check for total fire label
# import pandas as pd
#
# df = pd.read_csv("dataset_fire_final.csv", parse_dates=["date"])
#
# print(df.shape)
# print(df["fire"].value_counts())
# print(df.isna().sum())


# #Data validate 3: Check adjacency grid
# import pickle
#
# with open("grid_adjacency.pkl", "rb") as f:
#     adj = pickle.load(f)
#
# print("Total grid:", adj.keys().__len__())
# print("Example grid:", list(adj.keys())[12000])
# print("Neighbors:", adj[list(adj.keys())[12000]])