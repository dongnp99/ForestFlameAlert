import pandas as pd

df = pd.read_parquet(
    "daklak_final_dataset.parquet"
)
print(df.columns.tolist())
print(df.tail(3))
print("grid_id range:", df["grid_id"].min(), "→", df["grid_id"].max())
print("Số grid unique:", df["grid_id"].nunique())