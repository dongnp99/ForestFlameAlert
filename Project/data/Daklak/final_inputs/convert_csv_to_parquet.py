import pandas as pd

df = pd.read_csv(
    "dataset_fire_final.csv",
    parse_dates=["date"],
    dtype={
        "fire": "int8",
        "grid_id": "int32"
    }
)

df.to_parquet("dataset_fire_final.parquet", compression="snappy")