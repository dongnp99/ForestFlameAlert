import pandas as pd

df = pd.read_parquet("clstm_clean_data.parquet")

feature_cols = [c for c in df.columns if c not in ["grid_id", "date"]]

for i, f in enumerate(feature_cols):
    print(i, f)