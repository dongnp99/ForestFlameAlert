import pandas as pd
import pyarrow.parquet as pq
from Project.xgboost import xgb_config

PATH = "daklak_final_dataset_v2_human.parquet"

# --- Schema & file metadata (no full load needed) ---
pf = pq.read_metadata(PATH)
print(f"Rows: {pf.num_rows:,}  |  Row groups: {pf.num_row_groups}  |  Size: {pf.serialized_size / 1e6:.1f} MB")

# --- Load full dataframe ---
df = pq.read_table(
    PATH,
    columns=["date", "fire"] + xgb_config.FEATURE_COLS,
    filters=[("date", "<=", pd.Timestamp("2021-12-31"))]
)
