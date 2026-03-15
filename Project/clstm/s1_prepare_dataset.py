# scripts/01_prepare_dataset.py

import pandas as pd
from config import (
    RAW_DATA_PATH,
    OUTPUT_DATA_PATH,
    FEATURE_COLUMNS,
    GRID_ID_COL,
    DATE_COL
)

def main():
    print("Loading dataset...")
    df = pd.read_parquet(RAW_DATA_PATH)

    print("Original shape:", df.shape)

    df = df[FEATURE_COLUMNS]

    print("OK")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df = df.sort_values([DATE_COL, GRID_ID_COL]).reset_index(drop=True)

    print("Dataset sorted.")

    grid_counts = df.groupby(DATE_COL)[GRID_ID_COL].nunique()

    print("Grid per day:")
    print("Min:", grid_counts.min())
    print("Max:", grid_counts.max())

    missing_values = df.isna().sum().sum()
    print("Missing values:", missing_values)

    df.to_parquet(OUTPUT_DATA_PATH + '/clstm_clean_data.parquet', index=False)

    print("Saved cleaned dataset:", OUTPUT_DATA_PATH + '/clstm_clean_data.parquet')

if __name__ == "__main__":
    main()