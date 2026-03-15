import numpy as np
import pandas as pd

from config import OUTPUT_DATA_PATH, GRID_ID_COL, DATE_COL


def main():

    input_file = OUTPUT_DATA_PATH + "/clstm_clean_data.parquet"
    output_file = OUTPUT_DATA_PATH + "/clstm_tensor.npy"

    print("Loading dataset...")
    df = pd.read_parquet(input_file)

    print("Dataset shape:", df.shape)

    feature_cols = [c for c in df.columns if c not in [GRID_ID_COL, DATE_COL]]
    C = len(feature_cols)
    print("Channels:", C)

    df = df.sort_values([DATE_COL, GRID_ID_COL])

    grid_ids = np.sort(df[GRID_ID_COL].unique())
    grid_count = len(grid_ids)

    print("Total grids:", grid_count)

    unique_dates = np.sort(df[DATE_COL].unique())
    T = len(unique_dates)

    print("Total days:", T)

    H = int(np.sqrt(grid_count))
    W = int(np.ceil(grid_count / H))

    print("Grid shape:", H, W)

    grid_index = {gid: i for i, gid in enumerate(grid_ids)}

    tensor = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype="float32",
        shape=(T, H, W, C)
    )

    grouped = df.groupby(DATE_COL)

    for t, (date, day_df) in enumerate(grouped):

        day_df = day_df.fillna(0)

        grid = np.zeros((H, W, C), dtype=np.float32)

        grid_idx = day_df[GRID_ID_COL].map(grid_index).to_numpy()

        rows = grid_idx // W
        cols = grid_idx % W

        features = day_df[feature_cols].to_numpy(dtype=np.float32)

        grid[rows, cols] = features

        tensor[t] = grid

        if t % 50 == 0:
            print(f"Processed {t}/{T}")

    print("Tensor saved:", output_file)
    print("Tensor shape:", tensor.shape)


if __name__ == "__main__":
    main()