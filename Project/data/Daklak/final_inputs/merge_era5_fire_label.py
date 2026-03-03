import pandas as pd

# Load fire label (nhỏ → load 1 lần)
fire = pd.read_csv(
    "daklak_firms.csv",
    parse_dates=["date"]
)

fire["fire"] = 1

# Dùng set để lookup nhanh
fire_key = set(zip(fire.grid_id, fire.date))

chunksize = 500_000  # có thể tăng/giảm tùy RAM

out_file = "daklak_era5_firms.csv"
first = True

print(f"Processing...")

for chunk in pd.read_csv(
    "daklak_era5.csv",
    chunksize=chunksize,
    parse_dates=["date"]
):
    
    print(f"Chunking...")
    
    # gán fire = 1 nếu tồn tại trong FIRMS
    chunk["fire"] = [
        1 if (gid, d) in fire_key else 0
        for gid, d in zip(chunk.grid_id, chunk.date)
    ]

    chunk.to_csv(
        out_file,
        mode="w" if first else "a",
        header=first,
        index=False
    )
    first = False
    
    print(f"Chunking complete!")
