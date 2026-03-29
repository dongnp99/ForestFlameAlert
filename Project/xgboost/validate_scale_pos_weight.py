import pandas as pd

df = pd.read_parquet(
    "../data/Daklak/final_inputs/daklak_final_dataset.parquet"
)
df["year"] = df["date"].dt.year

# Fire rate theo từng split mới
splits = {
    "Train (2015–2021)": (df["date"] <= "2021-12-31"),
    "Val   (2022)":      (df["date"].between("2022-01-01", "2022-12-31")),
    "Test  (2023–2024)": (df["date"] >= "2023-01-01"),
}

print(f"{'Split':<22} {'Rows':>10} {'Fire':>8} {'Rate':>8}")
print("-" * 52)
for name, mask in splits.items():
    sub = df[mask]
    print(f"{name:<22} {len(sub):>10,} "
          f"{sub['fire'].sum():>8,} "
          f"{sub['fire'].mean()*100:>7.3f}%")

# scale_pos_weight mới cho XGBoost
train = df[df["date"] <= "2021-12-31"]
neg = (train["fire"] == 0).sum()
pos = (train["fire"] == 1).sum()
print(f"\nscale_pos_weight mới: {neg/pos:.1f}")