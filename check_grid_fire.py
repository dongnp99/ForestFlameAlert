import pandas as pd
import numpy as np

# Check predictions first
pred = pd.read_parquet('app_predictions.parquet')
g = pred[pred['grid_id'] == 28069]
print('=== app_predictions — grid_id 28069 ===')
row = g[g['date'] == '2024-01-14']
print(f'2024-01-14 row:')
print(row[['date','fire_prob','fire']].to_string())
print()

# Get raw features
DATA_PATH = 'Project/data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet'
df = pd.read_parquet(DATA_PATH, filters=[
  ('date', '>=', pd.Timestamp('2024-01-14')),
  ('date', '<=', pd.Timestamp('2024-01-14')),
])
row2 = df[df['grid_id'] == 29322]
print(f'Raw dataset rows found: {len(row2)}')
if len(row2) > 0:
  key_cols = ['date','fire','burn_season_flag','days_since_harvest','fire_count_prev_year',
              'deforestation_lag_1y','rain','rain_14d_sum','vpd','vpd_14d_mean',
              'neighbor_fire_1d','neighbor_fire_3d','neighbor_fire_7d',
              'fire_lag_1','fire_lag_3','ndvi','tree_cover_pct','lulc_class',
              'dist_road_km','nightlight_mean','pop_density','fire_freq_5y',
              'fire_count_prev_3y','days_since_last_fire']
  for col in key_cols:
      if col in row2.columns:
          print(f'  {col:<35} = {row2[col].values[0]}')