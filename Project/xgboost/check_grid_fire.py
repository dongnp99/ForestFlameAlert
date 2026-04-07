import pandas as pd
import numpy as np

DATE_VAL = '2024-01-14'
GRID_ID = 28069

# Check predictions first
pred = pd.read_parquet('models/v3/app_predictions_map.parquet')
g = pred[pred['grid_id'] == GRID_ID]
row = g[g['date'] == DATE_VAL]
print(row.to_string())

pred_new = pd.read_parquet('models/v4/app_predictions_map_upgrade.parquet')
g_new = pred_new[pred_new['grid_id'] == GRID_ID]
row_new = g_new[g_new['date'] == DATE_VAL]
print(row_new.to_string())

print()

# Get raw features
DATA_PATH = '../data/Daklak/final_inputs/daklak_final_dataset_v2_human.parquet'
df = pd.read_parquet(DATA_PATH, filters=[
  ('date', '>=', pd.Timestamp('2024-04-08')),
  ('date', '<=', pd.Timestamp('2024-04-08')),
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