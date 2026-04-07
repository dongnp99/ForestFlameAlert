import pandas as pd
import numpy as np

DATE_VAL = '2024-01-14'
GRID_ID = 28069

# Check predictions first
pred = pd.read_parquet('app_admin_lookup.parquet')
print(pred.values)

#
# predv4 = pd.read_parquet('../v4/app_predictions_map_upgrade.parquet', columns=['grid_id','date','fire_prob','fire'])
# gv4 = predv4[predv4['date'] == DATE_VAL]
# rowv4 = gv4[gv4['fire'] == 1]
# print(rowv4.to_string())
#
# merged = rowv4.merge(pred, on=['grid_id','date'], suffixes=('_old','_new'))
# print(merged.to_string())