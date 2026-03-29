File dữ liệu:
- daklak_era5.csv : Dữ liệu khí tượng từ ERA5
- daklak_dem.csv : Dữ liệu địa hình từ DEM
- daklak_firms.csv : Dữ liệu nhãn cháy từ FIRMS

File dữ liệu đã xử lý:
- daklak_era5_firms.csv : Dữ liệu merge ERA5 với FIRMS - có khí tượng và gán nhãn cháy
- dataset_fire_final.csv : Dữ liệu đã merge ERA5+DEM+FIRMS - có khí tượng, địa hình và gán nhãn cháy

Script xử lý dữ liệu
- merge_era5_DEM_fire_label.py : Script merge dữ liệu khí tượng, địa hình và gán nhãn cháy
- merge_era5_fire_label.py : Script merge dữ liệu khí tượng và gán nhãn cháy
- validate_data_tool.py : Script kiểm tra dữ liệu


Loading parquet data...
Loading train set...
Loading val set...
Loading test set...
Train size: 48079271
Val size: 6863095
Test size: 13744993
Train fire rate: 0.0012187580797554105
scale_pos_weight: 819.5073809239381
Creating QuantileDMatrix...
Training...
[0]	train-aucpr:0.01355	val-aucpr:0.00699
[100]	train-aucpr:0.03098	val-aucpr:0.01463
[200]	train-aucpr:0.03513	val-aucpr:0.01454
[252]	train-aucpr:0.03673	val-aucpr:0.01430
Evaluating...
===================================
Best iteration: 152
Validation AUC-PR: 0.014319053089918694
Test AUC-PR: 0.014980969108303405
Validation ROC-AUC: 0.8895781692500795
Test ROC-AUC: 0.8867595958428869
===================================
Model saved.