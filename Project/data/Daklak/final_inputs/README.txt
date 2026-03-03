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