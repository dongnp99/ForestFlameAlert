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

Loading parquet data...
Loading train set...
Train size : 48,079,271  |  fire rate : 0.001219
scale_pos_weight : 819.51
Creating dtrain QuantileDMatrix...
Loading val set...
Val size : 6,863,095
Creating dval QuantileDMatrix...
Loading test set...
Test size : 13,744,993
Creating dtest QuantileDMatrix...
Training...
[0]	train-aucpr:0.01134	val-aucpr:0.00632
[100]	train-aucpr:0.03549	val-aucpr:0.01798
[200]	train-aucpr:0.04012	val-aucpr:0.01803
[235]	train-aucpr:0.04195	val-aucpr:0.01803
Evaluating...
=========================================
Best iteration   : 135
Validation AUC-PR: 0.018050
Test AUC-PR      : 0.016699
Validation ROC-AUC: 0.920276
Test ROC-AUC      : 0.906608
=========================================
Model saved.

2026-03-30 09:12:16,637 | INFO | Loading train set...
2026-03-30 09:12:27,607 | INFO | Full train size: 48079271
2026-03-30 09:12:47,477 | INFO | After sampling: 15000000
2026-03-30 09:12:47,528 | INFO | Train fire rate   : 0.001327
2026-03-30 09:12:47,529 | INFO | Base scale_pos_weight: 752.73
2026-03-30 09:12:47,529 | INFO | Creating dtrain QuantileDMatrix...
2026-03-30 09:12:58,093 | INFO | Loading val set...
2026-03-30 09:13:03,551 | INFO | Val size: 6863095
2026-03-30 09:13:03,556 | INFO | Creating dval QuantileDMatrix...
2026-03-30 09:13:05,850 | INFO | Starting Optuna study  (n_trials=40)...
[I 2026-03-30 09:13:05,855] A new study created in memory with name: no-name-30d79931-1e2e-4585-8a5a-52bb19a70390
[I 2026-03-30 09:14:35,414] Trial 0 finished with value: 0.01746424921661279 and parameters: {'max_depth': 8, 'min_child_weight': 15, 'learning_rate': 0.06257960621774133, 'subsample': 0.8095304694689628, 'colsample_bytree': 0.6546065241548528, 'gamma': 0.7799726016810132, 'reg_lambda': 1.0517943155978948, 'reg_alpha': 4.330880728874676, 'scale_pos_weight': 828.4237170569176}. Best is trial 0 with value: 0.01746424921661279.
[I 2026-03-30 09:15:50,355] Trial 1 finished with value: 0.01712492657357403 and parameters: {'max_depth': 10, 'min_child_weight': 2, 'learning_rate': 0.07804414039052963, 'subsample': 0.8913549242801475, 'colsample_bytree': 0.6743186887373966, 'gamma': 0.9091248360355031, 'reg_lambda': 2.2423428436076214, 'reg_alpha': 1.5212112147976886, 'scale_pos_weight': 799.6849830847107}. Best is trial 0 with value: 0.01746424921661279.
[I 2026-03-30 09:17:20,913] Trial 2 finished with value: 0.01767062443410654 and parameters: {'max_depth': 8, 'min_child_weight': 6, 'learning_rate': 0.05477043815695467, 'subsample': 0.6488228512282146, 'colsample_bytree': 0.7022506269873263, 'gamma': 1.8318092164684585, 'reg_lambda': 4.832664850061842, 'reg_alpha': 3.925879806965068, 'scale_pos_weight': 677.3350943278393}. Best is trial 2 with value: 0.01767062443410654.
[I 2026-03-30 09:21:33,132] Trial 3 finished with value: 0.01742662462798689 and parameters: {'max_depth': 9, 'min_child_weight': 10, 'learning_rate': 0.01801927682679985, 'subsample': 0.8126406981655034, 'colsample_bytree': 0.6596834432905521, 'gamma': 0.3252579649263976, 'reg_lambda': 9.514412603906665, 'reg_alpha': 4.828160165372797, 'scale_pos_weight': 906.4376339410596}. Best is trial 2 with value: 0.01767062443410654.
[I 2026-03-30 09:23:42,749] Trial 4 finished with value: 0.01737527072238867 and parameters: {'max_depth': 7, 'min_child_weight': 3, 'learning_rate': 0.0594751467232902, 'subsample': 0.7540533728088604, 'colsample_bytree': 0.6427133821956725, 'gamma': 2.475884550556351, 'reg_lambda': 0.8266909505945748, 'reg_alpha': 4.546602010393911, 'scale_pos_weight': 699.5806276967597}. Best is trial 2 with value: 0.01767062443410654.
[I 2026-03-30 09:25:24,472] Trial 5 finished with value: 0.01737087871655166 and parameters: {'max_depth': 9, 'min_child_weight': 6, 'learning_rate': 0.0488044213765577, 'subsample': 0.7913485977701479, 'colsample_bytree': 0.6646990594339345, 'gamma': 4.847923138822793, 'reg_lambda': 7.863761821930588, 'reg_alpha': 4.697494707820946, 'scale_pos_weight': 938.966903605695}. Best is trial 2 with value: 0.01767062443410654.
[I 2026-03-30 09:26:29,822] Trial 6 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:28:13,921] Trial 7 finished with value: 0.01751668526171801 and parameters: {'max_depth': 8, 'min_child_weight': 5, 'learning_rate': 0.05027524540528615, 'subsample': 0.6493234787411669, 'colsample_bytree': 0.8807689432639139, 'gamma': 0.3727532183988541, 'reg_lambda': 9.875425897704915, 'reg_alpha': 3.861223846483287, 'scale_pos_weight': 676.974498322556}. Best is trial 2 with value: 0.01767062443410654.
[I 2026-03-30 09:28:56,389] Trial 8 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:31:26,667] Trial 9 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:32:34,780] Trial 10 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:34:49,827] Trial 11 finished with value: 0.01778572106471288 and parameters: {'max_depth': 7, 'min_child_weight': 5, 'learning_rate': 0.04359669829433836, 'subsample': 0.6070144326848139, 'colsample_bytree': 0.9116890104541674, 'gamma': 1.8102612936962341, 'reg_lambda': 9.468755204097603, 'reg_alpha': 3.035921370166128, 'scale_pos_weight': 650.306674560127}. Best is trial 11 with value: 0.01778572106471288.
[I 2026-03-30 09:35:35,069] Trial 12 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:36:25,315] Trial 13 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:44:04,980] Trial 14 finished with value: 0.01812123141951962 and parameters: {'max_depth': 7, 'min_child_weight': 8, 'learning_rate': 0.07438416808613015, 'subsample': 0.7120250858271117, 'colsample_bytree': 0.7446949348924494, 'gamma': 1.4469269115194994, 'reg_lambda': 8.10865738525344, 'reg_alpha': 2.2558886159148375, 'scale_pos_weight': 657.7777816326251}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 09:47:19,533] Trial 15 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:53:06,376] Trial 16 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 09:59:10,131] Trial 17 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:00:02,518] Trial 18 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:00:44,902] Trial 19 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:03:10,578] Trial 20 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:19:52,078] Trial 21 finished with value: 0.017573990531462 and parameters: {'max_depth': 8, 'min_child_weight': 6, 'learning_rate': 0.053807903877563595, 'subsample': 0.6425689531245006, 'colsample_bytree': 0.7243440736783036, 'gamma': 1.6970197033074734, 'reg_lambda': 5.294785622904417, 'reg_alpha': 3.779945539902825, 'scale_pos_weight': 668.700411898486}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 10:33:12,492] Trial 22 finished with value: 0.01792758213262376 and parameters: {'max_depth': 7, 'min_child_weight': 7, 'learning_rate': 0.05581096110366511, 'subsample': 0.6690221087548878, 'colsample_bytree': 0.6993455073231635, 'gamma': 2.0470469116772296, 'reg_lambda': 2.8024260040352087, 'reg_alpha': 3.380609297508156, 'scale_pos_weight': 690.3895680719079}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 10:38:35,271] Trial 23 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:44:30,963] Trial 24 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:50:31,825] Trial 25 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 10:56:51,395] Trial 26 pruned. Trial was pruned at iteration 152.
[I 2026-03-30 11:09:45,632] Trial 27 finished with value: 0.01759152650322451 and parameters: {'max_depth': 8, 'min_child_weight': 5, 'learning_rate': 0.056352524780929225, 'subsample': 0.6601599212023345, 'colsample_bytree': 0.6005655543493141, 'gamma': 2.7773760420790294, 'reg_lambda': 7.9713837228676985, 'reg_alpha': 1.572649844172774, 'scale_pos_weight': 653.0629288168063}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 11:18:48,773] Trial 28 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:19:32,889] Trial 29 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:21:47,198] Trial 30 finished with value: 0.01792300845305491 and parameters: {'max_depth': 8, 'min_child_weight': 8, 'learning_rate': 0.0647241497986781, 'subsample': 0.8271368042633379, 'colsample_bytree': 0.9190922071550528, 'gamma': 2.1962974751681292, 'reg_lambda': 8.955178712860535, 'reg_alpha': 3.143831655780326, 'scale_pos_weight': 691.5589710154331}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 11:22:40,497] Trial 31 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:26:20,272] Trial 32 finished with value: 0.01787375892480704 and parameters: {'max_depth': 7, 'min_child_weight': 7, 'learning_rate': 0.059171030931431716, 'subsample': 0.7777923238083877, 'colsample_bytree': 0.8549518744652688, 'gamma': 1.0722455714131751, 'reg_lambda': 9.194018865680095, 'reg_alpha': 2.835559471385671, 'scale_pos_weight': 659.6677165847755}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 11:29:16,016] Trial 33 finished with value: 0.01757608009630061 and parameters: {'max_depth': 8, 'min_child_weight': 9, 'learning_rate': 0.05889566030723311, 'subsample': 0.781840766831416, 'colsample_bytree': 0.854987319373599, 'gamma': 0.8366665705765215, 'reg_lambda': 8.445274137976078, 'reg_alpha': 2.3258122969431363, 'scale_pos_weight': 716.7253375649188}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 11:33:02,951] Trial 34 finished with value: 0.01782524864110433 and parameters: {'max_depth': 8, 'min_child_weight': 7, 'learning_rate': 0.07431976144782816, 'subsample': 0.8169709766942318, 'colsample_bytree': 0.9075734712627255, 'gamma': 0.6285021420773734, 'reg_lambda': 8.822509306001615, 'reg_alpha': 2.7300166662908802, 'scale_pos_weight': 794.3341230740838}. Best is trial 14 with value: 0.01812123141951962.
[I 2026-03-30 11:34:30,858] Trial 35 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:41:59,318] Trial 36 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:47:43,216] Trial 37 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 11:53:23,898] Trial 38 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 12:00:53,618] Trial 39 pruned. Trial was pruned at iteration 150.
2026-03-30 12:00:54,065 | INFO | ====================================
2026-03-30 12:00:54,065 | INFO | Best AUC-PR : 0.018121
2026-03-30 12:00:54,065 | INFO | Best Params : {'max_depth': 7, 'min_child_weight': 8, 'learning_rate': 0.07438416808613015, 'subsample': 0.7120250858271117, 'colsample_bytree': 0.7446949348924494, 'gamma': 1.4469269115194994, 'reg_lambda': 8.10865738525344, 'reg_alpha': 2.2558886159148375, 'scale_pos_weight': 657.7777816326251}
2026-03-30 12:00:54,065 | INFO | ====================================


2026-03-30 12:39:46,845 | INFO | Loading data...
2026-03-30 12:39:46,845 | INFO | Loading train set...
2026-03-30 12:40:12,427 | INFO | Train size: 48079271  |  fire rate: 0.001219
2026-03-30 12:40:12,430 | INFO | Creating dtrain QuantileDMatrix...
2026-03-30 12:40:43,295 | INFO | Loading val set...
2026-03-30 12:40:49,495 | INFO | Val size: 6863095
2026-03-30 12:40:49,496 | INFO | Creating dval QuantileDMatrix...
2026-03-30 12:40:51,846 | INFO | Loading test set...
2026-03-30 12:40:58,886 | INFO | Test size: 13744993
2026-03-30 12:40:58,886 | INFO | Creating dtest QuantileDMatrix...
2026-03-30 12:41:03,129 | INFO | Training final tuned model...
[0]	train-aucpr:0.01698	val-aucpr:0.00947
[100]	train-aucpr:0.03567	val-aucpr:0.01786
[200]	train-aucpr:0.03893	val-aucpr:0.01902
[300]	train-aucpr:0.04131	val-aucpr:0.01985
[400]	train-aucpr:0.04366	val-aucpr:0.02015
[472]	train-aucpr:0.04556	val-aucpr:0.02018
2026-03-30 14:37:26,942 | INFO | Evaluating...
2026-03-30 14:37:48,012 | INFO | ===================================
2026-03-30 14:37:48,014 | INFO | Best iteration: 372
2026-03-30 14:37:48,014 | INFO | Validation AUC-PR: 0.020223
2026-03-30 14:37:48,014 | INFO | Test AUC-PR: 0.019941
2026-03-30 14:37:48,014 | INFO | Validation ROC-AUC: 0.928813
2026-03-30 14:37:48,014 | INFO | Test ROC-AUC: 0.917452
2026-03-30 14:37:48,014 | INFO | ===================================
2026-03-30 14:37:48,147 | INFO | Model saved to models/xgb_human_features_tuned.json

2026-03-30 14:39:53,872 | INFO | Loading XGBoost model from: models/xgb_human_features_tuned.json
2026-03-30 14:39:55,441 | INFO | Model loaded.
2026-03-30 14:39:55,441 | INFO | ==================================================
2026-03-30 14:39:55,441 | INFO | Processing split: train
2026-03-30 14:39:56,175 | INFO |   Total dates: 2557
2026-03-30 14:40:48,054 | INFO |   Chunks done: 10 / 86
2026-03-30 14:41:41,724 | INFO |   Chunks done: 20 / 86
2026-03-30 14:42:36,022 | INFO |   Chunks done: 30 / 86
2026-03-30 14:43:30,193 | INFO |   Chunks done: 40 / 86
2026-03-30 14:44:24,264 | INFO |   Chunks done: 50 / 86
2026-03-30 14:45:18,295 | INFO |   Chunks done: 60 / 86
2026-03-30 14:46:12,519 | INFO |   Chunks done: 70 / 86
2026-03-30 14:47:07,059 | INFO |   Chunks done: 80 / 86
2026-03-30 14:47:38,520 | INFO |   Chunks done: 86 / 86
2026-03-30 14:47:38,827 | INFO |   Total samples : 48079271
2026-03-30 14:47:38,855 | INFO |   Fire rate     : 0.001219
2026-03-30 14:48:11,703 | INFO |   AUC-PR        : 0.045575
2026-03-30 14:48:11,703 | INFO |   AUC-ROC       : 0.966976
2026-03-30 14:48:11,805 | INFO |   Pred range    : [0.0000, 0.9988]  mean=0.139102
2026-03-30 14:48:13,768 | INFO |   Saved → xgb_prob_map_train.npy  (48079271 samples)
2026-03-30 14:48:13,769 | INFO |   Saved → xgb_targets_train.npy
2026-03-30 14:48:13,769 | INFO |   Saved → xgb_meta_train.parquet
2026-03-30 14:48:13,843 | INFO | ==================================================
2026-03-30 14:48:13,843 | INFO | Processing split: val
2026-03-30 14:48:14,151 | INFO |   Total dates: 365
2026-03-30 14:49:07,719 | INFO |   Chunks done: 10 / 13
2026-03-30 14:49:22,727 | INFO |   Chunks done: 13 / 13
2026-03-30 14:49:22,785 | INFO |   Total samples : 6863095
2026-03-30 14:49:22,792 | INFO |   Fire rate     : 0.000828
2026-03-30 14:49:26,823 | INFO |   AUC-PR        : 0.020223
2026-03-30 14:49:26,823 | INFO |   AUC-ROC       : 0.928813
2026-03-30 14:49:26,830 | INFO |   Pred range    : [0.0000, 0.9932]  mean=0.107360
2026-03-30 14:49:27,149 | INFO |   Saved → xgb_prob_map_val.npy  (6863095 samples)
2026-03-30 14:49:27,149 | INFO |   Saved → xgb_targets_val.npy
2026-03-30 14:49:27,149 | INFO |   Saved → xgb_meta_val.parquet
2026-03-30 14:49:27,184 | INFO | ==================================================
2026-03-30 14:49:27,184 | INFO | Processing split: test
2026-03-30 14:49:27,515 | INFO |   Total dates: 731
2026-03-30 14:50:21,583 | INFO |   Chunks done: 10 / 25
2026-03-30 14:51:15,537 | INFO |   Chunks done: 20 / 25
2026-03-30 14:51:41,370 | INFO |   Chunks done: 25 / 25
2026-03-30 14:51:41,447 | INFO |   Total samples : 13744993
2026-03-30 14:51:41,454 | INFO |   Fire rate     : 0.001193
2026-03-30 14:51:50,008 | INFO |   AUC-PR        : 0.019941
2026-03-30 14:51:50,008 | INFO |   AUC-ROC       : 0.917452
2026-03-30 14:51:50,022 | INFO |   Pred range    : [0.0000, 0.9954]  mean=0.130470
2026-03-30 14:51:50,638 | INFO |   Saved → xgb_prob_map_test.npy  (13744993 samples)
2026-03-30 14:51:50,638 | INFO |   Saved → xgb_targets_test.npy
2026-03-30 14:51:50,638 | INFO |   Saved → xgb_meta_test.parquet
2026-03-30 14:51:50,679 | INFO | ==================================================
2026-03-30 14:51:50,680 | INFO | SUMMARY
2026-03-30 14:51:50,680 | INFO | ==================================================
2026-03-30 14:51:50,680 | INFO |   train  AUC-PR=0.0456  AUC-ROC=0.9670
2026-03-30 14:51:50,680 | INFO |   val    AUC-PR=0.0202  AUC-ROC=0.9288
2026-03-30 14:51:50,680 | INFO |   test   AUC-PR=0.0199  AUC-ROC=0.9175
2026-03-30 14:51:50,680 | INFO | Output dir: models\prob_maps
2026-03-30 14:51:50,680 | INFO | Files ready for fusion:
2026-03-30 14:51:50,680 | INFO |   xgb_prob_map_train.npy  /  xgb_targets_train.npy  /  xgb_meta_train.parquet
2026-03-30 14:51:50,680 | INFO |   xgb_prob_map_val.npy  /  xgb_targets_val.npy  /  xgb_meta_val.parquet
2026-03-30 14:51:50,680 | INFO |   xgb_prob_map_test.npy  /  xgb_targets_test.npy  /  xgb_meta_test.parquet

2026-03-30 23:39:20,897 | INFO | Loading train set...
2026-03-30 23:39:31,943 | INFO | Full train size: 48079271
2026-03-30 23:39:45,424 | INFO | After sampling: 15000000
2026-03-30 23:39:45,469 | INFO | Train fire rate   : 0.001327
2026-03-30 23:39:45,469 | INFO | Base scale_pos_weight: 752.73
2026-03-30 23:39:45,553 | INFO | Human-fire samples upweighted: 17711  (x3.0)
2026-03-30 23:39:45,553 | INFO | Creating dtrain QuantileDMatrix...
2026-03-30 23:39:55,990 | INFO | Loading val set...
2026-03-30 23:40:01,276 | INFO | Val size: 6863095
2026-03-30 23:40:01,279 | INFO | Creating dval QuantileDMatrix...
2026-03-30 23:40:03,557 | INFO | Starting Optuna study  (n_trials=40)...
[I 2026-03-30 23:40:03,560] A new study created in memory with name: no-name-89539fa2-1799-486e-9d3e-58114ad1fbd1
[I 2026-03-30 23:41:05,123] Trial 0 finished with value: 0.01635405434568505 and parameters: {'max_depth': 8, 'min_child_weight': 15, 'learning_rate': 0.06257960621774133, 'subsample': 0.8095304694689628, 'colsample_bytree': 0.6546065241548528, 'gamma': 0.7799726016810132, 'reg_lambda': 1.0517943155978948, 'reg_alpha': 4.330880728874676, 'scale_pos_weight': 828.4237170569176}. Best is trial 0 with value: 0.01635405434568505.
[I 2026-03-30 23:42:04,356] Trial 1 finished with value: 0.01514345130157776 and parameters: {'max_depth': 10, 'min_child_weight': 2, 'learning_rate': 0.07804414039052963, 'subsample': 0.8913549242801475, 'colsample_bytree': 0.6743186887373966, 'gamma': 0.9091248360355031, 'reg_lambda': 2.2423428436076214, 'reg_alpha': 1.5212112147976886, 'scale_pos_weight': 799.6849830847107}. Best is trial 0 with value: 0.01635405434568505.
[I 2026-03-30 23:43:50,632] Trial 2 finished with value: 0.01615503835514661 and parameters: {'max_depth': 8, 'min_child_weight': 6, 'learning_rate': 0.05477043815695467, 'subsample': 0.6488228512282146, 'colsample_bytree': 0.7022506269873263, 'gamma': 1.8318092164684585, 'reg_lambda': 4.832664850061842, 'reg_alpha': 3.925879806965068, 'scale_pos_weight': 677.3350943278393}. Best is trial 0 with value: 0.01635405434568505.
[I 2026-03-30 23:46:46,751] Trial 3 finished with value: 0.01641742740637897 and parameters: {'max_depth': 9, 'min_child_weight': 10, 'learning_rate': 0.01801927682679985, 'subsample': 0.8126406981655034, 'colsample_bytree': 0.6596834432905521, 'gamma': 0.3252579649263976, 'reg_lambda': 9.514412603906665, 'reg_alpha': 4.828160165372797, 'scale_pos_weight': 906.4376339410596}. Best is trial 3 with value: 0.01641742740637897.
[I 2026-03-30 23:48:04,925] Trial 4 finished with value: 0.01638769484275288 and parameters: {'max_depth': 7, 'min_child_weight': 3, 'learning_rate': 0.0594751467232902, 'subsample': 0.7540533728088604, 'colsample_bytree': 0.6427133821956725, 'gamma': 2.475884550556351, 'reg_lambda': 0.8266909505945748, 'reg_alpha': 4.546602010393911, 'scale_pos_weight': 699.5806276967597}. Best is trial 3 with value: 0.01641742740637897.
[I 2026-03-30 23:49:36,337] Trial 5 finished with value: 0.01629391744244639 and parameters: {'max_depth': 9, 'min_child_weight': 6, 'learning_rate': 0.0488044213765577, 'subsample': 0.7913485977701479, 'colsample_bytree': 0.6646990594339345, 'gamma': 4.847923138822793, 'reg_lambda': 7.863761821930588, 'reg_alpha': 4.697494707820946, 'scale_pos_weight': 938.966903605695}. Best is trial 3 with value: 0.01641742740637897.
[I 2026-03-30 23:50:39,641] Trial 6 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:52:36,923] Trial 7 finished with value: 0.01651719373629185 and parameters: {'max_depth': 8, 'min_child_weight': 5, 'learning_rate': 0.05027524540528615, 'subsample': 0.6493234787411669, 'colsample_bytree': 0.8807689432639139, 'gamma': 0.3727532183988541, 'reg_lambda': 9.875425897704915, 'reg_alpha': 3.861223846483287, 'scale_pos_weight': 676.974498322556}. Best is trial 7 with value: 0.01651719373629185.
[I 2026-03-30 23:53:17,360] Trial 8 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:56:08,758] Trial 9 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:57:16,117] Trial 10 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:58:04,130] Trial 11 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:59:07,486] Trial 12 pruned. Trial was pruned at iteration 150.
[I 2026-03-30 23:59:58,566] Trial 13 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:08:07,835] Trial 14 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:14:21,322] Trial 15 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:24:18,351] Trial 16 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:25:04,279] Trial 17 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:25:59,219] Trial 18 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:29:44,528] Trial 19 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:38:29,434] Trial 20 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 00:48:27,524] Trial 21 finished with value: 0.01635914556372182 and parameters: {'max_depth': 7, 'min_child_weight': 2, 'learning_rate': 0.05647960590172366, 'subsample': 0.7493678568695524, 'colsample_bytree': 0.6347536350843663, 'gamma': 3.5936976664612414, 'reg_lambda': 2.4274712946377957, 'reg_alpha': 4.282024248846114, 'scale_pos_weight': 705.0157049479873}. Best is trial 7 with value: 0.01651719373629185.
[I 2026-03-31 00:58:54,504] Trial 22 finished with value: 0.01648958802486694 and parameters: {'max_depth': 7, 'min_child_weight': 3, 'learning_rate': 0.059640077066115356, 'subsample': 0.8449990071962079, 'colsample_bytree': 0.6982895424021949, 'gamma': 4.425211803925006, 'reg_lambda': 5.653710906646679, 'reg_alpha': 4.584210507715884, 'scale_pos_weight': 700.228060066007}. Best is trial 7 with value: 0.01651719373629185.
[I 2026-03-31 01:04:05,486] Trial 23 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:09:59,288] Trial 24 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:14:40,444] Trial 25 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:22:02,123] Trial 26 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:22:59,875] Trial 27 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:23:55,200] Trial 28 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:27:40,576] Trial 29 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:34:57,135] Trial 30 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:40:30,560] Trial 31 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:45:33,390] Trial 32 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 01:51:00,473] Trial 33 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 02:00:57,279] Trial 34 finished with value: 0.01671072666014994 and parameters: {'max_depth': 8, 'min_child_weight': 2, 'learning_rate': 0.05741824143332552, 'subsample': 0.7705102323506257, 'colsample_bytree': 0.6733292800810863, 'gamma': 4.661230074006847, 'reg_lambda': 4.471351286348217, 'reg_alpha': 4.631961421397106, 'scale_pos_weight': 638.4840070541702}. Best is trial 34 with value: 0.01671072666014994.
[I 2026-03-31 02:06:55,946] Trial 35 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 02:13:39,884] Trial 36 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 02:14:37,814] Trial 37 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 02:15:42,327] Trial 38 pruned. Trial was pruned at iteration 150.
[I 2026-03-31 02:21:34,507] Trial 39 pruned. Trial was pruned at iteration 150.
2026-03-31 02:21:34,616 | INFO | ====================================
2026-03-31 02:21:34,617 | INFO | Best AUC-PR : 0.016711
2026-03-31 02:21:34,617 | INFO | Best Params : {'max_depth': 8, 'min_child_weight': 2, 'learning_rate': 0.05741824143332552, 'subsample': 0.7705102323506257, 'colsample_bytree': 0.6733292800810863, 'gamma': 4.661230074006847, 'reg_lambda': 4.471351286348217, 'reg_alpha': 4.631961421397106, 'scale_pos_weight': 638.4840070541702}
2026-03-31 02:21:34,617 | INFO | ====================================

2026-03-31 08:18:20,939 | INFO | Loading data...
2026-03-31 08:18:20,939 | INFO | Loading train set...
2026-03-31 08:18:48,887 | INFO | Train size: 48079271  |  fire rate: 0.001219
2026-03-31 08:18:49,183 | INFO | Human-fire samples upweighted: 52816  (x3.0)
2026-03-31 08:18:49,184 | INFO | Creating dtrain QuantileDMatrix...
2026-03-31 08:19:20,102 | INFO | Loading val set...
2026-03-31 08:19:25,153 | INFO | Val size: 6863095
2026-03-31 08:19:25,153 | INFO | Creating dval QuantileDMatrix...
2026-03-31 08:19:27,278 | INFO | Loading test set...
2026-03-31 08:19:33,713 | INFO | Test size: 13744993
2026-03-31 08:19:33,713 | INFO | Creating dtest QuantileDMatrix...
2026-03-31 08:19:38,048 | INFO | Training final tuned model...
[0]	train-aucpr:0.04773	val-aucpr:0.01059
[100]	train-aucpr:0.09959	val-aucpr:0.01651
[200]	train-aucpr:0.11016	val-aucpr:0.01773
[300]	train-aucpr:0.11601	val-aucpr:0.01823
[400]	train-aucpr:0.12149	val-aucpr:0.01841
[500]	train-aucpr:0.12805	val-aucpr:0.01862
[600]	train-aucpr:0.13396	val-aucpr:0.01871
[700]	train-aucpr:0.14058	val-aucpr:0.01883
[800]	train-aucpr:0.14718	val-aucpr:0.01892
[900]	train-aucpr:0.15396	val-aucpr:0.01895
[955]	train-aucpr:0.15806	val-aucpr:0.01879
2026-03-31 12:33:10,126 | INFO | Evaluating...
 2026-03-31 12:33:57,456 | INFO | ===================================
2026-03-31 12:33:57,457 | INFO | Best iteration: 855
2026-03-31 12:33:57,457 | INFO | Validation AUC-PR: 0.018850
2026-03-31 12:33:57,457 | INFO | Test AUC-PR: 0.016679
2026-03-31 12:33:57,457 | INFO | Validation ROC-AUC: 0.918224
2026-03-31 12:33:57,457 | INFO | Test ROC-AUC: 0.905348
2026-03-31 12:33:57,457 | INFO | ===================================
2026-03-31 12:33:57,799 | INFO | Model saved to models/xgb_human_features_tuned_v2.json