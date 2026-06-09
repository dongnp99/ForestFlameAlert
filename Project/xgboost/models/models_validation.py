
# Model validate 1: check importance features
import xgboost as xgb

MODEL_PATH = "v5/xgb_human_features_tuned_v5_pathways.json"

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

importance = booster.get_score(importance_type="gain")

# sort giảm dần
importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

print("Top 45 features by gain:")
for k, v in list(importance.items())[:45]:
    print(k, v)