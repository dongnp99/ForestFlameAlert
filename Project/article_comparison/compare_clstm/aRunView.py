from pathlib import Path

import pandas as pd

HERE        = Path(__file__).parent
ROOT        = HERE.parent
XGB_PRED    = ROOT / "xgboost/models/v5/app_predictions_map.parquet"
df = pd.read_parquet(
        XGB_PRED,
        engine="pyarrow",
    )
print(df.columns)