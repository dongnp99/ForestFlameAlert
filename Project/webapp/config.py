"""
config.py — Path and runtime configuration for the FireWatch backend.
Edit this file to point to different model versions or data locations.
"""

from pathlib import Path

#  Root directories ─
WEBAPP_DIR  = Path(__file__).parent                     # Project/webapp/
PROJECT_DIR = WEBAPP_DIR.parent                         # Project/
ROOT_DIR    = PROJECT_DIR.parent                        # ForestFlameAlert/

#  Active model version ─
MODEL_VERSION = "v5"
MODEL_DIR     = PROJECT_DIR / "xgboost" / "models" / MODEL_VERSION

#  Data files ─
MAP_PATH     = MODEL_DIR / "app_predictions_map.parquet"
DETAIL_PATH  = MODEL_DIR / "app_predictions_detail.parquet"
SUMMARY_PATH = MODEL_DIR / "app_fire_type_summary.csv"

#  Grid geometry 
CELL_HALF = 0.004514   # half-size in degrees ≈ 500 m

#  Server ─
HOST = "0.0.0.0"
PORT = 8000