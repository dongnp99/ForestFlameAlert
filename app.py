import datetime
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
import xgboost as xgb
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MODEL_PATH  = BASE / "Project/xgboost/models/xgb_fire_after_tuned.json"
DATA_PATH   = BASE / "Project/data/Daklak/final_inputs/daklak_final_dataset.parquet"
COORDS_PATH = BASE / "Project/data/Daklak/final_inputs/raw_data/daklak_grid_lon_lat.csv"

FEATURE_COLS = [
    "tmean", "rh", "wind", "rain", "vpd",
    "rain_14d_sum", "rain_30d_sum", "vpd_14d_mean", "vpd_30d_mean",
    "fire_lag_1", "fire_lag_3",
    "dem_mean", "dem_stdev", "slp_mean", "slp_stdev",
    "sin_doy", "cos_doy",
    "neighbor_count", "neighbor_fire_1d", "neighbor_fire_3d", "neighbor_fire_7d",
    "vpd_neighbor_1d", "vpd_fire_lag_1",
    "ndvi", "ndvi_std", "ndwi", "nbr", "ndii", "delta_ndvi_14d", "delta_nbr_7d",
    "has_s2",
]

HIGH_RISK_THRESHOLD = 0.5
DAK_LAK_CENTER = [12.7, 108.0]


# ── Data loading & prediction (cached) ────────────────────────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load 2023 test set, run XGBoost, join grid coordinates."""
    df = pd.read_parquet(
        DATA_PATH,
        columns=["grid_id", "date"] + FEATURE_COLS + ["fire"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2023-12-31")].copy()

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    df["fire_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    coords = pd.read_csv(COORDS_PATH)
    df = df.merge(coords, on="grid_id", how="left")

    return df


# ── App layout ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ForestFlameAlert", page_icon="🔥", layout="wide")
st.title("🔥 ForestFlameAlert — Dak Lak Fire Probability (2023)")

with st.spinner("Loading model and running predictions on 2023 test set…"):
    pred_df = load_predictions()

# Date picker
selected_date = st.date_input(
    "Select date",
    value=datetime.date(2023, 1, 1),
    min_value=datetime.date(2023, 1, 1),
    max_value=datetime.date(2023, 12, 31),
)

day_df = pred_df[pred_df["date"] == pd.Timestamp(selected_date)]

if day_df.empty:
    st.warning("No data for selected date.")
    st.stop()

# ── Folium map ─────────────────────────────────────────────────────────────────
m = folium.Map(location=DAK_LAK_CENTER, zoom_start=8, tiles="CartoDB positron")

# Layer 1 — Fire Probability heatmap
prob_layer = folium.FeatureGroup(name="Fire Probability", show=True)
heat_data = day_df[["lat", "lon", "fire_prob"]].dropna().values.tolist()
HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=14,
    blur=10,
    max_zoom=13,
    gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
).add_to(prob_layer)
prob_layer.add_to(m)

# Layer 2 — Actual Fire (red dots)
fire_layer = folium.FeatureGroup(name="Actual Fire", show=True)
fire_grids = day_df[day_df["fire"] == 1][["grid_id", "lat", "lon"]].dropna()
for _, row in fire_grids.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.85,
        popup=f"Grid {int(row['grid_id'])}: fire confirmed",
        tooltip=f"Grid {int(row['grid_id'])}",
    ).add_to(fire_layer)
fire_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width="100%", height=600)

# ── Summary stats ──────────────────────────────────────────────────────────────
high_risk_count = int((day_df["fire_prob"] > HIGH_RISK_THRESHOLD).sum())
actual_fire_count = int((day_df["fire"] == 1).sum())
total_grids = len(day_df)

col1, col2, col3 = st.columns(3)
col1.metric("Total grids", total_grids)
col2.metric(f"High-risk grids (prob > {HIGH_RISK_THRESHOLD})", high_risk_count)
col3.metric("Actual fire grids", actual_fire_count)