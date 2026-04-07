from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
PRED_PATH    = BASE / "app_predictions.parquet"

HIGH_RISK_THRESHOLD = 0.5
DAK_LAK_CENTER      = [12.7, 108.0]

# ── Load precomputed predictions ───────────────────────────────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PRED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── App layout ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ForestFlameAlert", page_icon="🔥", layout="wide")
st.title("🔥 ForestFlameAlert — Dak Lak Fire Probability (2023–2024)")

if not PRED_PATH.exists():
    st.error(
        f"Predictions file not found: `{PRED_PATH}`\n\n"
        "Run `python precompute_predictions.py` first to generate it."
    )
    st.stop()

pred_df = load_predictions()

min_date = pred_df["date"].min().date()
max_date = pred_df["date"].max().date()

# ── Controls ───────────────────────────────────────────────────────────────────
selected_date = st.date_input(
    "Select date",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
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
high_risk_count   = int((day_df["fire_prob"] > HIGH_RISK_THRESHOLD).sum())
actual_fire_count = int((day_df["fire"] == 1).sum())
total_grids       = len(day_df)

col1, col2, col3 = st.columns(3)
col1.metric("Total grids", total_grids)
col2.metric(f"High-risk grids (prob > {HIGH_RISK_THRESHOLD})", high_risk_count)
col3.metric("Actual fire grids", actual_fire_count)