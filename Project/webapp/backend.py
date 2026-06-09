"""
backend.py — FastAPI server for Forest Fire webapp.

Data files and paths are configured in config.py.

Endpoints:
  GET /api/health
  GET /api/dates
  GET /api/prob-map?date=YYYY-MM-DD
  GET /api/grid-detail?grid_id=<int>&date=YYYY-MM-DD
  GET /api/fire-type-summary?date=YYYY-MM-DD

Run:
  pip install fastapi uvicorn[standard] scipy
  python backend.py
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import KDTree

from config import MAP_PATH, DETAIL_PATH, SUMMARY_PATH, CELL_HALF, HOST, PORT

# LULC class code → Vietnamese label
LULC_LABELS: dict[int, str] = {
    10: "Rừng",
    30: "Thảm cỏ",
    40: "Đất nông nghiệp",
    50: "Khu dân cư",
    60: "Đất trống",
    80: "Mặt nước",
}

#  Load map data 
print("Loading map file …", flush=True)
_map_raw = pd.read_parquet(MAP_PATH)
_map_raw["date"] = pd.to_datetime(_map_raw["date"])

_by_date: dict[str, pd.DataFrame] = {
    str(d.date()): g.reset_index(drop=True)
    for d, g in _map_raw.groupby("date")
}

_grid_meta = (
    _map_raw.drop_duplicates("grid_id")[["grid_id", "lat", "lon"]]
    .set_index("grid_id")
)
_all_grid_ids = _grid_meta.index.to_numpy()
_grid_coords  = _grid_meta[["lat", "lon"]].to_numpy()
_kd_tree      = KDTree(_grid_coords)

MIN_DATE = str(_map_raw["date"].min().date())
MAX_DATE = str(_map_raw["date"].max().date())
del _map_raw
print(f"Map ready: {len(_by_date)} dates, {len(_grid_meta)} grids "
      f"({MIN_DATE} → {MAX_DATE})", flush=True)

#  Load detail data ─
print("Loading detail file …", flush=True)
_detail_raw = pd.read_parquet(DETAIL_PATH)
_detail_raw["date"] = pd.to_datetime(_detail_raw["date"])

_by_grid: dict[int, pd.DataFrame] = {
    int(gid): g.reset_index(drop=True)
    for gid, g in _detail_raw.groupby("grid_id")
}
del _detail_raw
print(f"Detail ready: {len(_by_grid)} grids.", flush=True)

#  Load fire type summary ─
print("Loading fire type summary …", flush=True)
_summary_raw = pd.read_csv(SUMMARY_PATH)
_summary_raw["date"] = pd.to_datetime(_summary_raw["date"]).dt.date.astype(str)
_summary_by_date: dict[str, list[dict]] = {
    date: grp[["fire_subtype", "count"]].to_dict("records")
    for date, grp in _summary_raw.groupby("date")
}
del _summary_raw
print(f"Summary ready: {len(_summary_by_date)} dates.", flush=True)

#  Helpers 

def _get_day(date_str: str) -> pd.DataFrame:
    df = _by_date.get(date_str)
    if df is None:
        raise HTTPException(404, f"No data for date {date_str!r}. "
                            f"Available range: {MIN_DATE} → {MAX_DATE}")
    return df


def _prev_date_str(date_str: str) -> str:
    return str((pd.Timestamp(date_str) - timedelta(days=1)).date())


def _make_rect(lat: float, lon: float) -> dict[str, Any]:
    """GeoJSON Polygon rectangle centred at (lat, lon)."""
    h = CELL_HALF
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - h, lat - h],
            [lon + h, lat - h],
            [lon + h, lat + h],
            [lon - h, lat + h],
            [lon - h, lat - h],
        ]],
    }


def _find_neighbors(lat: float, lon: float, exclude_id: int, n: int = 8) -> list[int]:
    _, indices = _kd_tree.query([lat, lon], k=n + 1)
    result = [int(_all_grid_ids[i]) for i in indices if int(_all_grid_ids[i]) != exclude_id]
    return result[:n]


def _shap_to_dict(row: pd.Series) -> dict[str, float] | None:
    cols = ["shap_human", "shap_natural"]
    if not all(c in row.index for c in cols):
        return None
    return {
        "human":   round(float(row["shap_human"]),   1),
        "natural": round(float(row["shap_natural"]), 1),
    }


def _human_signals(row: pd.Series) -> dict[str, Any] | None:
    needed = ["burn_season_flag", "fire_freq_5y", "dist_road_km",
              "lulc_class", "dist_settlement_km"]
    if not all(c in row.index for c in needed):
        return None
    lulc_code = int(row["lulc_class"])
    dist_km   = float(row["dist_settlement_km"])
    return {
        "burning_season":     bool(row["burn_season_flag"]),
        "fire_count_5yr":     round(float(row["fire_freq_5y"]), 2),
        "dist_road_km":       round(float(row["dist_road_km"]), 2),
        "land_use":           LULC_LABELS.get(lulc_code, f"Mã {lulc_code}"),
        "nearest_settlement": f"cách {dist_km:.1f} km",
    }


def _weather_vals(row: pd.Series) -> dict[str, float] | None:
    needed = ["tmean", "rh", "vpd", "rain_14d_sum", "vpd_14d_mean"]
    if not all(c in row.index for c in needed):
        return None
    return {
        "tmean":         round(float(row["tmean"]),        1),
        "rh":            round(float(row["rh"]),           1),
        "vpd":           round(float(row["vpd"]),          3),
        "rain_14d_sum":  round(float(row["rain_14d_sum"]), 1),
        "vpd_14d_mean":  round(float(row["vpd_14d_mean"]), 3),
    }


def _fire_classification(row: pd.Series) -> dict[str, Any] | None:
    needed = ["fire_type", "fire_subtype", "confidence_score", "matched_pathways"]
    if not all(c in row.index for c in needed):
        return None
    return {
        "fire_type":        str(row["fire_type"]),
        "fire_subtype":     str(row["fire_subtype"]),
        "confidence_score": round(float(row["confidence_score"]), 3),
        "matched_pathways": str(row["matched_pathways"]),
    }


#  FastAPI app 
app = FastAPI(title="FireWatch Đắk Lắk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dates":  len(_by_date),
        "grids":  len(_grid_meta),
        "range":  f"{MIN_DATE} → {MAX_DATE}",
    }


@app.get("/api/dates")
def get_dates():
    return {"min_date": MIN_DATE, "max_date": MAX_DATE}


@app.get("/api/prob-map")
def get_prob_map(date: str = Query(..., description="YYYY-MM-DD")):
    """
    GeoJSON FeatureCollection of all grid cells for the given date.
    Each feature includes: grid_id, prob, prob_yesterday, fire,
    dominant_factor, center_lat, center_lon.
    """
    today_df = _get_day(date)
    prev_df  = _by_date.get(_prev_date_str(date))

    prev_prob: dict[int, float] = (
        dict(zip(prev_df["grid_id"].values, prev_df["fire_prob"].values.astype(float)))
        if prev_df is not None else {}
    )

    features = []
    for row in today_df.itertuples(index=False):
        gid  = int(row.grid_id)
        prob = float(row.fire_prob)
        dom  = str(row.dominant_factor) if hasattr(row, "dominant_factor") else "N/A"
        features.append({
            "type": "Feature",
            "geometry": _make_rect(row.lat, row.lon),
            "properties": {
                "grid_id":         gid,
                "prob":            round(prob, 4),
                "prob_yesterday":  round(prev_prob.get(gid, prob), 4),
                "fire":            int(row.fire),
                "dominant_factor": dom,
                "center_lat":      round(row.lat, 6),
                "center_lon":      round(row.lon, 6),
            },
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/grid-detail")
def get_grid_detail(
    grid_id: int = Query(...),
    date: str    = Query(..., description="YYYY-MM-DD"),
):
    """
    Full sidebar data for one grid on one date:
      - prob, prob_yesterday, fire
      - fire_classification (fire_type, fire_subtype, confidence_score, matched_pathways)
      - shap_values (% per group)
      - human_signals
      - weather_vals
      - history_30d  (last 30 days)
      - fire_events_12m (fire=1 rows in last 12 months)
      - neighbors (8 nearest grids with current probs)
    """
    if grid_id not in _grid_meta.index:
        raise HTTPException(404, f"grid_id {grid_id} not found")

    meta = _grid_meta.loc[grid_id]
    lat, lon = float(meta["lat"]), float(meta["lon"])

    today_map = _get_day(date)
    row_map   = today_map[today_map["grid_id"] == grid_id]
    if row_map.empty:
        raise HTTPException(404, f"grid_id {grid_id} has no data for {date}")
    prob_today = float(row_map["fire_prob"].iloc[0])
    fire_today = int(row_map["fire"].iloc[0])
    dom_today  = str(row_map["dominant_factor"].iloc[0]) if "dominant_factor" in row_map.columns else "N/A"

    prev_map = _by_date.get(_prev_date_str(date))
    if prev_map is not None:
        row_prev = prev_map[prev_map["grid_id"] == grid_id]
        prob_yesterday = float(row_prev["fire_prob"].iloc[0]) if not row_prev.empty else prob_today
    else:
        prob_yesterday = prob_today

    grid_hist = _by_grid.get(grid_id)

    fire_cls = shap_vals = human_sigs = weather = None
    if grid_hist is not None:
        ts = pd.Timestamp(date)
        row_detail = grid_hist[grid_hist["date"] == ts]
        if not row_detail.empty:
            r          = row_detail.iloc[0]
            fire_cls   = _fire_classification(r)
            shap_vals  = _shap_to_dict(r)
            human_sigs = _human_signals(r)
            weather    = _weather_vals(r)

    history_30d: list[dict] = []
    if grid_hist is not None:
        anchor = pd.Timestamp(date)
        mask   = (grid_hist["date"] >= anchor - timedelta(days=29)) & \
                 (grid_hist["date"] <= anchor)
        for r in grid_hist[mask].sort_values("date").itertuples(index=False):
            history_30d.append({
                "date":  str(r.date.date()),
                "prob":  round(float(r.fire_prob), 4),
                "fire":  int(r.fire),
            })

    fire_events_12m: list[dict] = []
    if grid_hist is not None:
        anchor = pd.Timestamp(date)
        mask   = (grid_hist["date"] >= anchor - timedelta(days=365)) & \
                 (grid_hist["date"] <= anchor) & \
                 (grid_hist["fire"] == 1)
        dominant_col_exists = "dominant_factor" in grid_hist.columns
        for r in grid_hist[mask].sort_values("date", ascending=False).itertuples(index=False):
            cause = str(r.dominant_factor) if dominant_col_exists else "N/A"
            fire_events_12m.append({
                "date":           str(r.date.date()),
                "prob":           round(float(r.fire_prob), 4),
                "analyzed_cause": f"Nhân tố chính: {cause}",
                "responded":      False,
            })

    neighbor_gids = _find_neighbors(lat, lon, exclude_id=grid_id)
    today_prob_map: dict[int, float] = dict(
        zip(today_map["grid_id"].values, today_map["fire_prob"].values.astype(float))
    )
    today_dom_map: dict[int, str] = {}
    if "dominant_factor" in today_map.columns:
        today_dom_map = dict(zip(today_map["grid_id"].values,
                                 today_map["dominant_factor"].astype(str).values))

    neighbors = sorted(
        [
            {
                "grid_id":         ngid,
                "prob":            round(today_prob_map.get(ngid, 0.0), 4),
                "delta":           round(today_prob_map.get(ngid, 0.0) - prob_today, 4),
                "dominant_factor": today_dom_map.get(ngid, "N/A"),
            }
            for ngid in neighbor_gids
        ],
        key=lambda x: x["prob"],
        reverse=True,
    )

    return {
        "grid_id":             grid_id,
        "center_lat":          round(lat, 6),
        "center_lon":          round(lon, 6),
        "prob":                round(prob_today, 4),
        "prob_yesterday":      round(prob_yesterday, 4),
        "fire":                fire_today,
        "dominant_factor":     dom_today,
        "fire_classification": fire_cls,
        "shap_values":         shap_vals,
        "human_signals":       human_sigs,
        "weather_vals":        weather,
        "history_30d":         history_30d,
        "fire_events_12m":     fire_events_12m,
        "neighbors":           neighbors,
    }


@app.get("/api/fire-type-summary")
def get_fire_type_summary(date: str = Query(..., description="YYYY-MM-DD")):
    """Daily fire subtype counts across all grids for the given date."""
    data = _summary_by_date.get(date, [])
    return {"date": date, "subtypes": data}


#  Entry point 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host=HOST, port=PORT, reload=False)
