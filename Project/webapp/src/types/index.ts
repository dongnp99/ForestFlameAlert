export interface HumanSignals {
  burning_season: boolean;
  fire_count_5yr: number;
  dist_road_km: number;
  land_use: string;
  nearest_settlement: string;
}

export interface WeatherVals {
  tmean: number;
  rh: number;
  vpd: number;
  rain_14d_sum: number;
  vpd_14d_mean: number;
}

export interface ShapValues {
  human:   number;   // % contribution from human-activity features
  natural: number;   // % contribution from natural factors (weather, fire spread, vegetation, terrain)
}

export interface GridProperties {
  grid_id: number;
  prob: number;
  prob_yesterday: number;
  dominant_factor: string;
  center_lat: number;
  center_lon: number;
  fire: number;
  // Optional — only available when original feature dataset is loaded
  human_signals?: HumanSignals | null;
  weather_vals?: WeatherVals | null;
}

export interface FireEvent {
  date: string;
  prob: number;
  analyzed_cause: string;
  responded: boolean;
}

export interface NeighborCell {
  grid_id: number;
  prob: number;
  delta: number;
  dominant_factor: string;
}

export interface DayHistory {
  date: string;
  prob: number;
  fire: 0 | 1;
}

export interface GridDetail extends GridProperties {
  shap_values?: ShapValues | null;
  fire_events_12m: FireEvent[];
  neighbors: NeighborCell[];
  history_30d: DayHistory[];
}

export type FilterMode = 'all' | 'high' | 'medium';

export interface ProbMapFeature extends GeoJSON.Feature<GeoJSON.Polygon, GridProperties> {}
export interface ProbMapCollection extends GeoJSON.FeatureCollection<GeoJSON.Polygon, GridProperties> {}
