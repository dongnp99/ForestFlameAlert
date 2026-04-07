import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, useMap, LayersControl, ZoomControl } from 'react-leaflet';
import { useFireStore } from '../store/useFireStore';
import MapLegend from './MapLegend';
import type { GridProperties, ProbMapCollection } from '../types';

// ── Fire probability thresholds ────────────────────────────────────────────────
const THRESH_HIGH   = 0.9;
const THRESH_MEDIUM = 0.6;
const THRESH_LOW    = 0.3;

// ── Color helpers ──────────────────────────────────────────────────────────────
function getProbColor(prob: number): string {
  if (prob > THRESH_HIGH)   return '#E74C3C';
  if (prob > THRESH_MEDIUM) return '#E67E22';
  if (prob > THRESH_LOW)    return '#F1C40F';
  return '#2ECC71';
}

function getProbOpacity(prob: number): number {
  if (prob > THRESH_HIGH)   return 0.90;
  if (prob > THRESH_MEDIUM) return 0.75;
  if (prob > THRESH_LOW)    return 0.60;
  return 0.40;
}

function getCellStyle(
  prob: number,
  fire: number,
  selected: boolean,
  showActualFires: boolean,
): L.PathOptions {
  if (showActualFires && fire === 1) {
    return {
      fillColor: '#FACC15',
      fillOpacity: 0.85,
      color: selected ? '#FFFFFF' : '#FDE68A',
      weight: selected ? 2.5 : 1.5,
    };
  }
  return {
    fillColor: getProbColor(prob),
    fillOpacity: getProbOpacity(prob),
    color: selected ? '#FFFFFF' : 'transparent',
    weight: selected ? 2 : 0,
  };
}

// ── Grid canvas layer ──────────────────────────────────────────────────────────
function GridCanvasLayer() {
  const map = useMap();
  const layerRef = useRef<L.GeoJSON | null>(null);
  const {
    gridCells, selectedGridId, setSelectedGridId,
    filterMode, threshold, showActualFires,
  } = useFireStore();

  useEffect(() => {
    if (!gridCells) return;

    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    const renderer = L.canvas({ padding: 0.5 });

    const filtered: ProbMapCollection = {
      ...gridCells,
      features: gridCells.features.filter((f) => {
        const { prob } = f.properties;
        if (filterMode === 'high'   && prob <= THRESH_HIGH) return false;
        if (filterMode === 'medium' && (prob < THRESH_LOW || prob > THRESH_HIGH)) return false;
        if (prob < threshold) return false;
        return true;
      }),
    };

    const geoLayer = L.geoJSON(filtered as GeoJSON.FeatureCollection, {
      renderer,
      style: (feature) => {
        const props = (feature as GeoJSON.Feature<GeoJSON.Geometry, GridProperties>).properties;
        const isSelected = props?.grid_id === selectedGridId;
        return getCellStyle(props?.prob ?? 0, props?.fire ?? 0, isSelected, showActualFires);
      },
      onEachFeature: (feature, layer) => {
        const { grid_id, prob, fire } = (feature as GeoJSON.Feature<GeoJSON.Geometry, GridProperties>).properties;
        const pct = (prob * 100).toFixed(1);
        const fireLabel = fire === 1 ? ' 🔥 Cháy xác nhận' : '';
        layer.bindTooltip(
          `<b>Grid ${grid_id}</b><br/>Xác suất: <b>${pct}%</b>${fireLabel}`,
          { sticky: true },
        );
        layer.on('click', () => setSelectedGridId(grid_id));
      },
    });

    geoLayer.addTo(map);
    layerRef.current = geoLayer;

    return () => {
      if (layerRef.current) map.removeLayer(layerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridCells, filterMode, threshold, showActualFires, map]);

  // Update selection highlight without rebuilding layer
  useEffect(() => {
    if (!layerRef.current) return;
    layerRef.current.eachLayer((l) => {
      const feature = (l as { feature?: GeoJSON.Feature<GeoJSON.Geometry, GridProperties> }).feature;
      if (feature) {
        const prob = feature.properties?.prob ?? 0;
        const fire = feature.properties?.fire ?? 0;
        const isSelected = feature.properties?.grid_id === selectedGridId;
        (l as L.Path).setStyle(getCellStyle(prob, fire, isSelected, showActualFires));
      }
    });
  }, [selectedGridId, showActualFires]);

  // Pan to selected grid
  useEffect(() => {
    if (selectedGridId === null || !gridCells) return;
    const feat = gridCells.features.find((f) => f.properties.grid_id === selectedGridId);
    if (feat) {
      const { center_lat, center_lon } = feat.properties;
      map.panTo([center_lat, center_lon], { animate: true });
    }
  }, [selectedGridId, gridCells, map]);

  return null;
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function MapView() {
  return (
    <div className="w-full h-full relative">
      <MapContainer
        center={[12.71, 108.23]}
        zoom={9}
        zoomControl={false}
        className="w-full h-full"
        preferCanvas={true}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="CartoDB DarkMatter">
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              subdomains="abcd"
              maxZoom={20}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="OpenStreetMap">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>'
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        <ZoomControl position="topright" />
        <GridCanvasLayer />
      </MapContainer>

      <MapSpinner />
      <MapLegend />
    </div>
  );
}

function MapSpinner() {
  const { isLoadingMap } = useFireStore();
  if (!isLoadingMap) return null;
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-[500]">
      <div className="bg-slate-900/70 rounded-xl px-4 py-3 flex items-center gap-3 shadow-xl">
        <svg className="w-5 h-5 animate-spin text-orange-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
        </svg>
        <span className="text-slate-300 text-sm">Đang tải {(18803).toLocaleString()} ô lưới...</span>
      </div>
    </div>
  );
}
