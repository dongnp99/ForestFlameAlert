import { useFireStore } from '../store/useFireStore';
import type { FilterMode } from '../types';

const FILTER_OPTIONS: { value: FilterMode; label: string }[] = [
  { value: 'all',    label: 'Tất cả' },
  { value: 'high',   label: 'Chỉ cao (>0.6)' },
  { value: 'medium', label: 'Chỉ trung bình (0.2–0.6)' },
];

export default function FilterBar() {
  const { filterMode, setFilterMode, threshold, setThreshold, isLoadingMap, showActualFires, setShowActualFires } = useFireStore();

  return (
    <div className="flex items-center gap-4 px-3 py-2 bg-slate-900/80 border-b border-slate-800 text-sm flex-shrink-0 flex-wrap">
      <div className="flex items-center gap-2">
        <span className="text-slate-400 whitespace-nowrap">Hiển thị:</span>
        <select
          value={filterMode}
          onChange={(e) => setFilterMode(e.target.value as FilterMode)}
          className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-orange-500 cursor-pointer"
        >
          {FILTER_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-slate-400 whitespace-nowrap">Ngưỡng:</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-24 accent-orange-500 cursor-pointer"
        />
        <span className="text-orange-400 font-mono text-xs w-8">{threshold.toFixed(2)}</span>
      </div>

      {/* Actual fire overlay toggle */}
      <button
        onClick={() => setShowActualFires(!showActualFires)}
        className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors cursor-pointer ${
          showActualFires
            ? 'bg-yellow-500/20 border-yellow-400/60 text-yellow-300'
            : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'
        }`}
      >
        <span
          className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
          style={{ backgroundColor: showActualFires ? '#FACC15' : '#64748B' }}
        />
        Cháy thực tế
      </button>

      {/* Data source badge */}
      <div className="flex items-center gap-1.5 text-slate-500 text-xs">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
        Dữ liệu: app_predictions.parquet (2023–2024)
      </div>

      {isLoadingMap && (
        <div className="flex items-center gap-1.5 text-orange-400 ml-auto">
          <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
          </svg>
          <span className="text-xs">Đang tải bản đồ...</span>
        </div>
      )}
    </div>
  );
}
