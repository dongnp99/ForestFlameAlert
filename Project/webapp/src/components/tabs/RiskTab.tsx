import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useFireStore } from '../../store/useFireStore';
import type { FireClassification } from '../../types';
import html2canvas from 'html2canvas';
import { useRef } from 'react';

const PATHWAY_LABELS: Record<string, string> = {
  P1: 'Đốt nương rẫy sau thu hoạch',
  P2: 'Phá rừng / Lịch sử cháy',
  P3: 'Cháy lặp lại theo mùa',
};

const GROUP_COLORS: Record<string, string> = {
  human:   '#EF9F27',   // orange — human activity
  natural: '#3B82F6',   // blue   — natural factors
};

const GROUP_LABELS: Record<string, string> = {
  human:   'Con người',
  natural: 'Tự nhiên',
};

function FireClassificationCard({ fc }: { fc: FireClassification }) {
  const isHuman   = fc.fire_type === 'Con người';
  const border    = isHuman ? 'border-orange-500'   : 'border-blue-500';
  const bg        = isHuman ? 'bg-orange-950/20'    : 'bg-blue-950/20';
  const label     = isHuman ? 'text-orange-400'     : 'text-blue-400';
  const pillBg    = isHuman ? 'bg-orange-900/50 text-orange-300' : 'bg-blue-900/50 text-blue-300';
  const tagBg     = isHuman ? 'bg-orange-800/60 text-orange-200' : 'bg-blue-800/60 text-blue-200';
  const typeIcon  = isHuman ? '👤' : '🌿';
  const confPct   = Math.round(fc.confidence_score * 100);
  const confBar   = confPct >= 70 ? 'bg-emerald-500' : confPct >= 50 ? 'bg-yellow-500' : 'bg-red-500';
  const hasPathway = fc.matched_pathways && fc.matched_pathways !== 'None';

  return (
    <div className={`border-l-4 ${border} pl-3 ${bg} rounded-r-lg py-3 pr-3 space-y-2`}>
      <p className={`${label} text-xs font-semibold uppercase tracking-wide`}>
        Phân loại nguyên nhân cháy
      </p>
      <div className="flex items-center justify-between gap-2">
        <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${pillBg}`}>
          {typeIcon} {fc.fire_type}
        </span>
        <span className="text-slate-300 text-xs text-right">{fc.fire_subtype}</span>
      </div>
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Độ tin cậy phân loại</span>
          <span className="font-mono">{confPct}%</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div className={`h-full ${confBar} rounded-full`} style={{ width: `${confPct}%` }} />
        </div>
      </div>
      {hasPathway && (
        <div className="flex items-center gap-2">
          <span className={`text-xs px-1.5 py-0.5 rounded font-mono font-bold ${tagBg}`}>
            {fc.matched_pathways}
          </span>
          <span className="text-slate-400 text-xs">
            {PATHWAY_LABELS[fc.matched_pathways] ?? fc.matched_pathways}
          </span>
        </div>
      )}
    </div>
  );
}

function ProbBadge({ prob }: { prob: number }) {
  const pct = (prob * 100).toFixed(1);
  const color =
    prob > 0.6 ? 'text-red-400' :
    prob > 0.4 ? 'text-orange-400' :
    prob > 0.2 ? 'text-yellow-400' : 'text-emerald-400';
  return <span className={`text-5xl font-black tabular-nums ${color}`}>{pct}%</span>;
}

function DeltaBadge({ delta }: { delta: number }) {
  const sign = delta >= 0 ? '+' : '';
  const color = delta > 0 ? 'text-red-400 bg-red-900/30' : 'text-emerald-400 bg-emerald-900/30';
  return (
    <span className={`inline-flex items-center gap-0.5 text-xs px-2 py-0.5 rounded-full font-mono ${color}`}>
      {sign}{(delta * 100).toFixed(1)}% so với hôm qua
    </span>
  );
}

export default function RiskTab() {
  const { gridDetail, selectedDate } = useFireStore();
  const exportRef = useRef<HTMLDivElement>(null);

  if (!gridDetail) {
    return (
      <div className="flex items-center justify-center h-40 text-slate-500 text-sm">
        Chọn một ô lưới trên bản đồ để xem chi tiết
      </div>
    );
  }

  const { grid_id, prob, prob_yesterday, center_lat, center_lon, fire_classification, shap_values, human_signals, weather_vals, fire } = gridDetail;
  const delta = prob - prob_yesterday;

  const handleExport = async () => {
    if (!exportRef.current) return;
    const canvas = await html2canvas(exportRef.current, { backgroundColor: '#0f172a' });
    const link = document.createElement('a');
    link.download = `firewatch_grid${grid_id}_${selectedDate}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div ref={exportRef} className="p-4 space-y-4">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white font-mono">Grid {grid_id}</h2>
          <div className="flex items-center gap-2">
            {fire === 1 && (
              <span className="text-xs bg-red-900/60 text-red-400 px-2 py-0.5 rounded-full font-medium">
                🔥 Cháy xác nhận
              </span>
            )}
            <span className="text-xs text-slate-500 font-mono">{selectedDate}</span>
          </div>
        </div>
        <p className="text-xs text-slate-400 font-mono">
          {center_lat.toFixed(6)}°N, {center_lon.toFixed(6)}°E
        </p>
      </div>

      {/* Probability display */}
      <div className="bg-slate-800/50 rounded-xl p-4 flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-xs mb-1 uppercase tracking-wide">Xác suất cháy</p>
          <ProbBadge prob={prob} />
        </div>
        <div className="text-right space-y-2">
          <DeltaBadge delta={delta} />
          <p className="text-slate-500 text-xs">Hôm qua: {(prob_yesterday * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Fire Classification */}
      {fire_classification
        ? <FireClassificationCard fc={fire_classification} />
        : (
          <div className="border-l-4 border-slate-600/50 pl-3 bg-slate-800/20 rounded-r-lg py-3 pr-3">
            <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide">
              Phân loại nguyên nhân cháy — không có dữ liệu
            </p>
          </div>
        )
      }

      {/* SHAP contribution — only if available */}
      {shap_values ? (
        <div>
          <p className="text-slate-400 text-xs mb-3 uppercase tracking-wide font-medium">
            Đóng góp theo nhóm (SHAP)
          </p>
          <div className="h-10 mb-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={[Object.entries(shap_values).reduce((acc, [k, v]) => ({ ...acc, [k]: v }), { name: 'Đóng góp' })]}
                margin={{ left: 0, right: 0, top: 0, bottom: 0 }}
              >
                <XAxis type="number" domain={[0, 100]} hide />
                <YAxis type="category" dataKey="name" hide />
                <Tooltip
                  formatter={(value: number, name: string) => [`${value}%`, GROUP_LABELS[name] ?? name]}
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: 12 }}
                  labelStyle={{ display: 'none' }}
                />
                {Object.keys(shap_values).map((key) => (
                  <Bar key={key} dataKey={key} stackId="a" fill={GROUP_COLORS[key]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            {Object.entries(shap_values).sort(([, a], [, b]) => b - a).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: GROUP_COLORS[key] }} />
                  <span className="text-slate-300 text-xs">{GROUP_LABELS[key]}</span>
                </div>
                <span className="text-slate-400 text-xs font-mono">{val}%</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/30 border border-slate-700/50 rounded-lg p-3 text-xs text-slate-500 text-center">
          SHAP values không có trong parquet — cần tích hợp feature dataset gốc
        </div>
      )}

      {/* Human Activity Signals */}
      {human_signals ? (
        <div className="border-l-4 border-orange-500 pl-3 bg-orange-950/20 rounded-r-lg py-3 pr-3 space-y-1.5">
          <p className="text-orange-400 text-xs font-semibold uppercase tracking-wide mb-2">
            Human Activity Signals
          </p>
          {[
            { label: 'Mùa đốt rẫy',           value: human_signals.burning_season ? '✓ Có' : '✗ Không' },
            { label: 'Số lần cháy 5 năm',      value: `${human_signals.fire_count_5yr} lần` },
            { label: 'Khoảng cách tới đường',  value: `${human_signals.dist_road_km} km` },
            { label: 'Land use',               value: human_signals.land_use },
            { label: 'Khu dân cư gần nhất',    value: human_signals.nearest_settlement },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between text-xs gap-2">
              <span className="text-slate-400">{label}</span>
              <span className="text-slate-200 font-medium text-right">{value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="border-l-4 border-orange-500/40 pl-3 bg-orange-950/10 rounded-r-lg py-3 pr-3">
          <p className="text-orange-400/60 text-xs font-semibold uppercase tracking-wide">
            Human Activity Signals — không có trong parquet
          </p>
        </div>
      )}

      {/* Weather */}
      {weather_vals ? (
        <div className="border-l-4 border-blue-500 pl-3 bg-blue-950/20 rounded-r-lg py-3 pr-3 space-y-1.5">
          <p className="text-blue-400 text-xs font-semibold uppercase tracking-wide mb-2">Thời tiết</p>
          {[
            { label: 'Nhiệt độ TB (°C)',      value: weather_vals.tmean.toFixed(1) },
            { label: 'Độ ẩm tương đối (%)',   value: weather_vals.rh.toString() },
            { label: 'VPD (kPa)',              value: weather_vals.vpd.toFixed(2) },
            { label: 'Mưa 14 ngày (mm)',       value: weather_vals.rain_14d_sum.toFixed(1) },
            { label: 'VPD TB 14 ngày (kPa)',   value: weather_vals.vpd_14d_mean.toFixed(2) },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between text-xs gap-2">
              <span className="text-slate-400">{label}</span>
              <span className="text-slate-200 font-mono font-medium">{value}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="border-l-4 border-blue-500/40 pl-3 bg-blue-950/10 rounded-r-lg py-3 pr-3">
          <p className="text-blue-400/60 text-xs font-semibold uppercase tracking-wide">
            Thời tiết — không có trong parquet
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={handleExport}
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-medium transition-colors"
        >
          <span>📄</span> Tạo báo cáo
        </button>
        <button
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-orange-600 hover:bg-orange-500 text-white text-sm font-medium transition-colors"
        >
          <span>🛡</span> Gợi ý ứng phó
        </button>
      </div>
    </div>
  );
}
