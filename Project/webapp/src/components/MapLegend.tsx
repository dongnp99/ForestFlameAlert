import { useFireStore } from '../store/useFireStore';

export default function MapLegend() {
  const { showActualFires } = useFireStore();

  return (
    <div className="absolute bottom-8 left-3 z-[400] bg-slate-900/90 border border-slate-700/60 rounded-lg px-3 py-2.5 shadow-xl text-xs">
      <p className="text-slate-400 font-medium mb-2 uppercase tracking-wide text-[10px]">Xác suất cháy</p>

      {/* Gradient bar */}
      <div
        className="w-32 h-3 rounded-full mb-1.5"
        style={{
          background: 'linear-gradient(to right, #1D9E75, #EF9F27, #D85A30, #E24B4A)',
        }}
      />

      {/* Labels */}
      <div className="flex justify-between text-slate-400 text-[10px] mb-2">
        <span>0%</span>
        <span>30%</span>
        <span>60%</span>
        <span>80%</span>
        <span>100%</span>
      </div>

      {/* Discrete swatches */}
      <div className="flex flex-col gap-1">
        {[
          { color: '#1D9E75', label: 'Thấp  < 30%' },
          { color: '#EF9F27', label: 'TB    30–60%' },
          { color: '#D85A30', label: 'Cao   60–80%' },
          { color: '#E24B4A', label: 'Rất cao > 80%' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm flex-shrink-0"
              style={{ backgroundColor: color, opacity: 0.85 }}
            />
            <span className="text-slate-300">{label}</span>
          </div>
        ))}

        {showActualFires && (
          <div className="flex items-center gap-2 mt-1 pt-1 border-t border-slate-700/60">
            <span
              className="w-3 h-3 rounded-sm flex-shrink-0 border border-yellow-300"
              style={{ backgroundColor: '#FACC15', opacity: 0.85 }}
            />
            <span className="text-yellow-300">Cháy thực tế</span>
          </div>
        )}
      </div>
    </div>
  );
}
