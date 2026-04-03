import { useFireStore } from '../../store/useFireStore';

function ProbBar({ prob }: { prob: number }) {
  const color =
    prob > 0.6 ? 'bg-red-500' :
    prob > 0.4 ? 'bg-orange-500' :
    prob > 0.2 ? 'bg-yellow-500' : 'bg-emerald-500';
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 bg-slate-700 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full`} style={{ width: `${prob * 100}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-300">{(prob * 100).toFixed(0)}%</span>
    </div>
  );
}

export default function NeighborsTab() {
  const { gridDetail, setSelectedGridId } = useFireStore();

  if (!gridDetail?.neighbors?.length) {
    return <div className="p-4 text-slate-500 text-sm">Không có dữ liệu vùng lân cận.</div>;
  }

  const sorted = [...gridDetail.neighbors].sort((a, b) => b.prob - a.prob);

  return (
    <div className="p-4 space-y-3">
      <p className="text-slate-400 text-xs uppercase tracking-wide font-medium">
        {sorted.length} ô lưới lân cận — sắp xếp theo xác suất
      </p>

      <p className="text-slate-500 text-xs">
        Nhấp vào hàng để nhảy tới ô đó trên bản đồ.
      </p>

      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800">
            <th className="text-left py-1.5 font-medium">Grid ID</th>
            <th className="text-left py-1.5 font-medium">Xác suất</th>
            <th className="text-right py-1.5 font-medium">Δ</th>
            <th className="text-right py-1.5 font-medium hidden sm:table-cell">Nhân tố</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((n) => {
            const deltaSign = n.delta >= 0 ? '+' : '';
            const deltaColor = n.delta > 0 ? 'text-red-400' : 'text-emerald-400';
            return (
              <tr
                key={n.grid_id}
                onClick={() => setSelectedGridId(Number(n.grid_id))}
                className="border-b border-slate-800/50 hover:bg-slate-800/40 cursor-pointer transition-colors"
              >
                <td className="py-2 pr-2 text-orange-400 font-mono font-medium">#{n.grid_id}</td>
                <td className="py-2 pr-2">
                  <ProbBar prob={n.prob} />
                </td>
                <td className={`py-2 text-right font-mono ${deltaColor}`}>
                  {deltaSign}{(n.delta * 100).toFixed(0)}%
                </td>
                <td className="py-2 text-right text-slate-400 hidden sm:table-cell">
                  {n.dominant_factor}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
