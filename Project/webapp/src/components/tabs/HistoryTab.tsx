import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceDot,
  ResponsiveContainer,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import { useFireStore } from '../../store/useFireStore';
import type { DayHistory } from '../../types';

export default function HistoryTab() {
  const { gridDetail } = useFireStore();

  if (!gridDetail?.history_30d) {
    return <div className="p-4 text-slate-500 text-sm">Không có dữ liệu lịch sử.</div>;
  }

  const history = gridDetail.history_30d;
  const fireDays = history.filter((d) => d.fire === 1);

  // Top 5 days by prob
  const top5 = [...history].sort((a, b) => b.prob - a.prob).slice(0, 5);

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-slate-400 text-xs uppercase tracking-wide font-medium">
          Xác suất cháy — 30 ngày gần nhất
        </p>
        {fireDays.length > 0 && (
          <span className="text-xs bg-red-900/50 text-red-400 px-2 py-0.5 rounded-full">
            {fireDays.length} ngày cháy xác nhận
          </span>
        )}
      </div>

      {/* Line chart */}
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(d: string) => format(parseISO(d), 'dd/MM')}
              interval={4}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip
              formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Xác suất']}
              labelFormatter={(l: string) => format(parseISO(l), 'dd/MM/yyyy')}
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: 12 }}
            />
            <Line
              type="monotone"
              dataKey="prob"
              stroke="#EF9F27"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#EF9F27' }}
            />
            {/* Fire event dots */}
            {fireDays.map((d) => (
              <ReferenceDot
                key={d.date}
                x={d.date}
                y={d.prob}
                r={5}
                fill="#E24B4A"
                stroke="#fff"
                strokeWidth={1.5}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {fireDays.length > 0 && (
        <p className="text-xs text-slate-500">
          <span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-1.5 align-middle" />
          Chấm đỏ = ngày có cháy xác nhận
        </p>
      )}

      {/* Top 5 table */}
      <div>
        <p className="text-slate-400 text-xs uppercase tracking-wide font-medium mb-2">
          Top 5 ngày xác suất cao nhất
        </p>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              <th className="text-left py-1.5 font-medium">Ngày</th>
              <th className="text-right py-1.5 font-medium">Xác suất</th>
              <th className="text-right py-1.5 font-medium">Cháy</th>
            </tr>
          </thead>
          <tbody>
            {top5.map((d, i) => (
              <tr key={d.date} className={`border-b border-slate-800/50 ${i === 0 ? 'text-orange-300' : 'text-slate-300'}`}>
                <td className="py-1.5">{format(parseISO(d.date), 'dd/MM/yyyy')}</td>
                <td className="text-right font-mono">{(d.prob * 100).toFixed(1)}%</td>
                <td className="text-right">
                  {d.fire === 1
                    ? <span className="text-red-400 font-medium">Có</span>
                    : <span className="text-slate-600">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
