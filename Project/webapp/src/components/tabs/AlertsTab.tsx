import { format, parseISO } from 'date-fns';
import { useFireStore } from '../../store/useFireStore';

export default function AlertsTab() {
  const { gridDetail } = useFireStore();

  if (!gridDetail) return null;

  const { fire_events_12m, grid_id } = gridDetail;

  if (fire_events_12m.length === 0) {
    return (
      <div className="p-4 flex flex-col items-center justify-center gap-3 text-center py-12">
        <span className="text-4xl">✅</span>
        <p className="text-slate-400 text-sm">
          Không có sự kiện cháy xác nhận trong 12 tháng gần nhất tại <b className="text-slate-200">{grid_id}</b>.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-slate-400 text-xs uppercase tracking-wide font-medium">
          Sự kiện cháy xác nhận — 12 tháng qua
        </p>
        <span className="text-xs bg-red-900/50 text-red-400 px-2 py-0.5 rounded-full">
          {fire_events_12m.length} sự kiện
        </span>
      </div>

      {/* Vertical timeline */}
      <div className="relative pl-5">
        {/* Timeline line */}
        <div className="absolute left-[9px] top-2 bottom-2 w-px bg-slate-700" />

        <div className="space-y-5">
          {fire_events_12m.map((event, i) => (
            <div key={i} className="relative">
              {/* Dot */}
              <div
                className={`absolute -left-5 top-1 w-3.5 h-3.5 rounded-full border-2 z-10 ${
                  event.responded
                    ? 'bg-emerald-600 border-emerald-400'
                    : 'bg-red-600 border-red-400 animate-pulse'
                }`}
              />

              <div className="bg-slate-800/60 rounded-lg p-3 space-y-1.5">
                {/* Date + prob */}
                <div className="flex items-center justify-between">
                  <span className="text-slate-200 text-sm font-medium">
                    {format(parseISO(event.date), 'dd/MM/yyyy')}
                  </span>
                  <span className="text-orange-400 font-mono text-sm font-bold">
                    {(event.prob * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Cause */}
                <p className="text-slate-400 text-xs leading-relaxed">{event.analyzed_cause}</p>

                {/* Status */}
                <div className="flex items-center gap-1.5">
                  {event.responded ? (
                    <>
                      <span className="w-2 h-2 rounded-full bg-emerald-500" />
                      <span className="text-emerald-400 text-xs">Đã ứng phó</span>
                    </>
                  ) : (
                    <>
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      <span className="text-red-400 text-xs font-medium">Chưa ứng phó</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="border border-slate-700 rounded-lg p-3 text-xs mt-2">
        <div className="flex justify-between text-slate-400 mb-1">
          <span>Đã ứng phó:</span>
          <span className="text-emerald-400 font-medium">
            {fire_events_12m.filter((e) => e.responded).length} / {fire_events_12m.length}
          </span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Chưa ứng phó:</span>
          <span className="text-red-400 font-medium">
            {fire_events_12m.filter((e) => !e.responded).length} sự kiện
          </span>
        </div>
      </div>
    </div>
  );
}
