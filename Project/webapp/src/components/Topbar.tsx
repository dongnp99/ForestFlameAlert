import { format, parseISO, addDays } from 'date-fns';
import { vi } from 'date-fns/locale';
import { useFireStore } from '../store/useFireStore';

export default function Topbar() {
  const { selectedDate, setSelectedDate, minDate, maxDate } = useFireStore();

  function shiftDay(delta: number) {
    const next = format(addDays(parseISO(selectedDate), delta), 'yyyy-MM-dd');
    if (next >= minDate && next <= maxDate) setSelectedDate(next);
  }

  return (
    <header className="flex items-center gap-4 px-4 h-[52px] bg-slate-900 border-b border-slate-800 flex-shrink-0 z-50">
      <div className="flex items-center gap-2 mr-2">
        <span className="text-2xl">🔥</span>
        <span className="font-bold text-base tracking-tight text-white whitespace-nowrap">
          Forest Flame Alert
        </span>
      </div>

      <div className="h-5 w-px bg-slate-700" />

      <div className="flex items-center gap-1">
        <button
          onClick={() => shiftDay(-1)}
          disabled={selectedDate <= minDate}
          className="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-sm leading-none"
        >
          ‹
        </button>
        <input
          type="date"
          value={selectedDate}
          min={minDate}
          max={maxDate}
          onChange={(e) => e.target.value && setSelectedDate(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-slate-100 focus:outline-none focus:border-orange-500 cursor-pointer"
        />
        <button
          onClick={() => shiftDay(1)}
          disabled={selectedDate >= maxDate}
          className="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-sm leading-none"
        >
          ›
        </button>
      </div>

      <div className="hidden sm:flex items-center gap-1 px-2 py-1 bg-emerald-900/50 border border-emerald-700/50 rounded text-emerald-400 text-xs font-mono">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        Model v5 · Human features pathways
      </div>

      <div className="flex-1" />

      <span className="hidden lg:block text-slate-500 text-xs">
        {format(parseISO(selectedDate), 'EEEE, dd/MM/yyyy', { locale: vi })}
      </span>
    </header>
  );
}
