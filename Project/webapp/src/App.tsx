import { useEffect, useCallback } from 'react';
import { format, addDays, subDays, parseISO } from 'date-fns';
import { useFireStore } from './store/useFireStore';
import Topbar from './components/Topbar';
import FilterBar from './components/FilterBar';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';

export default function App() {
  const {
    selectedDate, setSelectedDate, minDate, maxDate,
    gridCells, selectedGridId,
    loadProbMap, loadDateRange,
  } = useFireStore();

  // Load date range, then initial map data
  useEffect(() => {
    loadDateRange().then(() => loadProbMap(selectedDate));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard shortcuts: ← / → change date, Escape deselect
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === 'ArrowLeft') {
        const prev = format(subDays(parseISO(selectedDate), 1), 'yyyy-MM-dd');
        if (prev >= minDate) setSelectedDate(prev);
      }
      if (e.key === 'ArrowRight') {
        const next = format(addDays(parseISO(selectedDate), 1), 'yyyy-MM-dd');
        if (next <= maxDate) setSelectedDate(next);
      }
      if (e.key === 'Escape') useFireStore.getState().setSelectedGridId(null);
    },
    [selectedDate, minDate, maxDate, setSelectedDate],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const highRiskCount = gridCells
    ? gridCells.features.filter((f) => f.properties.prob > 0.6).length
    : 0;

  const sidebarOpen = selectedGridId !== null;

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100">
      <Topbar />

      {highRiskCount > 5 && (
        <div className="flex items-center gap-2 px-4 py-2 bg-red-600/90 text-sm font-medium z-50">
          <span>⚠</span>
          <span>{highRiskCount} vùng nguy cơ cháy cao hôm nay — Cần giám sát khẩn cấp</span>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden relative">
        <div className={`flex flex-col transition-all duration-300 ${sidebarOpen ? 'w-[65%]' : 'w-full'}`}>
          <FilterBar />
          <div className="flex-1 relative">
            <MapView />
          </div>
        </div>

        {sidebarOpen && (
          <div className="w-[35%] border-l border-slate-800 overflow-y-auto bg-slate-900">
            <Sidebar />
          </div>
        )}

        {/* Mobile bottom sheet */}
        {sidebarOpen && (
          <div className="md:hidden fixed bottom-0 left-0 right-0 h-2/3 bg-slate-900 border-t border-slate-700 overflow-y-auto z-[1000] rounded-t-2xl">
            <div className="flex justify-center py-2">
              <div className="w-10 h-1 bg-slate-600 rounded-full" />
            </div>
            <Sidebar />
          </div>
        )}
      </div>
    </div>
  );
}
