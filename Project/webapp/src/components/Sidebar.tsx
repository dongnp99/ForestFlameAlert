import { useState } from 'react';
import { useFireStore } from '../store/useFireStore';
import RiskTab from './tabs/RiskTab';
import HistoryTab from './tabs/HistoryTab';
import NeighborsTab from './tabs/NeighborsTab';
import AlertsTab from './tabs/AlertsTab';

const TABS = [
  { id: 'risk',      label: 'Phân tích rủi ro' },
  { id: 'history',   label: 'Lịch sử 30 ngày' },
  { id: 'neighbors', label: 'So sánh lân cận' },
] as const;

type TabId = (typeof TABS)[number]['id'];

export default function Sidebar() {
  const [activeTab, setActiveTab] = useState<TabId>('risk');
  const { selectedGridId, setSelectedGridId, isLoadingDetail } = useFireStore();

  if (!selectedGridId) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Sidebar header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 flex-shrink-0">
        <span className="text-slate-400 text-xs uppercase tracking-widest">Chi tiết ô lưới</span>
        <button
          onClick={() => setSelectedGridId(null)}
          className="text-slate-500 hover:text-slate-200 transition-colors text-lg leading-none"
          title="Đóng (Esc)"
        >
          ×
        </button>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-slate-800 flex-shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2.5 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'border-b-2 border-orange-500 text-white'
                : 'text-slate-500 hover:text-slate-300 border-b-2 border-transparent'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Loading overlay */}
      {isLoadingDetail ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <svg className="w-6 h-6 animate-spin text-orange-400" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
            </svg>
            <span className="text-slate-400 text-sm">Đang tải dữ liệu...</span>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'risk'      && <RiskTab />}
          {activeTab === 'history'   && <HistoryTab />}
          {activeTab === 'neighbors' && <NeighborsTab />}
          {activeTab === 'alerts'    && <AlertsTab />}
        </div>
      )}
    </div>
  );
}
