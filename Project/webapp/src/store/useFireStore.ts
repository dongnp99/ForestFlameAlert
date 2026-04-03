import { create } from 'zustand';
import { fetchProbMap, fetchGridDetail, fetchDateRange } from '../api/mockApi';
import type { FilterMode, GridDetail, ProbMapCollection } from '../types';

// Data is 2023-2024; default to last day of data range
const DEFAULT_DATE = '2024-04-08'; // highest fire activity date

interface FireState {
  selectedDate: string;
  minDate: string;
  maxDate: string;
  selectedGridId: number | null;
  gridCells: ProbMapCollection | null;
  gridDetail: GridDetail | null;
  isLoadingMap: boolean;
  isLoadingDetail: boolean;
  filterMode: FilterMode;
  threshold: number;
  onlyBurningSeason: boolean;
  showActualFires: boolean;

  setSelectedDate: (date: string) => void;
  setSelectedGridId: (id: number | null) => void;
  loadProbMap: (date: string) => Promise<void>;
  loadGridDetail: (gridId: number, date: string) => Promise<void>;
  loadDateRange: () => Promise<void>;
  setFilterMode: (mode: FilterMode) => void;
  setThreshold: (val: number) => void;
  setOnlyBurningSeason: (val: boolean) => void;
  setShowActualFires: (val: boolean) => void;
}

export const useFireStore = create<FireState>((set, get) => ({
  selectedDate: DEFAULT_DATE,
  minDate: '2023-01-01',
  maxDate: '2024-12-31',
  selectedGridId: null,
  gridCells: null,
  gridDetail: null,
  isLoadingMap: false,
  isLoadingDetail: false,
  filterMode: 'all',
  threshold: 0,
  onlyBurningSeason: false,
  showActualFires: false,

  setSelectedDate: (date) => {
    set({ selectedDate: date, selectedGridId: null, gridDetail: null });
    get().loadProbMap(date);
  },

  setSelectedGridId: (id) => {
    set({ selectedGridId: id, gridDetail: null });
    if (id !== null) get().loadGridDetail(id, get().selectedDate);
  },

  loadProbMap: async (date) => {
    set({ isLoadingMap: true });
    try {
      const data = await fetchProbMap(date);
      set({ gridCells: data });
    } finally {
      set({ isLoadingMap: false });
    }
  },

  loadGridDetail: async (gridId, date) => {
    set({ isLoadingDetail: true });
    try {
      const data = await fetchGridDetail(String(gridId), date);
      set({ gridDetail: data });
    } finally {
      set({ isLoadingDetail: false });
    }
  },

  loadDateRange: async () => {
    try {
      const { min_date, max_date } = await fetchDateRange();
      set({ minDate: min_date, maxDate: max_date });
    } catch {
      // keep defaults
    }
  },

  setFilterMode: (mode) => set({ filterMode: mode }),
  setThreshold: (val) => set({ threshold: val }),
  setOnlyBurningSeason: (val) => set({ onlyBurningSeason: val }),
  setShowActualFires: (val) => set({ showActualFires: val }),
}));
