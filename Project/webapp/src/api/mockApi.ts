import type { GridDetail, ProbMapCollection, FireTypeSummaryEntry } from '../types';

const API_BASE = '/api';

export async function fetchProbMap(date: string): Promise<ProbMapCollection> {
  const res = await fetch(`${API_BASE}/prob-map?date=${date}`);
  if (!res.ok) throw new Error(`prob-map ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchGridDetail(gridId: string, date: string): Promise<GridDetail> {
  const res = await fetch(`${API_BASE}/grid-detail?grid_id=${gridId}&date=${date}`);
  if (!res.ok) throw new Error(`grid-detail ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchDateRange(): Promise<{ min_date: string; max_date: string }> {
  const res = await fetch(`${API_BASE}/dates`);
  if (!res.ok) throw new Error('Failed to fetch date range');
  return res.json();
}

export async function fetchFireTypeSummary(date: string): Promise<{ date: string; subtypes: FireTypeSummaryEntry[] }> {
  const res = await fetch(`${API_BASE}/fire-type-summary?date=${date}`);
  if (!res.ok) throw new Error(`fire-type-summary ${res.status}: ${await res.text()}`);
  return res.json();
}
