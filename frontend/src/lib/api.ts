const BASE = "/api";

export interface GameConfig {
  id: string;
  name: string;
  pool: number;
  picks: number;
  draw_days: string[];
  draw_times: string[];
  ordered: boolean;
  low_boundary?: number;
}

export interface DrawRow {
  draw_date: string;
  weekday: string;
  game: string;
  draw_time: string | null;
  n1: number | null;
  n2: number | null;
  n3: number | null;
  n4: number | null;
  n5: number | null;
  n6: number | null;
  jackpot_php: number | null;
  notes: string | null;
}

export interface DrawsResponse {
  total: number;
  limit: number;
  offset: number;
  data: DrawRow[];
}

export interface CoverageRow {
  game: string;
  draws: number;
  first_draw: string;
  last_draw: string;
}

export interface AnalysisResult {
  game: string;
  test: string;
  conclusion: string;
  table: Record<string, unknown>[];
  meta: Record<string, unknown>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  games: () => get<GameConfig[]>("/games/"),
  game: (id: string) => get<GameConfig>(`/games/${id}`),
  draws: (game: string, limit = 50, offset = 0) =>
    get<DrawsResponse>(`/draws/?game=${game}&limit=${limit}&offset=${offset}`),
  coverage: () => get<CoverageRow[]>("/draws/coverage"),
  analysis: (game: string, test: string) =>
    get<AnalysisResult>(`/analysis/${game}/${test}`),
  tests: () => get<{ pick6: string[]; digit: string[] }>("/analysis/tests"),
};
