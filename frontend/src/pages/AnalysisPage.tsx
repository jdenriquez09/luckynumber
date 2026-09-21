import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

const PICK6_TESTS = [
  { id: "freq_chi_square", label: "Frequency (chi-square)" },
  { id: "distribution_tests", label: "Distribution Tests" },
  { id: "sequence_tests", label: "Sequence Tests" },
  { id: "gap_analysis", label: "Gap Analysis" },
  { id: "monte_carlo", label: "Monte Carlo" },
  { id: "jackpot_dynamics", label: "Jackpot Dynamics" },
  { id: "date_effects", label: "Date Effects" },
];

const DIGIT_TESTS = [
  { id: "digit_uniformity", label: "Digit Uniformity" },
  { id: "digit_repeat_rate", label: "Digit Repeat Rate" },
];

export default function AnalysisPage() {
  const { data: games } = useQuery({ queryKey: ["games"], queryFn: api.games });
  const [gameId, setGameId] = useState("655");
  const [testId, setTestId] = useState("freq_chi_square");

  const selectedGame = (games ?? []).find((g) => g.id === gameId);
  const tests = selectedGame?.ordered ? DIGIT_TESTS : PICK6_TESTS;

  const { data: result, isLoading, isError, error } = useQuery({
    queryKey: ["analysis", gameId, testId],
    queryFn: () => api.analysis(gameId, testId),
    enabled: !!gameId && !!testId,
  });

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Statistical Analysis</h1>

      <div className="flex gap-4 mb-8 flex-wrap">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Game</label>
          <select
            className="border border-slate-300 rounded px-3 py-2 text-sm bg-white"
            value={gameId}
            onChange={(e) => { setGameId(e.target.value); setTestId(tests[0].id); }}
          >
            {(games ?? []).map((g) => (
              <option key={g.id} value={g.id}>{g.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Test</label>
          <select
            className="border border-slate-300 rounded px-3 py-2 text-sm bg-white"
            value={testId}
            onChange={(e) => setTestId(e.target.value)}
          >
            {tests.map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </select>
        </div>
      </div>

      {isLoading && <p className="text-slate-500">Running analysis…</p>}

      {isError && (
        <div className="bg-red-50 border border-red-200 rounded p-4 text-red-700 text-sm">
          {(error as Error).message}
        </div>
      )}

      {result && (
        <div>
          <div className="bg-blue-50 border border-blue-200 rounded p-4 mb-6">
            <p className="text-sm text-blue-900">{result.conclusion}</p>
          </div>

          {result.table.length > 0 && (
            <div className="overflow-x-auto rounded border border-slate-200">
              <table className="text-sm w-full">
                <thead className="bg-slate-100">
                  <tr>
                    {Object.keys(result.table[0]).map((k) => (
                      <th key={k} className="px-3 py-2 text-left font-medium text-slate-600">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.table.slice(0, 100).map((row, i) => (
                    <tr key={i} className="border-t border-slate-100 hover:bg-slate-50">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-3 py-2">
                          {v === null ? "—" : String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.table.length > 100 && (
                <p className="text-xs text-slate-400 px-3 py-2">
                  Showing 100 of {result.table.length} rows.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
