import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from "recharts";
import { api } from "../lib/api";

export default function GamePage() {
  const { id } = useParams<{ id: string }>();
  const gameId = id ?? "";

  const { data: game } = useQuery({
    queryKey: ["game", gameId],
    queryFn: () => api.game(gameId),
  });

  const { data: freq, isLoading: freqLoading } = useQuery({
    queryKey: ["analysis", gameId, "freq_chi_square"],
    queryFn: () => api.analysis(gameId, "freq_chi_square"),
    enabled: !!game && !game.ordered,
  });

  const { data: draws, isLoading: drawsLoading } = useQuery({
    queryKey: ["draws", gameId],
    queryFn: () => api.draws(gameId, 20),
  });

  if (!game) return <p className="text-slate-500">Loading…</p>;

  const expected = freq?.table[0]
    ? (freq.table[0] as { expected: number }).expected
    : null;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-1">{game.name}</h1>
      <p className="text-slate-500 text-sm mb-6">
        {game.ordered
          ? `${game.picks}-digit ordered game · pool 0–${game.pool}`
          : `Pick ${game.picks} from ${game.low_boundary ?? 1}–${game.pool}`}
        {" · "}Draw days: {game.draw_days.join(", ")}
      </p>

      {!game.ordered && (
        <section className="mb-10">
          <h2 className="text-lg font-semibold mb-3">Number Frequency</h2>
          {freqLoading && <p className="text-slate-400 text-sm">Computing…</p>}
          {freq && (
            <>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={freq.table as object[]} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="number" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="observed" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                  {expected !== null && (
                    <ReferenceLine y={expected} stroke="#ef4444" strokeDasharray="4 2"
                      label={{ value: `Exp ${expected}`, position: "right", fontSize: 11 }} />
                  )}
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-slate-600 mt-2 bg-slate-50 rounded p-3">
                {freq.conclusion}
              </p>
            </>
          )}
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-3">Recent Draws</h2>
        {drawsLoading && <p className="text-slate-400 text-sm">Loading…</p>}
        {draws && (
          <div className="overflow-x-auto rounded border border-slate-200">
            <table className="text-sm w-full">
              <thead className="bg-slate-100">
                <tr>
                  <th className="px-3 py-2 text-left">Date</th>
                  <th className="px-3 py-2 text-left">Numbers</th>
                  <th className="px-3 py-2 text-right">Jackpot (₱)</th>
                  <th className="px-3 py-2 text-left">Notes</th>
                </tr>
              </thead>
              <tbody>
                {draws.data.map((row) => (
                  <tr key={`${row.draw_date}-${row.draw_time}`} className="border-t border-slate-100 hover:bg-slate-50">
                    <td className="px-3 py-2 whitespace-nowrap">{row.draw_date}</td>
                    <td className="px-3 py-2 font-mono">
                      {[row.n1, row.n2, row.n3, row.n4, row.n5, row.n6]
                        .filter((n) => n !== null)
                        .join(" – ")}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {row.jackpot_php ? row.jackpot_php.toLocaleString() : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-amber-700">{row.notes ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
