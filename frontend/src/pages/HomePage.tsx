import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, CoverageRow, GameConfig } from "../lib/api";

function CoverageCard({ row, game }: { row: CoverageRow; game: GameConfig }) {
  return (
    <Link
      to={`/game/${row.game}`}
      className="block bg-white rounded-lg border border-slate-200 p-5 hover:shadow-md transition-shadow"
    >
      <h2 className="font-semibold text-lg text-blue-700">{game.name}</h2>
      <p className="text-sm text-slate-500 mt-1">
        {game.ordered ? "Ordered digits" : `Pick ${game.picks} from 1–${game.pool}`}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-slate-400">Draws</span>
          <p className="font-bold text-slate-800">{row.draws.toLocaleString()}</p>
        </div>
        <div>
          <span className="text-slate-400">Last draw</span>
          <p className="font-bold text-slate-800">{row.last_draw}</p>
        </div>
      </div>
    </Link>
  );
}

export default function HomePage() {
  const { data: games } = useQuery({ queryKey: ["games"], queryFn: api.games });
  const { data: coverage, isLoading } = useQuery({
    queryKey: ["coverage"],
    queryFn: api.coverage,
  });

  const gameMap = Object.fromEntries((games ?? []).map((g) => [g.id, g]));

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">PCSO Lotto Statistical Analysis</h1>
      <p className="text-slate-600 mb-8 max-w-2xl">
        Historical draw results analysed for statistical consistency with a fair random process.
        This site does <strong>not</strong> predict future draws or suggest number picks.
      </p>

      {isLoading && <p className="text-slate-500">Loading coverage data…</p>}

      {coverage && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {coverage.map((row) =>
            gameMap[row.game] ? (
              <CoverageCard key={row.game} row={row} game={gameMap[row.game]} />
            ) : null,
          )}
        </div>
      )}

      {coverage?.length === 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4 text-yellow-800">
          No draw data yet. Run <code className="font-mono text-sm">python -m lotto scrape --game 655</code> to load data.
        </div>
      )}
    </div>
  );
}
