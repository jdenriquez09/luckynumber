"""
Backtesting framework — simulates a betting strategy against historical draws.

For each draw from MIN_HISTORY onwards, a combination is generated using ONLY
data available before that draw, then compared to the actual result.
This gives an honest measure of how a statistical strategy would have performed.

IMPORTANT: This framework will always show a net loss. Lotto draws are
independent random events — past data cannot predict future draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .games import GameConfig
from .predict import suggest_combinations

TICKET_PRICE_PHP = 20.0
MIN_HISTORY = 30  # draws needed before first prediction

# Fixed prize amounts per number-match tier (below jackpot). Source: PCSO.
FIXED_PRIZES: dict[str, dict[int, float]] = {
    "658": {5: 150_000, 4: 2_000, 3: 40},
    "655": {5: 100_000, 4: 1_500, 3: 30},
    "649": {5: 60_000,  4: 1_000, 3: 20},
    "645": {5: 25_000,  4: 500,   3: 20},
    "642": {5: 14_000,  4: 200,   3: 20},
    "6d":  {5: 20_000,  4: 800,   3: 50},
    "4d":  {3: 800,     2: 100},
    "3d":  {2: 100},
    "2d":  {},
}


def _prize_for_match(game_id: str, picks: int, matches: int, jackpot: float) -> float:
    if matches == picks:
        return jackpot
    return FIXED_PRIZES.get(game_id, {}).get(matches, 0.0)


def run_backtest(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    window: int | None = None,
    draw_time: str | None = None,
    seed: int = 42,
    max_rows: int = 200,
) -> dict:
    """
    Simulate a composite-score betting strategy against all historical draws.

    Parameters
    ----------
    df          : full draws DataFrame sorted ascending by draw_date
    game_cfg    : game configuration
    window      : rolling window for heat-map scoring (None = use all history)
    draw_time   : filter to a specific draw time for multi-draw games
    seed        : RNG seed for reproducibility
    max_rows    : how many per-draw result rows to return (most recent)

    Returns
    -------
    {
        "summary": { total_draws_tested, total_spent, total_won, net, roi_pct,
                     match_distribution, avg_matches, exact_matches },
        "results": [ {draw_date, predicted, actual, matches, prize_php}, ... ]
    }
    """
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    df = df.sort_values("draw_date").reset_index(drop=True)

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time].reset_index(drop=True)

    df_clean = df.dropna(subset=num_cols).reset_index(drop=True)

    if len(df_clean) < MIN_HISTORY + 1:
        return {"summary": {}, "results": [], "error": f"Need at least {MIN_HISTORY + 1} draws."}

    results: list[dict] = []
    total_spent = 0.0
    total_won = 0.0

    for i in range(MIN_HISTORY, len(df_clean)):
        history = df_clean.iloc[:i].copy()
        actual_row = df_clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in num_cols if pd.notna(actual_row[c]))

        try:
            combos = suggest_combinations(
                history, game_cfg, n=1, window=window,
                seed=seed + i, draw_time=None,
            )
        except Exception:
            continue

        if not combos:
            continue

        predicted = set(combos[0])
        matches = len(predicted & actual_nums)
        jackpot = float(actual_row.get("jackpot_php") or 0)
        prize = _prize_for_match(game_cfg.id, game_cfg.picks, matches, jackpot)

        total_spent += TICKET_PRICE_PHP
        total_won += prize

        results.append({
            "draw_date": str(actual_row["draw_date"])[:10],
            "predicted": sorted(predicted),
            "actual": sorted(actual_nums),
            "matches": matches,
            "prize_php": prize,
        })

    if not results:
        return {"summary": {}, "results": []}

    match_series = pd.Series([r["matches"] for r in results])
    match_dist = match_series.value_counts().sort_index().to_dict()

    summary = {
        "total_draws_tested": len(results),
        "total_spent_php": total_spent,
        "total_won_php": total_won,
        "net_php": total_won - total_spent,
        "roi_pct": (total_won - total_spent) / total_spent * 100,
        "avg_matches_per_draw": float(match_series.mean()),
        "match_distribution": {int(k): int(v) for k, v in match_dist.items()},
        "exact_matches": int(match_dist.get(game_cfg.picks, 0)),
        "best_match": int(match_series.max()),
    }

    return {"summary": summary, "results": results[-max_rows:]}
