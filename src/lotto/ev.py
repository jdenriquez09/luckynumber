"""
Expected Value (EV) calculator for PCSO lotto games.

EV is always negative for lottery games — this shows you by exactly how much
per ticket and what jackpot would be needed to break even.

Also provides multi-game comparison: which game gives the best (least bad)
expected value given current jackpot sizes.
"""
from __future__ import annotations

from math import comb
from .games import GameConfig, GAMES

TICKET_PRICE_PHP = 20.0

# Minimum advertised jackpot per game (PHP). Source: PCSO.
MIN_JACKPOTS: dict[str, float] = {
    "658": 49_500_000,
    "655": 29_700_000,
    "649": 16_000_000,
    "645": 8_000_000,
    "642": 6_000_000,
    "6d":  500_000,
    "4d":  10_000,
    "3d":  4_500,
    "2d":  4_000,
}

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

# Ticket price per game (PHP)
TICKET_PRICES: dict[str, float] = {
    "658": 20, "655": 20, "649": 20, "645": 20, "642": 20,
    "6d": 20, "4d": 20, "3d": 12, "2d": 10,
}


def _hypergeometric_prob(pool_size: int, picks: int, match: int) -> float:
    """P(exactly `match` correct numbers when drawing `picks` from `pool_size`)."""
    try:
        return comb(picks, match) * comb(pool_size - picks, picks - match) / comb(pool_size, picks)
    except (ValueError, ZeroDivisionError):
        return 0.0


def expected_value(
    game_cfg: GameConfig,
    jackpot_php: float | None = None,
    ticket_price: float | None = None,
) -> dict:
    """
    Compute full EV breakdown for one ticket of a given game.

    Returns
    -------
    {
        game, game_name, jackpot_php, ticket_price_php,
        total_ev_php, net_ev_php, return_pct,
        breakeven_jackpot_php,
        tiers: [ {matches, odds_str, probability, prize_php, ev_contribution_php} ]
    }
    """
    pool_size = game_cfg.pool - game_cfg.low_boundary + 1
    picks = game_cfg.picks
    jp = jackpot_php if jackpot_php is not None else MIN_JACKPOTS.get(game_cfg.id, 0)
    price = ticket_price if ticket_price is not None else TICKET_PRICES.get(game_cfg.id, TICKET_PRICE_PHP)
    fixed = FIXED_PRIZES.get(game_cfg.id, {})

    # Build prize tiers: jackpot + all fixed prizes
    tier_prizes: dict[int, float] = {picks: jp}
    tier_prizes.update(fixed)

    tiers = []
    total_ev = 0.0
    jackpot_ev = 0.0

    for match_count in sorted(tier_prizes, reverse=True):
        prize = tier_prizes[match_count]
        prob = _hypergeometric_prob(pool_size, picks, match_count)
        ev_contrib = prob * prize
        total_ev += ev_contrib
        if match_count == picks:
            jackpot_ev = ev_contrib

        tiers.append({
            "matches": match_count,
            "odds_str": f"1 in {round(1 / prob):,}" if prob > 0 else "N/A",
            "probability": prob,
            "prize_php": prize,
            "ev_contribution_php": ev_contrib,
        })

    net_ev = total_ev - price
    non_jackpot_ev = total_ev - jackpot_ev
    jackpot_prob = _hypergeometric_prob(pool_size, picks, picks)
    breakeven_jp = (price - non_jackpot_ev) / jackpot_prob if jackpot_prob > 0 else float("inf")

    return {
        "game": game_cfg.id,
        "game_name": game_cfg.name,
        "jackpot_php": jp,
        "ticket_price_php": price,
        "total_ev_php": total_ev,
        "net_ev_php": net_ev,
        "return_pct": total_ev / price * 100 if price else 0,
        "breakeven_jackpot_php": breakeven_jp,
        "tiers": tiers,
    }


def multi_game_ev(jackpots: dict[str, float] | None = None) -> list[dict]:
    """
    Compare EV across all games given current jackpot amounts.

    Parameters
    ----------
    jackpots : {game_id: current_jackpot_php} — None uses minimum jackpot.

    Returns list sorted by net EV descending (least bad first).
    """
    results = []
    for gid, gcfg in GAMES.items():
        jp = (jackpots or {}).get(gid)
        ev = expected_value(gcfg, jackpot_php=jp)
        results.append(ev)
    return sorted(results, key=lambda x: x["net_ev_php"], reverse=True)


def jackpots_from_db(conn) -> dict[str, float]:
    """Query most recent jackpot_php per game from the DB."""
    try:
        df = conn.execute("""
            SELECT game, jackpot_php
            FROM draws
            WHERE jackpot_php IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game ORDER BY draw_date DESC) = 1
        """).fetchdf()
        return dict(zip(df["game"].tolist(), df["jackpot_php"].tolist()))
    except Exception:
        return {}
