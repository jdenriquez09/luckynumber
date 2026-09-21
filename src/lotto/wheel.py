"""
Wheeling system and syndicate ticket generator for PCSO lotto.

Full wheel     : all C(n_keys, picks) combinations of selected key numbers.
Abbreviated    : greedy reduced set that still guarantees a minimum match level.
Syndicate      : diversified tickets with minimum overlap — maximum pool coverage
                 for a group of players splitting the cost.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from .games import GameConfig

TICKET_PRICE_PHP = 20.0


# ---------------------------------------------------------------------------
# Full wheel
# ---------------------------------------------------------------------------

def full_wheel(key_numbers: list[int], game_cfg: GameConfig) -> list[list[int]]:
    """Generate every C(len(key_numbers), picks) combination from key numbers."""
    keys = sorted(set(key_numbers))
    if len(keys) < game_cfg.picks:
        raise ValueError(f"Need at least {game_cfg.picks} key numbers, got {len(keys)}.")
    return [sorted(c) for c in combinations(keys, game_cfg.picks)]


def full_wheel_cost(n_keys: int, game_cfg: GameConfig) -> dict:
    """Return ticket count and cost for a full wheel without generating combos."""
    n_tickets = comb(n_keys, game_cfg.picks)
    return {
        "n_keys": n_keys,
        "picks": game_cfg.picks,
        "n_tickets": n_tickets,
        "cost_php": n_tickets * TICKET_PRICE_PHP,
    }


# ---------------------------------------------------------------------------
# Abbreviated wheel (greedy coverage)
# ---------------------------------------------------------------------------

def abbreviated_wheel(
    key_numbers: list[int],
    game_cfg: GameConfig,
    guarantee: int = 4,
) -> list[list[int]]:
    """
    Generate a reduced ticket set that guarantees `guarantee` matches if at
    least `guarantee` of the key numbers appear in the actual draw.

    Uses a greedy set-cover approach: add tickets that cover the most
    uncovered guarantee-subsets first.
    """
    keys = sorted(set(key_numbers))
    if len(keys) < game_cfg.picks:
        raise ValueError(f"Need at least {game_cfg.picks} key numbers.")
    guarantee = min(guarantee, game_cfg.picks)

    all_tickets = list(combinations(keys, game_cfg.picks))
    target_subsets = set(combinations(keys, guarantee))
    covered: set = set()
    selected: list[list[int]] = []

    remaining = list(all_tickets)
    while covered < target_subsets and remaining:
        best_ticket = max(
            remaining,
            key=lambda t: len(set(combinations(sorted(t), guarantee)) - covered),
        )
        new_covers = set(combinations(sorted(best_ticket), guarantee)) - covered
        if not new_covers:
            break
        covered |= new_covers
        selected.append(sorted(best_ticket))
        remaining.remove(best_ticket)

    return selected


# ---------------------------------------------------------------------------
# Syndicate optimizer
# ---------------------------------------------------------------------------

def syndicate_tickets(
    game_cfg: GameConfig,
    n_tickets: int,
    df: pd.DataFrame | None = None,
    window: int | None = None,
    draw_time: str | None = None,
    seed: int = 42,
) -> list[list[int]]:
    """
    Generate n_tickets combinations with minimum overlap between them.

    If df is provided, uses composite heat scores as base weights so that
    the syndicate still leans toward statistically active numbers while
    maximising pool coverage. Otherwise samples uniformly.
    """
    rng = np.random.default_rng(seed)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    picks = game_cfg.picks

    base_weights: np.ndarray | None = None
    if df is not None and not df.empty:
        from .predict import number_heat_map
        heat = number_heat_map(df, game_cfg, window=window, draw_time=draw_time)
        if not heat.empty:
            w = heat.set_index("number")["composite_score"].reindex(pool).fillna(0.01).values
            base_weights = np.clip(w.astype(float), 0.01, None)

    tickets: list[list[int]] = []
    number_usage = np.ones(len(pool), dtype=float)
    max_attempts = n_tickets * 100

    for _ in range(max_attempts):
        if len(tickets) >= n_tickets:
            break

        # Diversity penalty: down-weight numbers already used often
        usage_inv = 1.0 / number_usage
        if base_weights is not None:
            w = base_weights * usage_inv
        else:
            w = usage_inv
        w = w / w.sum()

        chosen = sorted(int(x) for x in rng.choice(pool, size=picks, replace=False, p=w))

        # Reject if overlap with any existing ticket exceeds half the picks
        max_allowed_overlap = picks // 2
        if any(len(set(chosen) & set(t)) > max_allowed_overlap for t in tickets):
            continue

        tickets.append(chosen)
        for n in chosen:
            number_usage[pool.index(n)] += 1

    return tickets


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------

def wheel_stats(tickets: list[list[int]], game_cfg: GameConfig) -> dict:
    """Coverage and overlap statistics for any set of tickets."""
    if not tickets:
        return {}

    pool = set(range(game_cfg.low_boundary, game_cfg.pool + 1))
    covered_numbers = set()
    covered_pairs: set = set()

    for t in tickets:
        covered_numbers.update(t)
        covered_pairs.update(combinations(sorted(t), 2))

    total_pairs = comb(len(pool), 2)

    overlaps = [
        len(set(tickets[i]) & set(tickets[j]))
        for i in range(len(tickets))
        for j in range(i + 1, len(tickets))
    ] if len(tickets) > 1 else [0]

    return {
        "n_tickets": len(tickets),
        "cost_php": len(tickets) * TICKET_PRICE_PHP,
        "numbers_covered": sorted(covered_numbers),
        "pool_coverage_pct": round(len(covered_numbers) / len(pool) * 100, 1),
        "pairs_covered": len(covered_pairs),
        "total_possible_pairs": total_pairs,
        "pair_coverage_pct": round(len(covered_pairs) / total_pairs * 100, 2),
        "avg_inter_ticket_overlap": round(float(np.mean(overlaps)), 2),
        "max_inter_ticket_overlap": int(np.max(overlaps)),
    }
