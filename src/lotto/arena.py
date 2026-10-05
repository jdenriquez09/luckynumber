"""Strategy Arena: walk-forward comparison of ticket-generation strategies.

Every strategy sees only the draws before the one it is scored on. Each draw
gets n tickets per strategy. A strategy's average matches per ticket is tested
against the exact random-play expectation, with a Bonferroni-adjusted p-value
because several strategies are compared at once. Crowd-aware strategies are
also scored on how much of a jackpot they would keep if they won.
"""
from __future__ import annotations

from math import comb, erfc, sqrt
from typing import Callable

import numpy as np
import pandas as pd

from .analysis import AnalysisResult
from .crowd import DEFAULT_CROWD_QUANTILE, CrowdModel, crowd_avoiding_tickets
from .games import GameConfig

MIN_HISTORY = 60
STRATEGY_LABELS = {
    "uniform": "Uniform random",
    "hot": "Hot numbers (last 100 draws)",
    "overdue": "Overdue numbers (longest gap)",
    "ml": "Lean ML scores",
    "crowd": "Crowd-avoiding random",
    "ml_crowd": "Lean ML + crowd-avoiding",
}


def _size(cfg: GameConfig) -> int:
    return cfg.pool - cfg.low_boundary + 1


def clean_history(df: pd.DataFrame, cfg: GameConfig, draw_time: str | None) -> pd.DataFrame:
    cols = [f"n{i}" for i in range(1, cfg.picks + 1)]
    out = df
    if draw_time and "draw_time" in out.columns:
        out = out[out["draw_time"] == draw_time]
    out = out.dropna(subset=cols + ["draw_date"]).copy()
    out["draw_date"] = pd.to_datetime(out["draw_date"], errors="coerce")
    out = out.dropna(subset=["draw_date"])
    if "draw_time" in out.columns:
        out = out.sort_values(["draw_date", "draw_time"], kind="stable")
    else:
        out = out.sort_values("draw_date", kind="stable")
    return out.reset_index(drop=True)


def _numbers(clean: pd.DataFrame, cfg: GameConfig, ordered_columns: bool) -> np.ndarray:
    """(draws, picks) matrix in published order for ordered games, else sorted."""
    if cfg.ordered and "pub_1" in clean.columns and clean["pub_1"].notna().all():
        cols = [f"pub_{i}" for i in range(1, cfg.picks + 1)]
    else:
        cols = [f"n{i}" for i in range(1, cfg.picks + 1)]
    values = clean[cols].astype(int).to_numpy()
    return values if ordered_columns else np.sort(values, axis=1)


def _weight_matrix(cfg: GameConfig, counts: np.ndarray) -> np.ndarray:
    """Positive sampling weights shaped (positions, size)."""
    return np.maximum(np.asarray(counts, dtype=float), 1e-6)


def sample_tickets(cfg: GameConfig, weights: np.ndarray, n: int,
                   rng: np.random.Generator) -> list[list[int]]:
    """n distinct tickets from per-number (or per-position) weights."""
    size = _size(cfg)
    total = size ** cfg.picks if cfg.ordered else comb(size, cfg.picks)
    n = min(n, total)
    weights = np.asarray(weights, dtype=float).reshape(cfg.picks if cfg.ordered else 1, size)
    weights = weights / weights.sum(axis=1, keepdims=True)
    seen: dict[tuple[int, ...], None] = {}
    for _ in range(50):
        m = max(4 * (n - len(seen)), 8)
        if cfg.ordered:
            draws = np.column_stack([rng.choice(size, size=m, p=weights[pos])
                                     for pos in range(cfg.picks)])
        else:
            keys = np.log(weights[0]) + rng.gumbel(size=(m, size))
            draws = np.sort(np.argpartition(-keys, cfg.picks, axis=1)[:, :cfg.picks], axis=1)
        for row in draws:
            seen.setdefault(tuple(int(v) + cfg.low_boundary for v in row))
            if len(seen) >= n:
                return [list(t) for t in seen]
    return [list(t) for t in seen]


def _counts(history: np.ndarray, cfg: GameConfig, last: int) -> np.ndarray:
    size, recent = _size(cfg), history[-last:] - cfg.low_boundary
    if cfg.ordered:
        return np.vstack([np.bincount(recent[:, p], minlength=size) for p in range(cfg.picks)])
    return np.bincount(recent.reshape(-1), minlength=size)[None, :]


def _gaps(history: np.ndarray, cfg: GameConfig, last: int = 400) -> np.ndarray:
    size, recent = _size(cfg), history[-last:] - cfg.low_boundary
    horizon = len(recent)
    rows = []
    for position in range(cfg.picks if cfg.ordered else 1):
        gaps = np.full(size, horizon, dtype=float)
        for value in range(size):
            hit = (recent[:, position] == value) if cfg.ordered else (recent == value).any(axis=1)
            where = np.flatnonzero(hit)
            if where.size:
                gaps[value] = horizon - 1 - where[-1]
        rows.append(gaps)
    return np.vstack(rows)


def _ml_weights(clean: pd.DataFrame, cfg: GameConfig, upto: int, window: int,
                draw_time: str | None) -> np.ndarray:
    from .lean_ml import generate_ml_combinations
    frame = clean.iloc[max(0, upto - window):upto]
    result = generate_ml_combinations(frame, cfg, n=1, draw_time=draw_time, seed=0)
    size = _size(cfg)
    if not result["scores"]:
        return np.ones((cfg.picks if cfg.ordered else 1, size))
    weights = np.ones((cfg.picks if cfg.ordered else 1, size))
    for row in result["scores"]:
        weights[row.get("position", 1) - 1 if cfg.ordered else 0, row["number"] - cfg.low_boundary] = row["score"]
    return weights


def available_strategies(cfg: GameConfig, crowd: CrowdModel | None) -> list[str]:
    names = ["uniform", "hot", "overdue", "ml"]
    if crowd is not None and not cfg.ordered and cfg.id in crowd.game_ids:
        names += ["crowd", "ml_crowd"]
    return names


def _matches(ticket: list[int], actual: np.ndarray, cfg: GameConfig) -> int:
    if cfg.ordered:
        return int(sum(int(a) == int(t) for a, t in zip(actual, ticket)))
    return len(set(ticket) & {int(v) for v in actual})


def _expected_matches(cfg: GameConfig) -> float:
    return 1 / _size(cfg) * cfg.picks if cfg.ordered else cfg.picks ** 2 / cfg.pool


def _normal_two_sided(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2)))


def run_arena(
    df: pd.DataFrame, cfg: GameConfig, strategies: list[str] | None = None,
    test_draws: int = 200, tickets_per_draw: int = 5, draw_time: str | None = None,
    crowd: CrowdModel | None = None, ml_window: int = 400, ml_refit_every: int = 10,
    seed: int = 42, on_progress: Callable[[float], None] | None = None,
) -> AnalysisResult:
    clean = clean_history(df, cfg, draw_time)
    if len(clean) < MIN_HISTORY + 20:
        raise ValueError(f"Need at least {MIN_HISTORY + 20} usable draws; found {len(clean)}.")
    names = strategies or available_strategies(cfg, crowd)
    unknown = set(names) - set(STRATEGY_LABELS)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    if any(s in ("crowd", "ml_crowd") for s in names) and (
            crowd is None or cfg.ordered or cfg.id not in crowd.game_ids):
        raise ValueError("Crowd strategies need a fitted crowd model and a six-number game.")
    ordered_matrix = _numbers(clean, cfg, ordered_columns=True)
    history_matrix = _numbers(clean, cfg, ordered_columns=cfg.ordered)
    start = max(MIN_HISTORY, len(clean) - int(test_draws))
    steps = list(range(start, len(clean)))
    expected = _expected_matches(cfg)
    records: list[dict] = []
    ml_cache: dict[str, np.ndarray] = {}
    for step_index, t in enumerate(steps):
        past = history_matrix[:t]
        actual = ordered_matrix[t] if cfg.ordered else history_matrix[t]
        if "ml" in names or "ml_crowd" in names:
            if step_index % ml_refit_every == 0 or "ml" not in ml_cache:
                ml_cache["ml"] = _ml_weights(clean, cfg, t, ml_window, draw_time)
        for index, name in enumerate(names):
            rng = np.random.default_rng([seed, t, index])
            if name == "uniform":
                tickets = sample_tickets(cfg, np.ones((cfg.picks if cfg.ordered else 1, _size(cfg))),
                                         tickets_per_draw, rng)
            elif name == "hot":
                tickets = sample_tickets(cfg, _weight_matrix(cfg, _counts(past, cfg, 100) + 1),
                                         tickets_per_draw, rng)
            elif name == "overdue":
                tickets = sample_tickets(cfg, _weight_matrix(cfg, _gaps(past, cfg) + 1),
                                         tickets_per_draw, rng)
            elif name == "ml":
                tickets = sample_tickets(cfg, ml_cache["ml"], tickets_per_draw, rng)
            elif name == "crowd":
                tickets = crowd_avoiding_tickets(crowd, cfg, tickets_per_draw, 4000,
                                                 seed=int(rng.integers(2**31)),
                                                 quantile=DEFAULT_CROWD_QUANTILE)
            else:
                tickets = crowd_avoiding_tickets(crowd, cfg, tickets_per_draw, 4000,
                                                 seed=int(rng.integers(2**31)),
                                                 weights=ml_cache["ml"][0],
                                                 quantile=DEFAULT_CROWD_QUANTILE)
            matches = [_matches(ticket, actual, cfg) for ticket in tickets]
            record = {"step": t, "strategy": name, "mean_matches": float(np.mean(matches)),
                      "best_match": max(matches)}
            if crowd is not None and not cfg.ordered and cfg.id in crowd.game_ids:
                record["share_factor"] = float(crowd.share_factor(np.array(tickets), cfg.id).mean())
            records.append(record)
        if on_progress:
            on_progress((step_index + 1) / len(steps))
    per_draw = pd.DataFrame(records)
    per_draw["draw_date"] = clean["draw_date"].to_numpy()[per_draw["step"].to_numpy()]
    rows = []
    hit_level = 2 if cfg.ordered and cfg.picks <= 3 else 3
    for name in names:
        part = per_draw[per_draw["strategy"] == name]
        mean = float(part["mean_matches"].mean())
        sd = float(part["mean_matches"].std(ddof=1)) if len(part) > 1 else float("nan")
        z = (mean - expected) / (sd / sqrt(len(part))) if sd and sd == sd else float("nan")
        p = _normal_two_sided(z) if z == z else float("nan")
        row = {"strategy": STRATEGY_LABELS[name], "key": name, "draws": len(part),
               "mean_matches": mean, "expected_random": expected,
               "edge_pct": (mean / expected - 1) * 100, "z": z, "p_value": p,
               "p_adjusted": min(1.0, p * len(names)) if p == p else float("nan"),
               f"draws_with_{hit_level}plus": float((part["best_match"] >= hit_level).mean())}
        if "share_factor" in part:
            row["jackpot_share_kept"] = float(part["share_factor"].mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("z", ascending=False, ignore_index=True)
    if "jackpot_share_kept" in table:
        base = table.loc[table["key"] == "uniform", "jackpot_share_kept"]
        if len(base):
            table["share_gain_pct"] = (table["jackpot_share_kept"] / float(base.iloc[0]) - 1) * 100
    significant = table[(table["p_adjusted"] < 0.05) & (table["z"] > 0) & (table["key"] != "uniform")]
    if len(significant):
        conclusion = "Strategies beating random after multiple-comparison adjustment: " + ", ".join(
            f"{r.strategy} (+{r.edge_pct:.1f}%, adj p={r.p_adjusted:.3f})" for r in significant.itertuples())
    else:
        top = table.iloc[0]
        conclusion = (f"No strategy beat random after multiple-comparison adjustment; best was "
                      f"{top['strategy']} at {top['edge_pct']:+.2f}% (adj p={top['p_adjusted']:.2f}).")
    return AnalysisResult(
        name="strategy_arena", table=table, conclusion=conclusion,
        meta={"per_draw": per_draw, "test_draws": len(steps), "tickets_per_draw": tickets_per_draw,
              "first_test_date": str(clean["draw_date"].iloc[steps[0]].date()),
              "last_test_date": str(clean["draw_date"].iloc[steps[-1]].date()),
              "expected_matches": expected, "ml_window": ml_window,
              "ml_refit_every": ml_refit_every, "seed": seed},
    )


def next_draw_tickets(
    df: pd.DataFrame, cfg: GameConfig, strategy: str, n: int = 5,
    draw_time: str | None = None, crowd: CrowdModel | None = None,
    ml_window: int = 400, seed: int = 42,
) -> list[list[int]]:
    """Tickets from one strategy for the upcoming draw, using all history."""
    clean = clean_history(df, cfg, draw_time)
    if len(clean) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} usable draws; found {len(clean)}.")
    matrix = _numbers(clean, cfg, ordered_columns=cfg.ordered)
    # Each strategy gets its own random stream. Sharing one seed shares the sampling
    # noise, which makes mildly weighted strategies return almost the same tickets.
    stream = list(STRATEGY_LABELS).index(strategy) if strategy in STRATEGY_LABELS else 0
    rng = np.random.default_rng([seed, stream])
    uniform = np.ones((cfg.picks if cfg.ordered else 1, _size(cfg)))
    if strategy == "uniform":
        return sample_tickets(cfg, uniform, n, rng)
    if strategy == "hot":
        return sample_tickets(cfg, _weight_matrix(cfg, _counts(matrix, cfg, 100) + 1), n, rng)
    if strategy == "overdue":
        return sample_tickets(cfg, _weight_matrix(cfg, _gaps(matrix, cfg) + 1), n, rng)
    ml = _ml_weights(clean, cfg, len(clean), ml_window, draw_time)
    if strategy == "ml":
        return sample_tickets(cfg, ml, n, rng)
    if strategy in ("crowd", "ml_crowd") and crowd is not None and not cfg.ordered:
        return crowd_avoiding_tickets(crowd, cfg, n, 20000, seed=int(rng.integers(2**31)),
                                      weights=ml[0] if strategy == "ml_crowd" else None,
                                      quantile=DEFAULT_CROWD_QUANTILE)
    raise ValueError(f"Strategy {strategy!r} is unavailable for {cfg.name}.")


def ai_shortlist(
    df: pd.DataFrame, cfg: GameConfig, crowd: CrowdModel, n_candidates: int = 40,
    max_overlap: int = 4, draw_time: str | None = None, ml_window: int = 400, seed: int = 42,
) -> dict:
    """A diverse menu of locally generated candidates for the AI to choose from.

    Candidates come from the ML-weighted, least-crowded pool. Each carries its
    crowd index, jackpot share kept and average ML support, and the menu comes
    with per-number context (ML score, recent frequency, draws since seen).
    """
    if crowd is None or cfg.ordered or cfg.id not in crowd.game_ids:
        raise ValueError(f"AI shortlist is unavailable for {cfg.name}.")
    clean = clean_history(df, cfg, draw_time)
    if len(clean) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} usable draws; found {len(clean)}.")
    matrix = _numbers(clean, cfg, ordered_columns=False)
    ml = _ml_weights(clean, cfg, len(clean), ml_window, draw_time)[0]
    ml = ml / ml.mean()
    tickets = crowd_avoiding_tickets(crowd, cfg, n_candidates, 20000, seed=seed,
                                     weights=ml, max_overlap=max_overlap,
                                     quantile=DEFAULT_CROWD_QUANTILE)
    array = np.array(tickets)
    candidates = [
        {"ticket": ticket, "crowd_index": float(ci), "share_kept": float(sf),
         "ml_support": float(ml[np.array(ticket) - cfg.low_boundary].mean())}
        for ticket, ci, sf in zip(tickets, crowd.crowd_index(array, cfg.id),
                                  crowd.share_factor(array, cfg.id))
    ]
    counts, gaps = _counts(matrix, cfg, 100)[0], _gaps(matrix, cfg)[0]
    numbers = sorted({v for ticket in tickets for v in ticket})
    context = {v: {"ml_vs_average": float(ml[v - cfg.low_boundary]),
                   "freq_last100": int(counts[v - cfg.low_boundary]),
                   "draws_since_seen": int(gaps[v - cfg.low_boundary])} for v in numbers}
    return {"candidates": candidates, "number_context": context, "max_overlap": max_overlap}
