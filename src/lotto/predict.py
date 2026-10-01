"""
Statistical probability scoring for PCSO lotto historical draws.

IMPORTANT: These scores reflect historical draw frequencies ONLY.
Lotto draws are independent random events. Past draws cannot predict future draws.
This module does NOT claim predictive power — it measures historical patterns only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .games import GameConfig

DISCLAIMER = (
    "These scores reflect historical draw frequencies only. "
    "Lotto draws are statistically independent random events. "
    "A number that appeared more or less frequently in the past "
    "is NOT more or less likely to appear in the future. "
    "This analysis does not constitute gambling advice."
)


def _get_number_cols(game_cfg: GameConfig) -> list[str]:
    return [f"n{i}" for i in range(1, game_cfg.picks + 1)]


def _clean_draws(df: pd.DataFrame, game_cfg: GameConfig) -> pd.DataFrame:
    cols = _get_number_cols(game_cfg)
    clean = df.dropna(subset=cols).copy()
    sort_cols = [c for c in ("draw_date", "draw_time") if c in clean.columns]
    if sort_cols:
        clean = clean.sort_values(sort_cols, kind="stable")
    return clean.reset_index(drop=True)


def number_heat_map(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    window: int | None = None,
    draw_time: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-number statistical scores from historical draw data.

    Parameters
    ----------
    df       : draws DataFrame sorted by draw_date ascending
    game_cfg : game configuration
    window   : if set, only use the last `window` draws for 'recent' frequency;
               None means use all draws for both overall and recent counts

    Returns
    -------
    DataFrame with columns:
        number, total_draws, expected_draws, recent_draws,
        heat_score, current_gap, expected_gap, gap_score, composite_score
    """
    if draw_time is not None and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean_draws(df, game_cfg)
    if clean.empty:
        return pd.DataFrame()

    cols = _get_number_cols(game_cfg)
    n_draws = len(clean)
    pool_size = game_cfg.pool - game_cfg.low_boundary + 1  # actual number of distinct values
    k = game_cfg.picks

    # Expected appearances per number over n_draws draws
    expected_per_draw = k / pool_size
    expected_total = n_draws * expected_per_draw

    # All numbers in pool
    all_nums = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    # Total frequency across all draws (safe for nullable Int64 columns)
    all_raw = pd.array(clean[cols].values.flatten(), dtype=object)
    all_flat = np.array([int(v) for v in all_raw if v is not None and not pd.isna(v)])
    total_counts = pd.Series(all_flat).value_counts()

    # Recent frequency — last `window` draws
    if window is not None and window < n_draws:
        recent_df = clean.tail(window)
    else:
        recent_df = clean

    n_recent = len(recent_df)
    recent_raw = pd.array(recent_df[cols].values.flatten(), dtype=object)
    recent_flat = np.array([int(v) for v in recent_raw if v is not None and not pd.isna(v)])
    recent_counts = pd.Series(recent_flat).value_counts()
    expected_recent = n_recent * expected_per_draw

    # Current gap: how many draws since each number last appeared
    # Build a reverse index: for each draw (latest = index 0), record which numbers appeared
    clean_reset = clean.reset_index(drop=True)
    current_gaps: dict[int, int] = {}
    for num in all_nums:
        # Find the last row (most recent) where this number appears
        mask = (clean_reset[cols] == num).any(axis=1)
        found_indices = mask[mask].index
        if len(found_indices) == 0:
            current_gaps[num] = n_draws  # never appeared
        else:
            last_idx = found_indices[-1]
            current_gaps[num] = n_draws - 1 - last_idx  # draws since last appearance

    expected_gap = pool_size / k  # geometric distribution mean

    rows = []
    for num in all_nums:
        tc = int(total_counts.get(num, 0))
        rc = int(recent_counts.get(num, 0))
        cg = current_gaps[num]

        # Heat score: ratio of recent frequency to expected (clamped)
        heat = (rc / max(expected_recent, 1e-9)) if expected_recent > 0 else 1.0

        # Gap score: overdue numbers score high, recently-drawn score low
        gap_s = min(cg / expected_gap, 3.0) / 3.0

        # Composite: average of heat and gap scores, then normalise across numbers later
        composite_raw = 0.5 * heat + 0.5 * gap_s

        rows.append({
            "number": num,
            "total_draws": tc,
            "expected_draws": round(expected_total, 1),
            "recent_draws": rc,
            "heat_score": round(heat, 4),
            "current_gap": cg,
            "expected_gap": round(expected_gap, 1),
            "gap_score": round(gap_s, 4),
            "_composite_raw": composite_raw,
        })

    result = pd.DataFrame(rows)

    # Normalise composite to [0, 1]
    c_min = result["_composite_raw"].min()
    c_max = result["_composite_raw"].max()
    if c_max > c_min:
        result["composite_score"] = (
            (result["_composite_raw"] - c_min) / (c_max - c_min)
        ).round(4)
    else:
        result["composite_score"] = 0.5

    result = result.drop(columns=["_composite_raw"])
    return result.sort_values("composite_score", ascending=False).reset_index(drop=True)


def random_combinations(
    game_cfg: GameConfig,
    n: int = 5,
    seed: int | None = None,
) -> list[list[int]]:
    """Uniform random tickets, with no historical or model weighting.

    Ordered games preserve positions and allow repeats. Unordered tickets have
    distinct numbers and are sorted. Duplicate tickets are excluded; requesting
    more than the finite ticket space returns that space, in random order.
    A supplied seed makes the output reproducible, not predictive.
    """
    from math import comb
    from numbers import Integral

    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    size = game_cfg.pool - game_cfg.low_boundary + 1
    if size < 1 or game_cfg.picks < 1 or (not game_cfg.ordered and game_cfg.picks > size):
        raise ValueError("Invalid game pool or pick count")
    total = size ** game_cfg.picks if game_cfg.ordered else comb(size, game_cfg.picks)
    target = min(n, total)
    rng = np.random.default_rng(seed)
    pool = np.arange(game_cfg.low_boundary, game_cfg.pool + 1)
    seen: set[tuple[int, ...]] = set()
    result: list[list[int]] = []
    while len(result) < target:
        chosen = rng.choice(pool, size=game_cfg.picks, replace=game_cfg.ordered)
        ticket = tuple(int(x) for x in (chosen if game_cfg.ordered else np.sort(chosen)))
        if ticket not in seen:
            seen.add(ticket)
            result.append(list(ticket))
    return result


def suggest_combinations(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n: int = 5,
    window: int | None = None,
    seed: int | None = None,
    draw_time: str | None = None,
) -> list[list[int]]:
    """Compatibility wrapper for uniform sampling, stable for a data snapshot.

    History, window and draw time affect only the default reproducibility seed;
    they never weight or exclude numbers. Empty filtered history returns no tickets.
    """
    if window is not None and window < 1:
        raise ValueError("window must be positive")
    if draw_time is not None and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    if df.empty:
        return []
    clean = _clean_draws(df, game_cfg)
    if window is not None:
        clean = clean.tail(window)
    if clean.empty:
        return []
    if seed is None:
        import hashlib
        cols = [c for c in ("draw_date", "draw_time", *_get_number_cols(game_cfg)) if c in clean]
        snapshot = pd.util.hash_pandas_object(clean[cols], index=False).values.tobytes()
        digest = hashlib.sha256(f"uniform-v1|{game_cfg.id}|{window}|{draw_time}".encode() + snapshot)
        seed = int.from_bytes(digest.digest()[:8], "big")
    return random_combinations(game_cfg, n=n, seed=seed)


def position_heat_map(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    draw_time: str | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Per-position frequency analysis for ordered games (2D, 3D, 4D, 6D).
    Returns a dict keyed by 1-based position index.
    Each DataFrame has columns: digit, count, frequency_pct, expected_pct, deviation_z.
    """
    if draw_time is not None and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean_draws(df, game_cfg)
    if clean.empty:
        return {}

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    expected_pct = 100.0 / pool_size
    n = len(clean)

    result: dict[int, pd.DataFrame] = {}
    # Use pub_* columns for ordered games (preserve draw order / position)
    pub_cols = [f"pub_{i}" for i in range(1, game_cfg.picks + 1)]
    use_pub = all(c in clean.columns for c in pub_cols)
    num_cols = pub_cols if use_pub else [f"n{i}" for i in range(1, game_cfg.picks + 1)]

    for pos_idx, col in enumerate(num_cols, 1):
        col_data = clean[col].dropna().astype(int)
        counts = col_data.value_counts().reindex(pool, fill_value=0)
        freq_pct = counts / n * 100
        # z-score: (obs - exp) / sqrt(n * p * (1-p)) where p = 1/pool_size
        p_exp = 1.0 / pool_size
        std_err = np.sqrt(n * p_exp * (1 - p_exp))
        z_scores = (counts.values - n * p_exp) / std_err if std_err > 0 else np.zeros(pool_size)
        result[pos_idx] = pd.DataFrame({
            "digit": pool,
            "count": counts.values.astype(int),
            "frequency_pct": np.round(freq_pct.values, 2),
            "expected_pct": round(expected_pct, 2),
            "deviation_z": np.round(z_scores, 3),
        })

    return result


def heat_map_summary(heat: pd.DataFrame, top_n: int = 5) -> str:
    """
    Return a plain-text summary of hot, cold, and overdue numbers
    for use as context in the Azure OpenAI prompt.
    """
    if heat.empty:
        return "No data available."

    hot = heat.nlargest(top_n, "heat_score")["number"].tolist()
    cold = heat.nsmallest(top_n, "heat_score")["number"].tolist()
    overdue = heat.nlargest(top_n, "current_gap")["number"].tolist()

    return (
        f"Top {top_n} hottest numbers (highest recent frequency): {hot}\n"
        f"Top {top_n} coldest numbers (lowest recent frequency): {cold}\n"
        f"Top {top_n} most overdue (longest gap since last draw): {overdue}"
    )


# ===========================================================================
# Features 6 & 7 — Combination Quality Scorer + Your Numbers Analyzer
# ===========================================================================

def combination_quality(
    combo: list,
    game_cfg,
    df,
) -> dict:
    """
    Feature 6 — Score any combination on multiple statistical dimensions.

    Returns a dict with:
        sum_value, sum_percentile, odd_count, even_count, high_count, low_count,
        spread, consecutive_pairs, heat_score_avg, quality_score (0-100),
        verdict (Excellent / Good / Average / Below Average / Poor)
    """
    import pandas as pd
    import numpy as np

    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    mid = (game_cfg.low_boundary + game_cfg.pool) / 2.0

    nums = sorted(int(n) for n in combo)

    # --- sum percentile ---
    df_clean = df.dropna(subset=num_cols)
    all_sums = df_clean[num_cols].sum(axis=1)
    combo_sum = sum(nums)
    sum_pct = float((all_sums < combo_sum).mean() * 100) if len(all_sums) else 50.0

    # --- odd / even ---
    odd_count = sum(1 for n in nums if n % 2 == 1)
    even_count = len(nums) - odd_count

    # --- high / low ---
    high_count = sum(1 for n in nums if n > mid)
    low_count = len(nums) - high_count

    # --- spread (max - min) ---
    spread = nums[-1] - nums[0] if len(nums) > 1 else 0
    max_spread = game_cfg.pool - game_cfg.low_boundary

    # --- consecutive pairs ---
    consec_pairs = sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)

    # --- heat score average ---
    heat_avg = 0.0
    try:
        heat = number_heat_map(df, game_cfg)
        if not heat.empty:
            scores = heat.set_index("number")["composite_score"].reindex(nums).fillna(0.5)
            heat_avg = float(scores.mean())
    except Exception:
        heat_avg = 0.5

    # --- quality score (0-100) ---
    # Ideal: sum near median (sum_pct ~50), balanced odd/even, balanced high/low,
    #        good spread (>50% of max), few consecutive pairs, above-average heat
    picks = game_cfg.picks
    ideal_each = picks / 2.0
    odd_bal = 1 - abs(odd_count - ideal_each) / ideal_each if ideal_each else 1
    high_bal = 1 - abs(high_count - ideal_each) / ideal_each if ideal_each else 1
    sum_score = 1 - abs(sum_pct - 50) / 50
    spread_score = spread / max_spread if max_spread else 1
    consec_penalty = max(0, 1 - consec_pairs / max(1, picks - 1))
    heat_component = heat_avg

    quality_score = (
        odd_bal * 20 +
        high_bal * 20 +
        sum_score * 20 +
        spread_score * 20 +
        consec_penalty * 10 +
        heat_component * 10
    )
    quality_score = round(min(100, max(0, quality_score)), 1)

    if quality_score >= 80:
        verdict = "Excellent"
    elif quality_score >= 65:
        verdict = "Good"
    elif quality_score >= 50:
        verdict = "Average"
    elif quality_score >= 35:
        verdict = "Below Average"
    else:
        verdict = "Poor"

    return {
        "combination": nums,
        "sum_value": combo_sum,
        "sum_percentile": round(sum_pct, 1),
        "odd_count": odd_count,
        "even_count": even_count,
        "high_count": high_count,
        "low_count": low_count,
        "spread": spread,
        "consecutive_pairs": consec_pairs,
        "heat_score_avg": round(heat_avg, 3),
        "quality_score": quality_score,
        "verdict": verdict,
    }


def score_user_combination(
    combo: list,
    game_cfg,
    df,
) -> dict:
    """
    Feature 7 — "Your Numbers" analyzer: same as combination_quality but framed
    for user-facing display with plain-language commentary per dimension.
    """
    result = combination_quality(combo, game_cfg, df)

    lines = []
    picks = game_cfg.picks
    sum_pct = result["sum_percentile"]
    if sum_pct < 25:
        lines.append(f"Draw sum {result['sum_value']} is in the lowest 25% — unusually low total.")
    elif sum_pct > 75:
        lines.append(f"Draw sum {result['sum_value']} is in the highest 25% — unusually high total.")
    else:
        lines.append(f"Draw sum {result['sum_value']} is near the historical median — balanced range.")

    odd, even = result["odd_count"], result["even_count"]
    ideal = picks // 2
    if abs(odd - ideal) <= 1:
        lines.append(f"Odd/even split {odd}/{even} is well balanced.")
    else:
        lines.append(f"Odd/even split {odd}/{even} is skewed — historically most draws are more balanced.")

    high, low = result["high_count"], result["low_count"]
    if abs(high - ideal) <= 1:
        lines.append(f"High/low split {high}/{low} is well balanced.")
    else:
        lines.append(f"High/low split {high}/{low} is skewed — consider a more even spread.")

    sp = result["spread"]
    max_sp = game_cfg.pool - game_cfg.low_boundary
    if sp < max_sp * 0.4:
        lines.append(f"Spread of {sp} is narrow — all numbers are clustered close together.")
    elif sp >= max_sp * 0.7:
        lines.append(f"Spread of {sp} covers a wide range of the pool — good distribution.")
    else:
        lines.append(f"Spread of {sp} is moderate.")

    cp = result["consecutive_pairs"]
    if cp == 0:
        lines.append("No consecutive number pairs — no extra penalty.")
    elif cp == 1:
        lines.append("One consecutive pair — slight clustering.")
    else:
        lines.append(f"{cp} consecutive pairs — heavily clustered numbers, rarely seen together.")

    result["commentary"] = lines
    return result


# ===========================================================================
# Feature: Combination History Lookup
# ===========================================================================

def find_combination_history(
    combo: list,
    game_cfg,
    df,
) -> dict:
    """
    Check whether a combination has ever been drawn before.

    For unordered games (pick-6, 4D, 6D): compares sorted number sets.
    For ordered games (2D, 3D):           compares published digit order (pub_1..pub_n).

    Returns
    -------
    {
        "combo":        sorted list of numbers queried,
        "found":        bool,
        "times_drawn":  int,
        "draws":        DataFrame with columns [draw_date, draw_time, jackpot_php, notes]
                        (one row per matching draw, sorted ascending),
        "last_drawn":   str date or None,
        "first_drawn":  str date or None,
    }
    """
    import pandas as pd
    import numpy as np

    picks = game_cfg.picks
    nums = sorted(int(n) for n in combo)

    if game_cfg.ordered:
        # Use published columns (position matters for 2D/3D)
        pub_cols = [f"pub_{i}" for i in range(1, picks + 1)]
        use_cols = pub_cols if all(c in df.columns for c in pub_cols) else [f"n{i}" for i in range(1, picks + 1)]
        df_c = df.dropna(subset=use_cols).copy()
        def _matches(row):
            return [int(row[c]) for c in use_cols] == nums
    else:
        # Sorted set match for unordered games
        num_cols = [f"n{i}" for i in range(1, picks + 1)]
        df_c = df.dropna(subset=num_cols).copy()
        def _matches(row):
            vals = sorted(int(row[c]) for c in num_cols)
            return vals == nums

    mask = df_c.apply(_matches, axis=1)
    hits = df_c[mask].copy()

    if hits.empty:
        return {
            "combo": nums, "found": False, "times_drawn": 0,
            "draws": pd.DataFrame(), "last_drawn": None, "first_drawn": None,
        }

    hits = hits.sort_values("draw_date")
    keep_cols = ["draw_date"]
    if "draw_time" in hits.columns:
        keep_cols.append("draw_time")
    if "jackpot_php" in hits.columns:
        keep_cols.append("jackpot_php")
    if "notes" in hits.columns:
        keep_cols.append("notes")

    result_df = hits[keep_cols].copy()
    result_df["draw_date"] = result_df["draw_date"].astype(str).str[:10]

    return {
        "combo": nums,
        "found": True,
        "times_drawn": len(result_df),
        "draws": result_df.reset_index(drop=True),
        "first_drawn": result_df["draw_date"].iloc[0],
        "last_drawn": result_df["draw_date"].iloc[-1],
    }
