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
    return df.dropna(subset=cols).copy()


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

    # Total frequency across all draws
    all_flat = clean[cols].values.flatten().astype(float)
    all_flat = all_flat[~np.isnan(all_flat)].astype(int)
    total_counts = pd.Series(all_flat).value_counts()

    # Recent frequency — last `window` draws
    if window is not None and window < n_draws:
        recent_df = clean.tail(window)
    else:
        recent_df = clean

    n_recent = len(recent_df)
    recent_flat = recent_df[cols].values.flatten().astype(float)
    recent_flat = recent_flat[~np.isnan(recent_flat)].astype(int)
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

        # Gap score: expected_gap / current_gap — high = overdue, low = recently appeared
        gap_s = expected_gap / max(cg, 1)

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


def suggest_combinations(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n: int = 5,
    window: int | None = None,
    seed: int | None = None,
    draw_time: str | None = None,
) -> list[list[int]]:
    """
    Generate n combinations by weighted random sampling (without replacement per combo).
    Weight = composite_score clamped to >= 0.01 so every number has non-zero probability.
    Returns each combination sorted ascending.
    """
    rng = np.random.default_rng(seed)

    # Ordered games (2D/3D/4D/6D): sample each position independently
    if game_cfg.ordered:
        pos_heat = position_heat_map(df, game_cfg, draw_time=draw_time)
        if not pos_heat:
            return []
        seen: set[tuple] = set()
        combos: list[list[int]] = []
        max_attempts = n * 20
        attempts = 0
        while len(combos) < n and attempts < max_attempts:
            attempts += 1
            digits = []
            for pos in range(1, game_cfg.picks + 1):
                ph = pos_heat[pos]
                w = np.clip(ph["deviation_z"].values + 2.0, 0.01, None)
                w = w / w.sum()
                digits.append(int(rng.choice(ph["digit"].values, p=w)))
            key = tuple(digits)
            if key not in seen:
                seen.add(key)
                combos.append(list(digits))
        return combos

    heat = number_heat_map(df, game_cfg, window=window, draw_time=draw_time)
    if heat.empty:
        return []

    numbers = heat["number"].values
    weights = np.clip(heat["composite_score"].values, 0.01, None)
    weights = weights / weights.sum()

    seen_s: set[frozenset] = set()
    combos_s: list[list[int]] = []
    max_attempts = n * 20
    attempts = 0
    while len(combos_s) < n and attempts < max_attempts:
        attempts += 1
        chosen = rng.choice(numbers, size=game_cfg.picks, replace=False, p=weights)
        combo = sorted(int(x) for x in chosen)
        key = frozenset(combo)
        if key not in seen_s:
            seen_s.add(key)
            combos_s.append(combo)

    return combos_s


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
