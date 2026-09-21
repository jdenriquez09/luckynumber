"""
Statistical analysis functions for PCSO lotto draws.

Every public function returns an AnalysisResult with:
  - table: pd.DataFrame of key statistics
  - conclusion: plain-English finding with honest uncertainty
  - meta: dict of raw values (p-values, stats) for programmatic use

No function claims predictive power. Results are tests of fairness only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .games import GameConfig

rng = np.random.default_rng(seed=42)


@dataclass
class AnalysisResult:
    name: str
    table: pd.DataFrame
    conclusion: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_draws(df: pd.DataFrame, game_cfg: GameConfig) -> pd.DataFrame:
    """Return only rows with all numbers present."""
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    return df.dropna(subset=num_cols).copy()


def _all_numbers(df: pd.DataFrame, game_cfg: GameConfig) -> np.ndarray:
    cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    return df[cols].values.flatten().astype(int)


# ---------------------------------------------------------------------------
# 1. Frequency chi-square (pick-6)
# ---------------------------------------------------------------------------

def freq_chi_square(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg)
    n_draws = len(df)
    nums = _all_numbers(df, game_cfg)

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    observed = np.array([np.sum(nums == n) for n in pool])
    expected_freq = n_draws * game_cfg.picks / len(pool)
    expected = np.full(len(pool), expected_freq)

    chi2, p = stats.chisquare(observed, f_exp=expected)

    table = pd.DataFrame({
        "number": pool,
        "observed": observed,
        "expected": np.round(expected, 2),
        "deviation": np.round((observed - expected) / np.sqrt(expected), 3),
    })

    if p < 0.05:
        conclusion = (
            f"Chi-square test: χ²={chi2:.2f}, df={len(pool)-1}, p={p:.4f}. "
            f"The frequency distribution departs significantly from uniform at α=0.05 "
            f"(n={n_draws} draws). This warrants closer inspection but does not indicate "
            f"predictability — sampling variation alone can produce such results."
        )
    else:
        conclusion = (
            f"Chi-square test: χ²={chi2:.2f}, df={len(pool)-1}, p={p:.4f}. "
            f"No significant departure from a uniform distribution detected "
            f"(n={n_draws} draws, α=0.05). Results are consistent with a fair draw."
        )

    return AnalysisResult(
        name="freq_chi_square",
        table=table,
        conclusion=conclusion,
        meta={"chi2": chi2, "p": p, "n_draws": n_draws},
    )


# ---------------------------------------------------------------------------
# 2. Distribution tests: sum, odd/even, low/high, consecutive pairs
# ---------------------------------------------------------------------------

def distribution_tests(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg)
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    matrix = df[num_cols].values.astype(int)
    n = len(matrix)

    # Draw sum
    draw_sums = matrix.sum(axis=1)
    pool = np.arange(game_cfg.low_boundary, game_cfg.pool + 1)
    # Expected mean and std of sum of k uniform draws without replacement (hypergeometric moments)
    N = len(pool)
    k = game_cfg.picks
    mu_sum = k * (game_cfg.low_boundary + game_cfg.pool) / 2
    var_sum = k * (N - k) / (N - 1) * np.var(pool, ddof=0)
    z_sum = (draw_sums.mean() - mu_sum) / np.sqrt(var_sum / n)
    p_sum = 2 * stats.norm.sf(abs(z_sum))

    # Odd/even
    odd_counts = (matrix % 2 == 1).sum(axis=1)
    _, p_odd = stats.chisquare(np.bincount(odd_counts, minlength=k + 1))

    # Low/high (split at midpoint)
    mid = (game_cfg.low_boundary + game_cfg.pool) / 2
    low_counts = (matrix <= mid).sum(axis=1)
    _, p_low = stats.chisquare(np.bincount(low_counts, minlength=k + 1))

    # Consecutive pairs
    def count_consec(row: np.ndarray) -> int:
        s = np.sort(row)
        return int(np.sum(np.diff(s) == 1))

    consec = np.array([count_consec(r) for r in matrix])
    mean_consec = consec.mean()

    rows = [
        {"test": "draw_sum", "observed_mean": round(draw_sums.mean(), 2),
         "expected_mean": round(mu_sum, 2), "p_value": round(p_sum, 4)},
        {"test": "odd_count_distribution", "observed_mean": round(odd_counts.mean(), 2),
         "expected_mean": round(k / 2, 2), "p_value": round(p_odd, 4)},
        {"test": "low_count_distribution", "observed_mean": round(low_counts.mean(), 2),
         "expected_mean": round(k / 2, 2), "p_value": round(p_low, 4)},
        {"test": "consecutive_pairs_mean", "observed_mean": round(mean_consec, 3),
         "expected_mean": None, "p_value": None},
    ]
    table = pd.DataFrame(rows)

    sig = [r for r in rows if r["p_value"] is not None and r["p_value"] < 0.05]
    if sig:
        tests_str = ", ".join(r["test"] for r in sig)
        conclusion = (
            f"Distribution tests (n={n}): {tests_str} show p<0.05. "
            f"Apply Bonferroni correction (α/4=0.0125) before concluding non-randomness."
        )
    else:
        conclusion = (
            f"Distribution tests (n={n}): sum, odd/even, low/high all consistent "
            f"with fair random draws (all p≥0.05)."
        )

    return AnalysisResult(name="distribution_tests", table=table, conclusion=conclusion,
                          meta={"n": n})


# ---------------------------------------------------------------------------
# 3. Pair co-occurrence with FDR correction
# ---------------------------------------------------------------------------

def pair_cooccurrence(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    from itertools import combinations

    df = _load_draws(df, game_cfg)
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    n = len(df)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    N = len(pool)
    k = game_cfg.picks

    # Expected co-occurrence probability for any pair
    # P(both drawn) = C(N-2, k-2) / C(N, k)
    from math import comb
    p_expected = comb(N - 2, k - 2) / comb(N, k)

    # Count observed co-occurrences (vectorized: iterate numpy rows, not pandas rows)
    from collections import Counter
    arr = df[num_cols].values.astype(int)
    pair_counts: Counter = Counter()
    for row in arr:
        pair_counts.update(combinations(sorted(row), 2))

    records = []
    p_values = []
    for (a, b), count in pair_counts.items():
        # Binomial test
        result = stats.binomtest(count, n, p_expected)
        p_values.append(result.pvalue)
        records.append({"num_a": a, "num_b": b, "observed": count,
                         "expected": round(n * p_expected, 2), "p_raw": round(result.pvalue, 5)})

    # FDR (Benjamini-Hochberg)
    if p_values:
        from statsmodels.stats.multitest import multipletests
        reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
        for i, rec in enumerate(records):
            rec["p_fdr"] = round(p_adj[i], 5)
            rec["significant"] = bool(reject[i])

    table = pd.DataFrame(records).sort_values("p_fdr" if records else "p_raw")
    sig_count = table["significant"].sum() if "significant" in table.columns else 0

    conclusion = (
        f"Pair co-occurrence (n={n}, {len(records)} pairs tested, FDR-BH corrected): "
        f"{sig_count} pairs significant after correction. "
        + ("This suggests some pairs co-occur more/less than expected. Inspect further."
           if sig_count > 0 else
           "No pairs are significant after FDR correction. Consistent with fair draws.")
    )

    return AnalysisResult(name="pair_cooccurrence", table=table, conclusion=conclusion,
                          meta={"n": n, "sig_count": int(sig_count)})


# ---------------------------------------------------------------------------
# 4. Gap analysis per number (geometric expectation)
# ---------------------------------------------------------------------------

def gap_analysis(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg).sort_values("draw_date")
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    n = len(df)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    N = len(pool)
    k = game_cfg.picks
    p_draw = k / N  # probability a specific number is drawn on any given draw

    records = []
    for num in pool:
        appearances = df[num_cols].apply(lambda row: num in row.values, axis=1)
        idx = np.where(appearances.values)[0]
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        mean_gap = gaps.mean()
        expected_gap = 1 / p_draw
        # KS test vs geometric
        ks_stat, ks_p = stats.kstest(gaps, "geom", args=(p_draw,))
        current_gap = n - 1 - idx[-1]
        records.append({
            "number": num,
            "appearances": len(idx),
            "mean_gap": round(mean_gap, 1),
            "expected_gap": round(expected_gap, 1),
            "current_gap": current_gap,
            "ks_p": round(ks_p, 4),
        })

    table = pd.DataFrame(records).sort_values("ks_p")
    sig = (table["ks_p"] < 0.05).sum() if not table.empty else 0

    conclusion = (
        f"Gap analysis (n={n} draws, {len(records)} numbers): "
        f"{sig} numbers have gaps that deviate from the geometric distribution at p<0.05. "
        f"A 'long absence' (large current_gap) is NOT evidence a number is 'overdue' — "
        f"under fair random draws, past gaps carry no information about future draws."
    )

    return AnalysisResult(name="gap_analysis", table=table, conclusion=conclusion,
                          meta={"n": n, "p_draw": p_draw})


# ---------------------------------------------------------------------------
# 5. Sequence tests: runs, autocorrelation, lag-1 overlap
# ---------------------------------------------------------------------------

def sequence_tests(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg).sort_values("draw_date")
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    n = len(df)

    records = []

    # Wald-Wolfowitz runs test on the median-split of draw sums
    draw_sums = df[num_cols].sum(axis=1).values
    median_sum = np.median(draw_sums)
    binary = (draw_sums > median_sum).astype(int)
    n1, n2 = binary.sum(), (1 - binary).sum()
    runs = 1 + np.sum(np.diff(binary) != 0)
    # Normal approximation
    mu_r = 2 * n1 * n2 / n + 1
    var_r = (mu_r - 1) * (mu_r - 2) / (n - 1)
    z_r = (runs - mu_r) / np.sqrt(var_r) if var_r > 0 else 0
    p_runs = float(2 * stats.norm.sf(abs(z_r)))
    records.append({"test": "wald_wolfowitz_runs", "statistic": round(float(z_r), 3),
                    "p_value": round(p_runs, 4), "n": n})

    # Lag-1 autocorrelation of draw sums
    if n > 10:
        corr, p_corr = stats.pearsonr(draw_sums[:-1], draw_sums[1:])
        records.append({"test": "lag1_autocorr_sum", "statistic": round(float(corr), 4),
                        "p_value": round(float(p_corr), 4), "n": n - 1})

    # Lag-1 set overlap (how many numbers repeat draw-to-draw)
    matrix = df[num_cols].values.astype(int)
    overlaps = [
        len(set(matrix[i]) & set(matrix[i + 1]))
        for i in range(len(matrix) - 1)
    ]
    # Hypergeometric expected overlap
    N = len(range(game_cfg.low_boundary, game_cfg.pool + 1))
    k = game_cfg.picks
    from math import comb
    exp_overlap = k * k / N  # approx; exact = k^2 / N for large N
    t_stat, p_overlap = stats.ttest_1samp(overlaps, popmean=exp_overlap)
    records.append({"test": "lag1_set_overlap", "statistic": round(float(t_stat), 3),
                    "p_value": round(float(p_overlap), 4), "n": len(overlaps)})

    table = pd.DataFrame(records)
    sig = (table["p_value"] < 0.05).sum()

    conclusion = (
        f"Sequence tests (n={n}): {sig}/3 tests significant at p<0.05. "
        + ("Some sequential structure detected — apply multiple-comparison correction "
           f"(α/3≈0.017) before interpreting."
           if sig > 0 else
           "No significant sequential patterns. Results consistent with independence.")
    )

    return AnalysisResult(name="sequence_tests", table=table, conclusion=conclusion,
                          meta={"n": n})


# ---------------------------------------------------------------------------
# 6. Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_sim: int = 10_000,
) -> AnalysisResult:
    df = _load_draws(df, game_cfg)
    n = len(df)
    pool = np.arange(game_cfg.low_boundary, game_cfg.pool + 1)
    k = game_cfg.picks

    real_sums = df[[f"n{i}" for i in range(1, k + 1)]].sum(axis=1).values
    real_mean = real_sums.mean()
    real_std = real_sums.std()

    # Each simulated draw picks k distinct numbers from pool (without replacement per draw)
    pool_tiled = np.tile(pool, (n, 1))
    sim_means = np.array([
        rng.permuted(pool_tiled, axis=1)[:, :k].sum(axis=1).mean()
        for _ in range(n_sim)
    ])

    percentile = float(np.mean(sim_means <= real_mean) * 100)

    table = pd.DataFrame({
        "metric": ["real_mean_sum", "sim_mean_sum", "sim_std", "percentile"],
        "value": [
            round(real_mean, 2),
            round(sim_means.mean(), 2),
            round(sim_means.std(), 4),
            round(percentile, 1),
        ],
    })

    conclusion = (
        f"Monte Carlo (n_sim={n_sim}, n_draws={n}): "
        f"the real mean draw sum ({real_mean:.2f}) sits at the "
        f"{percentile:.1f}th percentile of simulated fair histories. "
        + ("This is within the expected range for a fair process."
           if 2.5 <= percentile <= 97.5 else
           f"This is in the tail of the simulated distribution (outside the 95% CI), "
           f"suggesting a possible deviation from uniformity.")
    )

    return AnalysisResult(name="monte_carlo", table=table, conclusion=conclusion,
                          meta={"n": n, "n_sim": n_sim, "percentile": percentile})


# ---------------------------------------------------------------------------
# 7. Jackpot dynamics
# ---------------------------------------------------------------------------

def jackpot_dynamics(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg).sort_values("draw_date").copy()
    df["jackpot_php"] = pd.to_numeric(df.get("jackpot_php"), errors="coerce")

    # Detect resets (jackpot drops significantly = winner)
    jp = df["jackpot_php"].dropna()
    if len(jp) < 2:
        return AnalysisResult(
            name="jackpot_dynamics",
            table=pd.DataFrame(),
            conclusion="Insufficient jackpot data.",
        )

    resets = jp.diff() < 0
    n_winners = resets.sum()
    total_draws = len(jp)
    winner_rate = n_winners / total_draws

    rollover_lengths: list[int] = []
    cur_run = 0
    for reset in resets:
        if reset:
            rollover_lengths.append(cur_run)
            cur_run = 0
        else:
            cur_run += 1

    mean_rollover = np.mean(rollover_lengths) if rollover_lengths else 0

    table = pd.DataFrame({
        "metric": ["total_draws", "jackpot_wins", "winner_rate_pct", "mean_rollover_draws"],
        "value": [total_draws, n_winners, round(winner_rate * 100, 2), round(mean_rollover, 1)],
    })

    conclusion = (
        f"Jackpot dynamics (n={total_draws} draws with jackpot data): "
        f"{n_winners} wins detected ({winner_rate*100:.1f}% of draws), "
        f"mean rollover streak {mean_rollover:.1f} draws."
    )

    return AnalysisResult(name="jackpot_dynamics", table=table, conclusion=conclusion,
                          meta={"n_winners": int(n_winners), "winner_rate": winner_rate})


# ---------------------------------------------------------------------------
# 8. Date effects (weekday/month rolling stability)
# ---------------------------------------------------------------------------

def date_effects(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    df = _load_draws(df, game_cfg).copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df["month"] = df["draw_date"].dt.month
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    df["draw_sum"] = df[num_cols].sum(axis=1)

    records = []
    for label, col in [("month", "month"), ("weekday", "weekday")]:
        if col not in df.columns:
            continue
        groups = df.groupby(col)["draw_sum"]
        f_stat, p_val = stats.f_oneway(*[g.values for _, g in groups if len(g) > 1])
        records.append({
            "factor": label,
            "groups": df[col].nunique(),
            "F": round(float(f_stat), 3),
            "p_value": round(float(p_val), 4),
        })

    table = pd.DataFrame(records)
    sig = (table["p_value"] < 0.05).sum() if not table.empty else 0

    conclusion = (
        f"Date effects (ANOVA on draw sum by weekday/month): "
        + (f"{sig} factor(s) significant at p<0.05. Investigate for possible systematic bias."
           if sig > 0 else
           "No significant weekday or monthly bias detected.")
    )

    return AnalysisResult(name="date_effects", table=table, conclusion=conclusion, meta={})


# ---------------------------------------------------------------------------
# Digit-game analyses
# ---------------------------------------------------------------------------

def rolling_frequency(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    window: int = 50,
    draw_time: str | None = None,
) -> pd.DataFrame:
    """
    Rolling frequency of each number over a sliding window of draws.
    Returns long-format DataFrame: draw_date, number, rolling_count.
    """
    clean = _load_draws(df, game_cfg).sort_values("draw_date").reset_index(drop=True)
    if draw_time and "draw_time" in clean.columns:
        clean = clean[clean["draw_time"] == draw_time].reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame(columns=["draw_date", "number", "rolling_count"])

    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    dates = clean["draw_date"].values
    matrix = clean[num_cols].values.astype(int)
    n = len(matrix)

    records = []
    for i in range(window - 1, n):
        block = matrix[max(0, i - window + 1): i + 1]
        flat = block.flatten()
        for num in pool:
            records.append({
                "draw_date": dates[i],
                "number": num,
                "rolling_count": int(np.sum(flat == num)),
            })

    return pd.DataFrame(records)


def draw_time_comparison(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """
    Chi-square contingency test: do number frequencies differ across draw times?
    Only meaningful for games with multiple draw times (2D, 3D).
    """
    if len(game_cfg.draw_times) < 2:
        return AnalysisResult(
            name="draw_time_comparison",
            table=pd.DataFrame(),
            conclusion="This game has only one draw time — comparison not applicable.",
        )

    clean = _load_draws(df, game_cfg)
    if "draw_time" not in clean.columns:
        return AnalysisResult(
            name="draw_time_comparison",
            table=pd.DataFrame(),
            conclusion="draw_time column not found in data.",
        )

    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    draw_times = sorted(clean["draw_time"].dropna().unique())
    _LABELS = {"14:00": "2:00 PM", "17:00": "5:00 PM", "21:00": "9:00 PM"}

    # Build frequency table: rows = numbers, columns = draw times
    freq_table = pd.DataFrame(index=pool, columns=draw_times, dtype=int).fillna(0)
    for dt in draw_times:
        sub = clean[clean["draw_time"] == dt]
        flat = sub[num_cols].values.flatten().astype(int)
        for num, cnt in zip(*np.unique(flat, return_counts=True)):
            if num in freq_table.index:
                freq_table.loc[num, dt] = int(cnt)

    chi2, p, dof, _ = stats.chi2_contingency(freq_table.values)
    n = len(clean)

    table = freq_table.reset_index().rename(columns={"index": "number"})
    table.columns = ["number"] + [_LABELS.get(dt, dt) for dt in draw_times]

    conclusion = (
        f"Draw-time comparison chi-square (n={n} draws, {len(draw_times)} slots): "
        f"χ²={chi2:.2f}, df={dof}, p={p:.4f}. "
        + ("Number frequency distributions differ significantly across draw times at p<0.05. "
           "Each draw time has a distinct historical frequency profile."
           if p < 0.05 else
           "No significant difference in number frequencies across draw times. "
           "Each draw time slot appears statistically uniform.")
    )

    return AnalysisResult(
        name="draw_time_comparison",
        table=table,
        conclusion=conclusion,
        meta={"chi2": chi2, "p": p, "dof": dof, "n": n},
    )


def digit_uniformity(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Per-position digit uniformity test for ordered digit games."""
    df = _load_draws(df, game_cfg)
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    n = len(df)

    records = []
    for pos, col in enumerate(num_cols, 1):
        counts = df[col].value_counts().reindex(pool, fill_value=0)
        chi2, p = stats.chisquare(counts.values)
        records.append({
            "position": pos,
            "chi2": round(float(chi2), 2),
            "p_value": round(float(p), 4),
            "most_frequent": int(counts.idxmax()),
            "least_frequent": int(counts.idxmin()),
        })

    table = pd.DataFrame(records)
    sig = (table["p_value"] < 0.05).sum()

    conclusion = (
        f"Digit uniformity (n={n} draws, {game_cfg.picks} positions): "
        f"{sig} position(s) deviate from uniform at p<0.05. "
        + ("Apply Bonferroni correction." if sig > 0 else "All positions consistent with uniform.")
    )

    return AnalysisResult(name="digit_uniformity", table=table, conclusion=conclusion,
                          meta={"n": n})


def digit_repeat_rate(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Rate at which the same digit repeats in adjacent positions."""
    df = _load_draws(df, game_cfg)
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    matrix = df[num_cols].values.astype(int)

    adjacent_repeats = np.sum(np.diff(matrix, axis=1) == 0, axis=1)
    repeat_rate = adjacent_repeats.mean() / (game_cfg.picks - 1)
    expected_rate = 1 / (game_cfg.pool - game_cfg.low_boundary + 1)

    t_stat, p_val = stats.ttest_1samp(adjacent_repeats / (game_cfg.picks - 1),
                                       popmean=expected_rate)

    table = pd.DataFrame({
        "metric": ["observed_adjacent_repeat_rate", "expected_rate", "t_stat", "p_value"],
        "value": [round(repeat_rate, 4), round(expected_rate, 4),
                  round(float(t_stat), 3), round(float(p_val), 4)],
    })

    conclusion = (
        f"Adjacent repeat rate: {repeat_rate:.4f} observed vs {expected_rate:.4f} expected. "
        f"t={t_stat:.2f}, p={p_val:.4f}. "
        + ("Significant — repeated adjacent digits occur more/less than chance."
           if p_val < 0.05 else
           "No significant deviation from expected repeat rate.")
    )

    return AnalysisResult(name="digit_repeat_rate", table=table, conclusion=conclusion,
                          meta={"n": len(df)})


# ===========================================================================
# Features 2, 3, 4, 5, 8, 9, 14, 15
# ===========================================================================

def structural_break(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 2 — Chi-square on first-half vs second-half frequency distribution."""
    from scipy.stats import chi2_contingency
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    df = df.sort_values("draw_date").reset_index(drop=True)
    clean = df.dropna(subset=num_cols)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    mid = len(clean) // 2
    first = clean.iloc[:mid]
    second = clean.iloc[mid:]

    def _freq(part):
        _raw = pd.array(part[num_cols].values.flatten(), dtype=object)
        nums = np.array([x for x in _raw if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=int)
        return pd.Series(nums).value_counts().reindex(pool, fill_value=0).values

    f1, f2 = _freq(first), _freq(second)
    # chi2_contingency requires no all-zero columns; keep only numbers seen in either half
    _mask = (f1 + f2) > 0
    f1_fit, f2_fit = f1[_mask], f2[_mask]
    if f1_fit.sum() == 0 or f2_fit.sum() == 0 or len(f1_fit) < 2:
        chi2, p, dof = 0.0, 1.0, 0
    else:
        chi2, p, dof, _ = chi2_contingency(np.array([f1_fit, f2_fit]))
    ratio = (f2 / (f2.sum() + 1e-9)) / (f1 / (f1.sum() + 1e-9) + 1e-9)
    mid_date = str(clean.iloc[mid]["draw_date"])[:10]

    rows = [{"number": n, "freq_first_half": int(f1[i]),
              "freq_second_half": int(f2[i]),
              "change_ratio": round(float(ratio[i]), 3)}
            for i, n in enumerate(pool)]

    conclusion = (
        f"Comparing first {mid} vs last {len(clean)-mid} draws (split at {mid_date}): "
        f"chi2={chi2:.2f}, p={p:.4f}. "
        + ("Significant shift detected — frequencies appear to have changed over time."
           if p < 0.05 else "No significant structural break detected.")
    )
    return AnalysisResult(
        name="Structural Break", table=pd.DataFrame(rows), conclusion=conclusion,
        meta={"chi2": chi2, "p": p, "dof": dof, "mid_date": mid_date,
              "n_first": mid, "n_second": len(clean) - mid}
    )


def entropy_score(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 3 — Shannon entropy vs maximum possible (100% = perfectly random)."""
    from scipy.stats import entropy as scipy_entropy
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)

    all_nums = pd.array(df[num_cols].values.flatten(), dtype=object)
    all_nums = np.array([x for x in all_nums if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=int)
    counts = pd.Series(all_nums).value_counts().reindex(pool, fill_value=0)
    probs = counts / counts.sum()

    obs_entropy = float(scipy_entropy(probs.values + 1e-12, base=2))
    max_entropy = float(np.log2(pool_size))
    entropy_pct = obs_entropy / max_entropy * 100

    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["draw_date"]).dt.year
    year_rows = []
    for year, grp in df2.groupby("year"):
        _yr = pd.array(grp[num_cols].values.flatten(), dtype=object)
        yn = np.array([x for x in _yr if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=int)
        yc = pd.Series(yn).value_counts().reindex(pool, fill_value=0)
        yp = yc / yc.sum()
        ye = float(scipy_entropy(yp.values + 1e-12, base=2))
        year_rows.append({"year": int(year), "entropy_bits": round(ye, 4),
                           "max_bits": round(max_entropy, 4),
                           "pct_of_max": round(ye / max_entropy * 100, 2)})

    conclusion = (
        f"Overall entropy: {obs_entropy:.4f} / {max_entropy:.4f} bits "
        f"({entropy_pct:.1f}% of maximum). "
        + ("Close to maximum randomness — consistent with a fair lottery."
           if entropy_pct >= 95
           else f"Shows {100 - entropy_pct:.1f}% less entropy than a perfectly random process.")
    )
    return AnalysisResult(
        name="Entropy / Randomness", table=pd.DataFrame(year_rows),
        conclusion=conclusion,
        meta={"obs_entropy": obs_entropy, "max_entropy": max_entropy,
              "entropy_pct": entropy_pct, "pool_size": pool_size}
    )


def power_analysis(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 4 — 95% Wilson CIs on number frequencies; minimum detectable bias."""
    from math import sqrt
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    k = game_cfg.picks
    n = len(df.dropna(subset=num_cols))

    all_nums = pd.array(df[num_cols].values.flatten(), dtype=object)
    all_nums = np.array([x for x in all_nums if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=int)
    counts = pd.Series(all_nums).value_counts().reindex(pool, fill_value=0)
    total_obs = int(counts.sum())
    expected_p = k / pool_size
    expected_n = n * expected_p
    z = 1.96

    rows = []
    for num in pool:
        cnt = int(counts[num])
        p_hat = cnt / total_obs
        denom = 1 + z ** 2 / total_obs
        centre = (p_hat + z ** 2 / (2 * total_obs)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / total_obs
                             + z ** 2 / (4 * total_obs ** 2)) / denom
        ci_lo = max(0.0, centre - margin) * total_obs
        ci_hi = min(1.0, centre + margin) * total_obs
        rows.append({"number": num, "observed": cnt, "expected": round(expected_n, 1),
                      "ci_95_lo": round(ci_lo, 1), "ci_95_hi": round(ci_hi, 1),
                      "includes_expected": ci_lo <= expected_n <= ci_hi})

    se = sqrt(expected_p * (1 - expected_p) / n) if n else 0
    min_det_pct = 1.96 * se / expected_p * 100 if expected_p else 0
    draws_for_1pct = int(((1.96 + 0.84) / (0.01 / sqrt(expected_p * (1 - expected_p)))) ** 2) if expected_p else 0
    n_outside = sum(1 for r in rows if not r["includes_expected"])

    conclusion = (
        f"With {n} draws, minimum detectable frequency bias: {min_det_pct:.1f}%. "
        f"{n_outside}/{pool_size} numbers outside their 95% CI. "
        f"Need ~{draws_for_1pct:,} draws to detect a 1% bias reliably."
    )
    return AnalysisResult(
        name="Power Analysis", table=pd.DataFrame(rows), conclusion=conclusion,
        meta={"n_draws": n, "min_detectable_pct": min_det_pct,
              "draws_for_1pct": draws_for_1pct, "n_outside_ci": n_outside}
    )


def cross_game_correlation(
    dfs: dict,
    game_cfgs: dict,
) -> pd.DataFrame:
    """Feature 5 — Spearman correlation between game daily draw-sum series."""
    from scipy.stats import spearmanr
    sum_series: dict = {}
    for gid, df_g in dfs.items():
        gcfg = game_cfgs.get(gid)
        if gcfg is None or df_g.empty:
            continue
        num_cols = [f"n{i}" for i in range(1, gcfg.picks + 1)]
        df_g = df_g.dropna(subset=num_cols).copy()
        df_g["draw_date"] = pd.to_datetime(df_g["draw_date"]).dt.date
        df_g["_sum"] = df_g[num_cols].sum(axis=1)
        sum_series[gid] = df_g.groupby("draw_date")["_sum"].mean()

    if len(sum_series) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame(sum_series).dropna(how="all")
    game_ids = list(combined.columns)
    n = len(game_ids)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            shared = combined[[game_ids[i], game_ids[j]]].dropna()
            if len(shared) < 5:
                continue
            rho, _ = spearmanr(shared[game_ids[i]], shared[game_ids[j]])
            matrix[i, j] = matrix[j, i] = rho
    return pd.DataFrame(matrix, index=game_ids, columns=game_ids).round(3)


def anomaly_detection(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 8 — Flag out-of-range, duplicate, and extreme-sum draws."""
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = set(range(game_cfg.low_boundary, game_cfg.pool + 1))
    df = df.copy()
    df["_sum"] = df[num_cols].apply(lambda r: sum(v for v in r if pd.notna(v)), axis=1)
    mean_s, std_s = df["_sum"].mean(), df["_sum"].std()

    anomalies = []
    for _, row in df.iterrows():
        nums = [row[c] for c in num_cols]
        valid = [int(n) for n in nums if pd.notna(n)]
        issues = []
        out_rng = [n for n in valid if n not in pool]
        if out_rng:
            issues.append(f"Out-of-range: {out_rng}")
        if len(set(valid)) < len(valid):
            issues.append("Duplicate numbers")
        s = row["_sum"]
        if std_s > 0 and abs(s - mean_s) > 3 * std_s:
            issues.append(f"Extreme sum {s:.0f} (z={(s-mean_s)/std_s:+.2f})")
        if issues:
            anomalies.append({"draw_date": str(row["draw_date"])[:10],
                               "draw_time": row.get("draw_time", ""),
                               "numbers": valid, "draw_sum": float(s),
                               "issues": "; ".join(issues)})

    table = (pd.DataFrame(anomalies) if anomalies
             else pd.DataFrame(columns=["draw_date", "draw_time", "numbers", "draw_sum", "issues"]))
    conclusion = (
        f"{len(anomalies)} anomalies flagged in {len(df)} draws — "
        + ("all draws appear valid." if not anomalies
           else "review flagged rows for data quality issues.")
    )
    return AnalysisResult(
        name="Anomaly Detection", table=table, conclusion=conclusion,
        meta={"n_draws": len(df), "n_anomalies": len(anomalies)}
    )


def draw_fingerprint_pca(df: pd.DataFrame, game_cfg: GameConfig) -> dict:
    """Feature 9 — PCA to 2D for draw cluster visualization. Returns coords DataFrame."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    df_clean = df.dropna(subset=num_cols).reset_index(drop=True).copy()
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_idx = {n: i for i, n in enumerate(pool)}
    pool_size = len(pool)

    X = np.zeros((len(df_clean), pool_size), dtype=np.float32)
    for i, row in df_clean[num_cols].iterrows():
        for v in row:
            if pd.notna(v) and int(v) in pool_idx:
                X[i, pool_idx[int(v)]] = 1.0

    n_components = min(2, pool_size, len(df_clean))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    result_df = pd.DataFrame({
        "draw_date": df_clean["draw_date"].astype(str).str[:10],
        "pc1": coords[:, 0] if coords.shape[1] > 0 else 0,
        "pc2": coords[:, 1] if coords.shape[1] > 1 else 0,
        "draw_sum": df_clean[num_cols].sum(axis=1),
        "year": pd.to_datetime(df_clean["draw_date"]).dt.year,
    })
    return {"coords": result_df,
            "explained_variance": pca.explained_variance_ratio_.tolist()}


def jackpot_rollover_prob(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 14 — Historical rollover rate and probability table for n consecutive rollovers."""
    df = df.sort_values("draw_date").copy()
    df["jackpot_php"] = pd.to_numeric(df["jackpot_php"], errors="coerce")
    df["next_jp"] = df["jackpot_php"].shift(-1)
    df["rolled"] = df["jackpot_php"] < df["next_jp"]
    n_valid = int(df["rolled"].notna().sum())
    n_roll = int(df["rolled"].sum())
    roll_rate = n_roll / n_valid if n_valid else 0.5
    win_rate = 1 - roll_rate

    rows = []
    for n_rolls in range(1, 21):
        prob = roll_rate ** n_rolls
        rows.append({"consecutive_rollovers": n_rolls,
                      "probability_pct": round(prob * 100, 4),
                      "odds_str": f"1 in {round(1/prob):,}" if prob > 1e-12 else ">1 in 1T"})

    conclusion = (
        f"Historical rollover rate: {roll_rate*100:.1f}% of draws produce no winner. "
        f"Win rate: {win_rate*100:.1f}%. "
        f"P(5+ consecutive rollovers): {roll_rate**5*100:.2f}%. "
        f"Based on {n_valid} draw transitions."
    )
    return AnalysisResult(
        name="Jackpot Rollover Probability", table=pd.DataFrame(rows),
        conclusion=conclusion,
        meta={"rollover_rate": roll_rate, "win_rate": win_rate, "n_transitions": n_valid}
    )


def never_drawn_patterns(df: pd.DataFrame, game_cfg: GameConfig) -> AnalysisResult:
    """Feature 15 — Combinations (ordered games) or pairs (pick-6) that never appeared."""
    from itertools import combinations as _combs, product as _prod
    num_cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    df_clean = df.dropna(subset=num_cols)

    if game_cfg.ordered and pool_size ** game_cfg.picks <= 100_000:
        all_possible = set(_prod(pool, repeat=game_cfg.picks))
        pub_cols = [f"pub_{i}" for i in range(1, game_cfg.picks + 1)]
        use_cols = pub_cols if all(c in df_clean.columns for c in pub_cols) else num_cols
        drawn: set = set()
        for _, row in df_clean[use_cols].iterrows():
            combo = tuple(int(row[c]) for c in use_cols if pd.notna(row[c]))
            if len(combo) == game_cfg.picks:
                drawn.add(combo)
        never = sorted(all_possible - drawn)
        rows = [{"combination": "-".join(str(x) for x in c)} for c in never[:500]]
        conclusion = (
            f"{len(never):,} of {len(all_possible):,} possible combinations "
            f"({len(never)/len(all_possible)*100:.1f}%) have never been drawn."
        )
        return AnalysisResult(
            name="Never-Drawn Combinations", table=pd.DataFrame(rows),
            conclusion=conclusion,
            meta={"total_possible": len(all_possible), "never_count": len(never),
                  "drawn_count": len(drawn)}
        )

    all_pairs = set(_combs(pool, 2))
    drawn_pairs: set = set()
    for _, row in df_clean[num_cols].iterrows():
        nums = sorted(int(row[c]) for c in num_cols if pd.notna(row[c]))
        drawn_pairs.update(_combs(nums, 2))
    never_p = sorted(all_pairs - drawn_pairs)
    rows = [{"num_a": a, "num_b": b} for a, b in never_p[:500]]
    conclusion = (
        f"{len(never_p):,} of {len(all_pairs):,} possible pairs "
        f"({len(never_p)/len(all_pairs)*100:.1f}%) have never appeared together in a draw."
    )
    return AnalysisResult(
        name="Never-Drawn Pairs", table=pd.DataFrame(rows), conclusion=conclusion,
        meta={"total_pairs": len(all_pairs), "never_pairs": len(never_p),
              "drawn_pairs": len(drawn_pairs)}
    )
