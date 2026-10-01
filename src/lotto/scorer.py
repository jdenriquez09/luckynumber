"""
Enhanced multi-feature scoring engine for PCSO lotto number analysis.

Computes per-number features at multiple time windows, combines them with
optimised weights, and generates structurally-balanced combinations that
satisfy sum-range, odd/even, and spread constraints.

The weights are found by walk-forward logistic regression: for each
historical draw, features are computed from prior data only, then a model
learns which feature values best predict whether a number will appear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .games import GameConfig

# ---------------------------------------------------------------------------
# Default weights — learned via optimize_weights() on 6/55 historical data.
# Overridden at runtime when the optimizer runs.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "heat_10": 0.06,
    "heat_30": 0.05,
    "heat_50": 0.08,
    "heat_all": 0.09,
    "gap_score": 0.22,
    "trend": 0.05,
    "recency": 0.30,
    "consensus": 0.15,
    "pair_boost": 0.00,
}


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    sum_pct_lo: float = 20.0
    sum_pct_hi: float = 80.0
    min_odd: int = 2
    max_odd: int = 4
    min_spread_ratio: float = 0.45
    max_consec_pairs: int = 1
    candidate_multiplier: int = 50


def _num_cols(game_cfg: GameConfig) -> list[str]:
    return [f"n{i}" for i in range(1, game_cfg.picks + 1)]


def _clean(df: pd.DataFrame, game_cfg: GameConfig) -> pd.DataFrame:
    cols = _num_cols(game_cfg)
    return df.dropna(subset=cols).sort_values("draw_date").reset_index(drop=True)


def _freq_at_window(clean: pd.DataFrame, game_cfg: GameConfig,
                    window: int | None) -> dict[int, float]:
    """Frequency ratio (observed / expected) for each number over a window."""
    cols = _num_cols(game_cfg)
    pool_size = game_cfg.pool - game_cfg.low_boundary + 1
    expected_per_draw = game_cfg.picks / pool_size

    subset = clean.tail(window) if window and window < len(clean) else clean
    n = len(subset)
    if n == 0:
        return {}
    expected = n * expected_per_draw

    flat = pd.array(subset[cols].values.flatten(), dtype=object)
    vals = [int(v) for v in flat if v is not None and not pd.isna(v)]
    counts = pd.Series(vals).value_counts()

    return {
        num: counts.get(num, 0) / max(expected, 1e-9)
        for num in range(game_cfg.low_boundary, game_cfg.pool + 1)
    }


def _current_gaps(clean: pd.DataFrame, game_cfg: GameConfig) -> dict[int, int]:
    """Number of draws since each number last appeared."""
    cols = _num_cols(game_cfg)
    n = len(clean)
    gaps: dict[int, int] = {}
    for num in range(game_cfg.low_boundary, game_cfg.pool + 1):
        mask = (clean[cols] == num).any(axis=1)
        found = mask[mask].index
        gaps[num] = n - 1 - found[-1] if len(found) > 0 else n
    return gaps


def _pair_cooccurrence(clean: pd.DataFrame, game_cfg: GameConfig,
                       window: int = 100) -> dict[int, float]:
    """Average co-occurrence rate of each number with all others (recent window)."""
    cols = _num_cols(game_cfg)
    subset = clean.tail(window) if window < len(clean) else clean
    n = len(subset)
    if n < 10:
        return {num: 1.0 for num in range(game_cfg.low_boundary, game_cfg.pool + 1)}

    from collections import Counter
    from itertools import combinations
    pair_counts: Counter = Counter()
    num_counts: Counter = Counter()

    for _, row in subset.iterrows():
        nums = sorted(int(row[c]) for c in cols if pd.notna(row[c]))
        for x in nums:
            num_counts[x] += 1
        for pair in combinations(nums, 2):
            pair_counts[pair] += 1

    pool = range(game_cfg.low_boundary, game_cfg.pool + 1)
    pool_size = game_cfg.pool - game_cfg.low_boundary + 1
    expected_pair = n * (game_cfg.picks * (game_cfg.picks - 1)) / (pool_size * (pool_size - 1))

    boost: dict[int, float] = {}
    for num in pool:
        partners = [pair_counts.get(tuple(sorted([num, other])), 0)
                    for other in pool if other != num]
        avg_pair = np.mean(partners) if partners else 0
        boost[num] = avg_pair / max(expected_pair, 1e-9)
    return boost


def _adaptive_pagerank_window(game_cfg: GameConfig) -> int:
    """Scale PageRank window by game draw frequency so it covers ~3 months of draws."""
    draws_per_week = len(game_cfg.draw_days) * len(game_cfg.draw_times)
    target_weeks = 13  # ~3 months
    return max(50, min(500, draws_per_week * target_weeks))


def _pagerank_feature(clean: pd.DataFrame, game_cfg: GameConfig,
                       window: int | None = None) -> dict[int, float]:
    """NetworkX PageRank over number co-occurrence graph (weighted by co-occurrence count)."""
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    try:
        import networkx as nx
    except ImportError:
        return {num: 0.5 for num in pool}

    if window is None:
        window = _adaptive_pagerank_window(game_cfg)

    from itertools import combinations as _combs
    cols = _num_cols(game_cfg)
    subset = clean.tail(window) if window < len(clean) else clean

    G = nx.Graph()
    G.add_nodes_from(pool)
    for _, row in subset.iterrows():
        nums = [int(row[c]) for c in cols if pd.notna(row[c])]
        for a, b in _combs(nums, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    pr = nx.pagerank(G, weight="weight", max_iter=100)
    vals = list(pr.values())
    lo, hi = min(vals), max(vals)
    if hi > lo:
        return {k: (v - lo) / (hi - lo) for k, v in pr.items()}
    return {k: 0.5 for k in pr}


def gnn_cooccurrence_scores(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    window: int | None = None,
    embedding_dim: int = 8,
    epochs: int = 50,
) -> dict[int, float]:
    """
    Lightweight GCN over the co-occurrence graph producing a single predictive score.

    Builds a 2-layer GCN (manual message-passing, no PyG dependency) that learns
    node embeddings from the adjacency matrix, then scores each number by its
    embedding norm — higher means more structurally connected in recent draws.
    Falls back to PageRank if torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return _pagerank_feature(df, game_cfg, window)

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    num_to_idx = {n: i for i, n in enumerate(pool)}
    n_nodes = len(pool)

    if window is None:
        window = _adaptive_pagerank_window(game_cfg)

    cols = _num_cols(game_cfg)
    clean = _clean(df, game_cfg)
    subset = clean.tail(window) if window < len(clean) else clean

    from itertools import combinations as _combs
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in subset.iterrows():
        nums = [int(row[c]) for c in cols if pd.notna(row[c])]
        for a, b in _combs(nums, 2):
            if a in num_to_idx and b in num_to_idx:
                ai, bi = num_to_idx[a], num_to_idx[b]
                adj[ai, bi] += 1
                adj[bi, ai] += 1

    # Normalize adjacency: D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    A = torch.tensor(adj_norm, dtype=torch.float32)
    # Node features: degree + pagerank as initial features
    pr = _pagerank_feature(clean, game_cfg, window)
    node_feats = torch.zeros(n_nodes, 2)
    for num in pool:
        i = num_to_idx[num]
        node_feats[i, 0] = deg[i] / max(deg.max(), 1.0)
        node_feats[i, 1] = pr.get(num, 0.5)

    class SimpleGCN(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, out_dim, bias=False)

        def forward(self, x, adj_mat):
            h = torch.relu(adj_mat @ self.w1(x))
            return adj_mat @ self.w2(h)

    model = SimpleGCN(2, embedding_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Self-supervised: reconstruct adjacency from embeddings
    target = torch.tensor(adj, dtype=torch.float32)
    target_norm = target / max(target.max().item(), 1.0)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        emb = model(node_feats, A)
        recon = torch.sigmoid(emb @ emb.T)
        loss = nn.functional.mse_loss(recon, target_norm)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        emb = model(node_feats, A)
        scores = emb.norm(dim=1).numpy()

    lo, hi = scores.min(), scores.max()
    if hi > lo:
        normed = (scores - lo) / (hi - lo)
    else:
        normed = np.full(n_nodes, 0.5)

    return {num: round(float(normed[num_to_idx[num]]), 5) for num in pool}


def compute_features(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    draw_time: str | None = None,
) -> pd.DataFrame:
    """
    Compute multi-window features for every number in the pool.

    Returns DataFrame with columns:
        number, heat_10, heat_30, heat_50, heat_all,
        gap_score, trend, recency, pair_boost, composite
    """
    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    if clean.empty:
        return pd.DataFrame()

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    expected_gap = pool_size / game_cfg.picks

    h10 = _freq_at_window(clean, game_cfg, 10)
    h30 = _freq_at_window(clean, game_cfg, 30)
    h50 = _freq_at_window(clean, game_cfg, 50)
    h_all = _freq_at_window(clean, game_cfg, None)
    gaps = _current_gaps(clean, game_cfg)
    pairs = _pair_cooccurrence(clean, game_cfg, window=100)

    rows = []
    for num in pool:
        gap = gaps[num]
        gap_s = min(gap / expected_gap, 3.0) / 3.0
        trend = h30.get(num, 1.0) - h_all.get(num, 1.0)
        recency = 1.0 / (1.0 + gap)

        rows.append({
            "number": num,
            "heat_10": h10.get(num, 0.0),
            "heat_30": h30.get(num, 0.0),
            "heat_50": h50.get(num, 0.0),
            "heat_all": h_all.get(num, 0.0),
            "gap_score": gap_s,
            "trend": trend,
            "recency": recency,
            "pair_boost": pairs.get(num, 1.0),
        })

    result = pd.DataFrame(rows)
    result["consensus"] = consensus_ranking(result)
    result["composite"] = _apply_weights(result, DEFAULT_WEIGHTS)
    return result


# ---------------------------------------------------------------------------
# Tier 4 — Multi-Window Consensus Ranking
# ---------------------------------------------------------------------------

def consensus_ranking(features: pd.DataFrame) -> pd.Series:
    """
    Rank numbers at each heat window independently, then average the ranks.

    Numbers that score well across ALL windows are more robust signals.
    Returns a consensus score (0-1, higher = stronger consensus).
    """
    window_cols = ["heat_10", "heat_30", "heat_50", "heat_all"]
    available = [c for c in window_cols if c in features.columns]
    if not available:
        return pd.Series(0.5, index=features.index)

    n = len(features)
    rank_sum = pd.Series(0.0, index=features.index)
    for col in available:
        ranks = features[col].rank(ascending=False)
        rank_sum += ranks

    avg_rank = rank_sum / len(available)
    # Normalise: rank 1 → score 1.0, rank n → score 0.0
    if n > 1:
        return ((n - avg_rank) / (n - 1)).round(4)
    return pd.Series(0.5, index=features.index)


def _apply_weights(features: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Combine features into a single composite score, normalised to [0, 1]."""
    feat_cols = [c for c in weights if c in features.columns]
    total_w = sum(weights[c] for c in feat_cols)
    if total_w == 0:
        return pd.Series(0.5, index=features.index)

    raw = sum(weights[c] * features[c] for c in feat_cols) / total_w

    lo, hi = raw.min(), raw.max()
    if hi > lo:
        return ((raw - lo) / (hi - lo)).round(4)
    return pd.Series(0.5, index=features.index)


def _historical_sums(clean: pd.DataFrame, game_cfg: GameConfig) -> np.ndarray:
    cols = _num_cols(game_cfg)
    flat = clean[cols].apply(pd.to_numeric, errors="coerce")
    return flat.sum(axis=1).dropna().values


def smart_combinations(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n: int = 5,
    window: int | None = None,
    seed: int | None = None,
    draw_time: str | None = None,
    config: ScoringConfig | None = None,
) -> list[dict]:
    """
    Generate structurally-balanced combinations using enhanced scoring.

    Returns list of dicts, each with:
        combination: list[int], quality: float, features: dict
    """
    import hashlib

    if config is None:
        config = ScoringConfig()

    if draw_time and "draw_time" in df.columns:
        df_filtered = df[df["draw_time"] == draw_time]
    else:
        df_filtered = df
    clean = _clean(df_filtered, game_cfg)
    if clean.empty:
        return []

    features = compute_features(df_filtered, game_cfg)
    if features.empty:
        return []

    # Deterministic seed from data state
    if seed is None:
        latest = pd.to_datetime(clean["draw_date"]).max()
        ver = f"smart|{game_cfg.id}|{latest.date()}|{window}|{draw_time}"
        seed = int(hashlib.md5(ver.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)

    # Historical sum distribution for filtering
    hist_sums = _historical_sums(clean, game_cfg)
    if len(hist_sums) < 30:
        sum_lo, sum_hi = 0, 9999
    else:
        sum_lo = float(np.percentile(hist_sums, config.sum_pct_lo))
        sum_hi = float(np.percentile(hist_sums, config.sum_pct_hi))

    numbers = features["number"].values
    scores = features["composite"].values
    weights = np.clip(scores, 0.01, None)
    weights = weights / weights.sum()

    mid = (game_cfg.low_boundary + game_cfg.pool) / 2.0
    max_spread = game_cfg.pool - game_cfg.low_boundary
    picks = game_cfg.picks

    # Generate many candidates, score them, return the best n
    score_index = features.set_index("number")["composite"]
    candidates: list[dict] = []
    seen: set[frozenset] = set()
    n_candidates = n * config.candidate_multiplier

    for _ in range(n_candidates):
        combo = sorted(rng.choice(numbers, size=picks, replace=False, p=weights).tolist())
        key = frozenset(combo)
        if key in seen:
            continue
        seen.add(key)

        combo_scores = score_index.reindex(combo).values
        avg_score = float(np.mean(combo_scores))

        combo_sum = sum(combo)
        odd_count = sum(1 for x in combo if x % 2 == 1)
        spread = combo[-1] - combo[0]
        consec = sum(1 for i in range(len(combo) - 1) if combo[i + 1] - combo[i] == 1)
        high_count = sum(1 for x in combo if x > mid)

        # Soft structural bonuses (add up to ~0.15 on top of avg_score)
        sum_pct = float((hist_sums < combo_sum).mean() * 100) if len(hist_sums) else 50.0
        sum_bonus = (1 - abs(sum_pct - 50) / 50) * 0.05
        odd_bonus = (1 - abs(odd_count - picks / 2) / (picks / 2)) * 0.03
        high_bonus = (1 - abs(high_count - picks / 2) / (picks / 2)) * 0.03
        spread_bonus = (spread / max_spread if max_spread else 1) * 0.02
        consec_bonus = (1 - consec / max(1, picks - 1)) * 0.02

        quality = avg_score + sum_bonus + odd_bonus + high_bonus + spread_bonus + consec_bonus

        candidates.append({
            "combination": combo,
            "quality": round(quality, 4),
            "sum": combo_sum,
            "sum_pct": round(sum_pct, 1),
            "odd_even": f"{odd_count}/{picks - odd_count}",
            "spread": spread,
            "avg_composite": round(avg_score, 4),
        })

    candidates.sort(key=lambda x: x["quality"], reverse=True)
    return candidates[:n]


def optimize_weights(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_test_draws: int = 300,
    draw_time: str | None = None,
) -> dict[str, float]:
    """
    Find optimal feature weights via walk-forward logistic regression.

    For each of the last n_test_draws draws, computes features from prior
    data and records which numbers appeared. Fits a logistic regression
    to learn which features best predict appearance.

    Returns normalised weight dict.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)

    if len(clean) < n_test_draws + 50:
        return dict(DEFAULT_WEIGHTS)

    cols = _num_cols(game_cfg)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    expected_gap = pool_size / game_cfg.picks
    feat_names = ["heat_10", "heat_30", "heat_50", "heat_all",
                  "gap_score", "trend", "recency", "pair_boost"]

    X_all = []
    y_all = []

    start_idx = len(clean) - n_test_draws

    for i in range(start_idx, len(clean)):
        history = clean.iloc[:i]
        if len(history) < 50:
            continue

        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        h10 = _freq_at_window(history, game_cfg, 10)
        h30 = _freq_at_window(history, game_cfg, 30)
        h50 = _freq_at_window(history, game_cfg, 50)
        h_all = _freq_at_window(history, game_cfg, None)
        gaps = _current_gaps(history, game_cfg)

        for num in pool:
            gap = gaps[num]
            gap_s = min(gap / expected_gap, 3.0) / 3.0
            trend = h30.get(num, 1.0) - h_all.get(num, 1.0)
            recency = 1.0 / (1.0 + gap)

            X_all.append([
                h10.get(num, 0.0),
                h30.get(num, 0.0),
                h50.get(num, 0.0),
                h_all.get(num, 0.0),
                gap_s,
                trend,
                recency,
                1.0,  # pair_boost placeholder (too slow per-draw)
            ])
            y_all.append(1 if num in actual_nums else 0)

    X = np.array(X_all)
    y = np.array(y_all)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_scaled, y)

    coefs = np.abs(lr.coef_[0])
    total = coefs.sum()
    if total == 0:
        return dict(DEFAULT_WEIGHTS)

    weights = {name: round(float(c / total), 4) for name, c in zip(feat_names, coefs)}
    return weights


def backtest_enhanced(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_test_draws: int = 500,
    draw_time: str | None = None,
    use_smart: bool = True,
) -> dict:
    """
    Backtest the enhanced scorer against historical draws.

    Returns comparison metrics: avg_matches, 3+_match_rate, best_match,
    and comparison against the basic strategy.
    """
    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)

    min_history = 50
    start_idx = max(min_history, len(clean) - n_test_draws)

    matches_enhanced = []
    matches_basic = []

    from .predict import suggest_combinations

    for i in range(start_idx, len(clean)):
        history = clean.iloc[:i]
        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        if use_smart:
            results = smart_combinations(history, game_cfg, n=1, seed=42 + i)
            if results:
                predicted = set(results[0]["combination"])
            else:
                continue
        else:
            features = compute_features(history, game_cfg)
            if features.empty:
                continue
            scores = features.set_index("number")["composite"]
            w = np.clip(scores.values, 0.01, None)
            w = w / w.sum()
            rng = np.random.default_rng(42 + i)
            combo = sorted(rng.choice(
                features["number"].values, size=game_cfg.picks,
                replace=False, p=w
            ).tolist())
            predicted = set(combo)

        m_enh = len(predicted & actual_nums)
        matches_enhanced.append(m_enh)

        try:
            basic_combos = suggest_combinations(history, game_cfg, n=1, seed=42 + i)
            if basic_combos:
                m_basic = len(set(basic_combos[0]) & actual_nums)
            else:
                m_basic = 0
        except Exception:
            m_basic = 0
        matches_basic.append(m_basic)

    if not matches_enhanced:
        return {"error": "Not enough data"}

    enh = np.array(matches_enhanced)
    bas = np.array(matches_basic)

    return {
        "draws_tested": len(enh),
        "enhanced": {
            "avg_matches": round(float(enh.mean()), 3),
            "match_3plus_rate": round(float((enh >= 3).mean() * 100), 2),
            "match_distribution": {int(k): int(v) for k, v in
                                   pd.Series(enh).value_counts().sort_index().items()},
            "best_match": int(enh.max()),
        },
        "basic": {
            "avg_matches": round(float(bas.mean()), 3),
            "match_3plus_rate": round(float((bas >= 3).mean() * 100), 2),
            "match_distribution": {int(k): int(v) for k, v in
                                   pd.Series(bas).value_counts().sort_index().items()},
            "best_match": int(bas.max()),
        },
    }


# ===========================================================================
# Tier 2 — Machine Learning Ensemble
# ===========================================================================

def _build_ml_features(history: pd.DataFrame, game_cfg: GameConfig) -> dict[int, list[float]]:
    """
    Build a rich feature vector per number from historical draws.

    Features (15 total per number):
        0: heat_10            — freq ratio over last 10 draws
        1: heat_30            — freq ratio over last 30 draws
        2: heat_50            — freq ratio over last 50 draws
        3: heat_100           — freq ratio over last 100 draws
        4: heat_all           — freq ratio over all draws
        5: gap_score          — expected_gap / current_gap
        6: recency            — 1 / (1 + current_gap)
        7: trend_short        — heat_10 - heat_50
        8: trend_long         — heat_30 - heat_all
        9: gap_zscore         — (current_gap - expected_gap) / std(gaps)
       10: freq_rank          — rank position by total frequency (0=most frequent, 1=least)
       11: last_5_appearances — count in last 5 draws
       12: weekday_affinity   — freq ratio on this weekday vs overall
       13: consensus          — multi-window consensus rank (0-1)
       14: pagerank           — NetworkX PageRank of co-occurrence graph (normalised 0-1)
    """
    cols = _num_cols(game_cfg)
    clean = _clean(history, game_cfg)
    if len(clean) < 20:
        return {}

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    pool_size = len(pool)
    k = game_cfg.picks
    expected_gap = pool_size / k
    n = len(clean)

    h10 = _freq_at_window(clean, game_cfg, 10)
    h30 = _freq_at_window(clean, game_cfg, 30)
    h50 = _freq_at_window(clean, game_cfg, 50)
    h100 = _freq_at_window(clean, game_cfg, 100)
    h_all = _freq_at_window(clean, game_cfg, None)
    h5 = _freq_at_window(clean, game_cfg, 5)
    gaps = _current_gaps(clean, game_cfg)

    # Gap z-score: how unusual is the current gap
    gap_vals = np.array([gaps[num] for num in pool], dtype=float)
    gap_std = gap_vals.std() if gap_vals.std() > 0 else 1.0

    # Total freq for rank
    flat = pd.array(clean[cols].values.flatten(), dtype=object)
    vals = [int(v) for v in flat if v is not None and not pd.isna(v)]
    total_counts = pd.Series(vals).value_counts()

    # Weekday affinity (last draw's weekday)
    last_weekday = clean.iloc[-1].get("weekday", None)
    if last_weekday and "weekday" in clean.columns:
        wd_draws = clean[clean["weekday"] == last_weekday]
        wd_freq = _freq_at_window(wd_draws, game_cfg, None) if len(wd_draws) > 5 else h_all
    else:
        wd_freq = h_all

    # Build consensus from multi-window heat
    feat_df = pd.DataFrame({
        "number": pool,
        "heat_10": [h10.get(num, 0.0) for num in pool],
        "heat_30": [h30.get(num, 0.0) for num in pool],
        "heat_50": [h50.get(num, 0.0) for num in pool],
        "heat_all": [h_all.get(num, 0.0) for num in pool],
    })
    cons = consensus_ranking(feat_df)

    freq_sorted = sorted(pool, key=lambda x: total_counts.get(x, 0), reverse=True)
    freq_rank_map = {num: i / max(1, pool_size - 1) for i, num in enumerate(freq_sorted)}

    pr = _pagerank_feature(clean, game_cfg)  # adaptive window by game frequency

    features: dict[int, list[float]] = {}
    for idx, num in enumerate(pool):
        gap = gaps[num]
        # gap_score = overdue signal: high when number hasn't appeared in a long time.
        # gap=0 (just appeared) → 0.0 (not overdue at all)
        # gap≈expected (9.67 for 6/58) → ~0.5 (average, neutral)
        # gap=2×expected → ~0.67 (overdue)
        # Separate from recency (hot-streak signal) which is highest at gap=0.
        # The ML learns which signal dominates from training data.
        gap_score_val = min(gap / expected_gap, 3.0) / 3.0  # normalised 0..1
        recency_val = 1.0 / (1.0 + gap)                     # highest when gap=0 (just drew)
        features[num] = [
            h10.get(num, 0.0),                          # 0: heat_10
            h30.get(num, 0.0),                          # 1: heat_30
            h50.get(num, 0.0),                          # 2: heat_50
            h100.get(num, 0.0),                         # 3: heat_100
            h_all.get(num, 0.0),                        # 4: heat_all
            gap_score_val,                               # 5: gap_score (overdue signal)
            recency_val,                                 # 6: recency (hot streak signal)
            h10.get(num, 0.0) - h50.get(num, 0.0),      # 7: trend_short
            h30.get(num, 0.0) - h_all.get(num, 0.0),    # 8: trend_long
            (gap - expected_gap) / gap_std,               # 9: gap_zscore
            freq_rank_map.get(num, 0.5),                  # 10: freq_rank
            h5.get(num, 0.0) * 5 * k / pool_size,        # 11: last_5_count (approx)
            wd_freq.get(num, 1.0),                        # 12: weekday_affinity
            float(cons.iloc[idx]),                        # 13: consensus
            pr.get(num, 0.5),                             # 14: pagerank
        ]
    return features


ML_MODEL_VERSION = "v7"  # bump when feature formula changes to bust session cache

ML_FEATURE_NAMES = [
    "heat_10", "heat_30", "heat_50", "heat_100", "heat_all",
    "gap_score", "recency", "trend_short", "trend_long",
    "gap_zscore", "freq_rank", "last_5_count",
    "weekday_affinity", "consensus", "pagerank",
]


def train_ml_ensemble(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_train_draws: int = 400,
    draw_time: str | None = None,
) -> dict:
    """
    Train XGBoost + LightGBM + Random Forest ensemble via walk-forward validation,
    with isotonic regression calibration fitted on a temporal holdout.

    Returns
    -------
    {
        "xgb": fitted XGBClassifier,
        "lgb": fitted LGBMClassifier (if lightgbm installed),
        "rf":  fitted RandomForestClassifier,
        "scaler": fitted StandardScaler,
        "xgb_calibrator": IsotonicRegression,
        "lgb_calibrator": IsotonicRegression (if available),
        "rf_calibrator":  IsotonicRegression,
        "feature_importance": dict feature_name -> importance,
        "has_lgb": bool,
    }
    """
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression

    try:
        from lightgbm import LGBMClassifier
        _has_lgb = True
    except ImportError:
        _has_lgb = False

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    # Cap to available data; need at least 50 draws of history before first training row
    n_train_draws = min(n_train_draws, len(clean) - 50)
    if n_train_draws < 10:
        return {}

    X_all: list[list[float]] = []
    y_all: list[int] = []
    start_idx = max(0, len(clean) - n_train_draws)

    for i in range(start_idx, len(clean)):
        history = clean.iloc[:i]
        if len(history) < 50:
            continue

        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        feats = _build_ml_features(history, game_cfg)
        if not feats:
            continue

        for num in pool:
            X_all.append(feats[num])
            y_all.append(1 if num in actual_nums else 0)

    X = np.array(X_all)
    y = np.array(y_all)

    # 3-way temporal split: 65% base training / 15% meta-learner / 20% calibration
    pool_size = len(pool)
    n_cal_draws = max(30, int(n_train_draws * 0.2))
    n_meta_draws = max(20, int(n_train_draws * 0.15))

    cal_row_start = max(0, len(X_all) - n_cal_draws * pool_size)
    meta_row_start = max(0, cal_row_start - n_meta_draws * pool_size)

    X_tr,   y_tr   = X[:meta_row_start],              y[:meta_row_start]
    X_meta, y_meta = X[meta_row_start:cal_row_start], y[meta_row_start:cal_row_start]
    X_cal,  y_cal  = X[cal_row_start:],               y[cal_row_start:]

    scaler = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr)
    X_meta_sc = scaler.transform(X_meta)
    X_cal_sc  = scaler.transform(X_cal)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr_sc, y_tr)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr_sc, y_tr)

    # Individual calibrators on the final holdout
    xgb_cal = IsotonicRegression(out_of_bounds="clip")
    xgb_cal.fit(xgb.predict_proba(X_cal_sc)[:, 1], y_cal)
    rf_cal = IsotonicRegression(out_of_bounds="clip")
    rf_cal.fit(rf.predict_proba(X_cal_sc)[:, 1], y_cal)

    result: dict = {
        "xgb": xgb, "rf": rf,
        "scaler": scaler,
        "xgb_calibrator": xgb_cal,
        "rf_calibrator": rf_cal,
        "has_lgb": _has_lgb,
    }

    if _has_lgb:
        lgb = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        lgb.fit(X_tr_sc, y_tr)
        lgb_cal = IsotonicRegression(out_of_bounds="clip")
        lgb_cal.fit(lgb.predict_proba(X_cal_sc)[:, 1], y_cal)
        result["lgb"] = lgb
        result["lgb_calibrator"] = lgb_cal

    # --- Stacked meta-learner ---
    # Trains LogisticRegression on RAW base model outputs on the held-out meta
    # split (novel data for all base models), then calibrates on the final holdout.
    # Learns optimal per-model blending weights rather than simple averaging.
    from sklearn.linear_model import LogisticRegression as _MetaLR

    _meta_cols = [
        xgb.predict_proba(X_meta_sc)[:, 1],
        rf.predict_proba(X_meta_sc)[:, 1],
    ]
    if _has_lgb:
        _meta_cols.append(lgb.predict_proba(X_meta_sc)[:, 1])

    if len(y_meta) >= 20 and y_meta.sum() >= 5:
        X_meta_stack = np.column_stack(_meta_cols)
        meta_lr = _MetaLR(C=0.1, max_iter=300, solver="lbfgs", random_state=42)
        meta_lr.fit(X_meta_stack, y_meta)

        _cal_cols = [
            xgb.predict_proba(X_cal_sc)[:, 1],
            rf.predict_proba(X_cal_sc)[:, 1],
        ]
        if _has_lgb:
            _cal_cols.append(lgb.predict_proba(X_cal_sc)[:, 1])
        X_cal_stack = np.column_stack(_cal_cols)
        meta_calibrator = IsotonicRegression(out_of_bounds="clip")
        meta_calibrator.fit(meta_lr.predict_proba(X_cal_stack)[:, 1], y_cal)

        result["meta"] = meta_lr
        result["meta_calibrator"] = meta_calibrator

    # Feature importance: average across all available models
    imps = [xgb.feature_importances_, rf.feature_importances_]
    if _has_lgb:
        lgb_raw = lgb.feature_importances_.astype(float)
        lgb_norm = lgb_raw / lgb_raw.sum() if lgb_raw.sum() > 0 else lgb_raw
        imps.append(lgb_norm)
    avg_imp = np.mean(imps, axis=0)
    total = avg_imp.sum()
    result["feature_importance"] = {
        name: round(float(imp / total), 4) if total > 0 else 0.0
        for name, imp in zip(ML_FEATURE_NAMES, avg_imp)
    }

    return result


def ml_predict_numbers(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble: dict,
    draw_time: str | None = None,
) -> pd.DataFrame:
    """
    Use a trained ML ensemble to predict per-number appearance probability.
    Applies isotonic calibration when available.

    Returns DataFrame with columns:
        number, xgb_prob, rf_prob, [lgb_prob,] ensemble_prob, rank
    """
    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]

    feats = _build_ml_features(df, game_cfg)
    if not feats:
        return pd.DataFrame()

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    X = np.array([feats[num] for num in pool])
    X_scaled = ensemble["scaler"].transform(X)

    xgb_raw = ensemble["xgb"].predict_proba(X_scaled)[:, 1]
    rf_raw = ensemble["rf"].predict_proba(X_scaled)[:, 1]

    xgb_probs = (ensemble["xgb_calibrator"].predict(xgb_raw)
                 if "xgb_calibrator" in ensemble else xgb_raw)
    rf_probs = (ensemble["rf_calibrator"].predict(rf_raw)
                if "rf_calibrator" in ensemble else rf_raw)

    model_probs = [xgb_probs, rf_probs]
    result_cols: dict = {
        "number": pool,
        "xgb_prob": np.round(xgb_probs, 4),
        "rf_prob": np.round(rf_probs, 4),
    }

    if ensemble.get("has_lgb") and "lgb" in ensemble:
        lgb_raw = ensemble["lgb"].predict_proba(X_scaled)[:, 1]
        lgb_probs = (ensemble["lgb_calibrator"].predict(lgb_raw)
                     if "lgb_calibrator" in ensemble else lgb_raw)
        model_probs.append(lgb_probs)
        result_cols["lgb_prob"] = np.round(lgb_probs, 4)

    # Use stacked meta-learner when available; fall back to simple averaging
    if "meta" in ensemble and "meta_calibrator" in ensemble:
        _meta_input = np.column_stack([xgb_raw, rf_raw] + (
            [ensemble["lgb"].predict_proba(X_scaled)[:, 1]]
            if ensemble.get("has_lgb") and "lgb" in ensemble else []
        ))
        _meta_raw = ensemble["meta"].predict_proba(_meta_input)[:, 1]
        ensemble_probs = ensemble["meta_calibrator"].predict(_meta_raw)
    else:
        ensemble_probs = np.mean(model_probs, axis=0)

    result_cols["ensemble_prob"] = np.round(ensemble_probs, 4)

    result = pd.DataFrame(result_cols)
    result["rank"] = result["ensemble_prob"].rank(ascending=False).astype(int)
    return result.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)


def ml_combinations(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble: dict,
    n: int = 5,
    seed: int | None = None,
    draw_time: str | None = None,
) -> list[dict]:
    """
    Generate combinations using ML ensemble probabilities as weights.

    Combines ML probability with structural quality scoring.
    """
    import hashlib

    preds = ml_predict_numbers(df, game_cfg, ensemble, draw_time=draw_time)
    if preds.empty:
        return []

    if seed is None:
        if draw_time and "draw_time" in df.columns:
            df_f = df[df["draw_time"] == draw_time]
        else:
            df_f = df
        clean = _clean(df_f, game_cfg)
        if clean.empty:
            return []
        latest = pd.to_datetime(clean["draw_date"]).max()
        ver = f"ml|{game_cfg.id}|{latest.date()}|{draw_time}|{len(clean)}"
        seed = int(hashlib.md5(ver.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)

    numbers = preds["number"].values
    probs = preds["ensemble_prob"].values
    weights = np.clip(probs, 0.001, None)
    weights = weights / weights.sum()

    picks = game_cfg.picks
    mid = (game_cfg.low_boundary + game_cfg.pool) / 2.0
    max_spread = game_cfg.pool - game_cfg.low_boundary

    # Historical sums for percentile calc
    if draw_time and "draw_time" in df.columns:
        df_f = df[df["draw_time"] == draw_time]
    else:
        df_f = df
    clean = _clean(df_f, game_cfg)
    hist_sums = _historical_sums(clean, game_cfg) if not clean.empty else np.array([])

    prob_index = preds.set_index("number")["ensemble_prob"]
    candidates: list[dict] = []
    seen: set[frozenset] = set()

    for _ in range(n * 50):
        combo = sorted(rng.choice(numbers, size=picks, replace=False, p=weights).tolist())
        key = frozenset(combo)
        if key in seen:
            continue
        seen.add(key)

        combo_probs = prob_index.reindex(combo).values
        avg_prob = float(np.mean(combo_probs))

        combo_sum = sum(combo)
        odd_count = sum(1 for x in combo if x % 2 == 1)
        spread = combo[-1] - combo[0]
        high_count = sum(1 for x in combo if x > mid)
        consec = sum(1 for i in range(len(combo) - 1) if combo[i + 1] - combo[i] == 1)

        sum_pct = float((hist_sums < combo_sum).mean() * 100) if len(hist_sums) else 50.0
        sum_bonus = (1 - abs(sum_pct - 50) / 50) * 0.05
        odd_bonus = (1 - abs(odd_count - picks / 2) / (picks / 2)) * 0.03
        high_bonus = (1 - abs(high_count - picks / 2) / (picks / 2)) * 0.03
        spread_bonus = (spread / max_spread if max_spread else 1) * 0.02
        consec_bonus = (1 - consec / max(1, picks - 1)) * 0.02

        quality = avg_prob + sum_bonus + odd_bonus + high_bonus + spread_bonus + consec_bonus

        candidates.append({
            "combination": combo,
            "quality": round(quality, 4),
            "ml_prob_avg": round(avg_prob, 4),
            "sum": combo_sum,
            "sum_pct": round(sum_pct, 1),
            "odd_even": f"{odd_count}/{picks - odd_count}",
            "spread": spread,
        })

    candidates.sort(key=lambda x: x["quality"], reverse=True)
    return candidates[:n]


def backtest_ml(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_test_draws: int = 200,
    n_train_draws: int = 300,
    draw_time: str | None = None,
    retrain_every: int = 75,
) -> dict:
    """
    Walk-forward backtest of the ML ensemble with periodic retraining.

    Retrains every `retrain_every` draws to match production behavior where
    the model is refreshed each session with new data.
    """
    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)

    min_data = n_train_draws + 50
    if len(clean) < min_data + n_test_draws:
        n_test_draws = max(50, len(clean) - min_data)

    start_idx = len(clean) - n_test_draws
    if start_idx < min_data:
        return {"error": "Not enough data for ML backtest"}

    matches_ml = []
    matches_basic = []
    from .predict import suggest_combinations

    ensemble = None
    last_train_idx = -1

    for i in range(start_idx, len(clean)):
        # Retrain periodically to match production behavior
        draws_since_train = i - last_train_idx
        if ensemble is None or draws_since_train >= retrain_every:
            train_data = clean.iloc[:i]
            ensemble = train_ml_ensemble(
                train_data, game_cfg,
                n_train_draws=min(n_train_draws, len(train_data) - 50),
            )
            last_train_idx = i
            if not ensemble:
                return {"error": "ML training failed"}

        history = clean.iloc[:i]
        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        # ML prediction
        results = ml_combinations(history, game_cfg, ensemble, n=1, seed=42 + i)
        if results:
            m_ml = len(set(results[0]["combination"]) & actual_nums)
        else:
            m_ml = 0
        matches_ml.append(m_ml)

        # Basic prediction
        try:
            basic_combos = suggest_combinations(history, game_cfg, n=1, seed=42 + i)
            m_basic = len(set(basic_combos[0]) & actual_nums) if basic_combos else 0
        except Exception:
            m_basic = 0
        matches_basic.append(m_basic)

    ml_arr = np.array(matches_ml)
    bas_arr = np.array(matches_basic)

    return {
        "draws_tested": len(ml_arr),
        "retrain_every": retrain_every,
        "n_retrains": max(1, n_test_draws // retrain_every),
        "feature_importance": ensemble.get("feature_importance", {}),
        "ml": {
            "avg_matches": round(float(ml_arr.mean()), 3),
            "match_3plus_rate": round(float((ml_arr >= 3).mean() * 100), 2),
            "match_distribution": {int(k): int(v) for k, v in
                                   pd.Series(ml_arr).value_counts().sort_index().items()},
            "best_match": int(ml_arr.max()),
        },
        "basic": {
            "avg_matches": round(float(bas_arr.mean()), 3),
            "match_3plus_rate": round(float((bas_arr >= 3).mean() * 100), 2),
            "match_distribution": {int(k): int(v) for k, v in
                                   pd.Series(bas_arr).value_counts().sort_index().items()},
            "best_match": int(bas_arr.max()),
        },
    }


# ===========================================================================
# Ordered-game (2D / 3D / 4D / 6D) position-aware ML
# ===========================================================================

_ORDERED_FEATURE_NAMES = [
    "heat_10", "heat_30", "heat_100", "heat_all",
    "gap_score", "gap_raw", "weekday_affinity",
]


def _build_ml_features_ordered(
    clean: pd.DataFrame,
    game_cfg: GameConfig,
    position: int,
) -> dict[int, list[float]]:
    """
    Per-digit feature vector for one draw position (1-indexed).

    Digit `d` in position `pos` is treated as its own outcome:
    frequency, gap, and weekday-affinity are all position-specific.
    """
    pos_col = f"n{position}"
    if pos_col not in clean.columns or clean.empty:
        return {}

    digits = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    n = len(clean)
    pool_size = len(digits)

    # Weekday of most-recent draw (for affinity computation)
    last_weekday = clean.iloc[-1].get("weekday", None)

    feats: dict[int, list[float]] = {}
    for d in digits:
        mask = clean[pos_col] == d

        h10  = mask.tail(10).sum()  / max(min(10, n), 1)
        h30  = mask.tail(30).sum()  / max(min(30, n), 1)
        h100 = mask.tail(100).sum() / max(min(100, n), 1)
        hall = mask.sum()           / max(n, 1)

        found = mask[mask].index
        gap_raw = n - 1 - int(found[-1]) if len(found) > 0 else n
        expected_gap = pool_size  # each digit expected every pool_size draws in this position
        gap_score = expected_gap / max(gap_raw, 1)

        if last_weekday is not None and "weekday" in clean.columns:
            wd_rows = clean[clean["weekday"] == last_weekday]
            wd_freq = (wd_rows[pos_col] == d).sum() / max(len(wd_rows), 1)
        else:
            wd_freq = hall

        feats[d] = [h10, h30, h100, hall, gap_score, float(gap_raw) / max(n, 1), wd_freq]

    return feats


def train_ml_ensemble_ordered(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_train_draws: int = 400,
) -> dict:
    """
    Train one XGB+RF ensemble per draw position for ordered games (2D/3D/4D/6D).

    Returns {
        "positions": {1: {"xgb": ..., "rf": ..., "scaler": ..., "xgb_cal": ..., "rf_cal": ...}, ...},
        "picks": game_cfg.picks,
        "pool":  game_cfg.pool,
        "feature_importance": {pos: {feat: imp, ...}, ...},
    }
    """
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression

    clean = _clean(df, game_cfg)
    n_train_draws = min(n_train_draws, len(clean) - 30)
    if n_train_draws < 10:
        return {}

    start_idx = max(0, len(clean) - n_train_draws)
    digits = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    position_models: dict[int, dict] = {}
    feature_importance: dict[int, dict] = {}

    for pos in range(1, game_cfg.picks + 1):
        pos_col = f"n{pos}"
        X_all: list[list[float]] = []
        y_all: list[int] = []

        for i in range(start_idx, len(clean)):
            history = clean.iloc[:i]
            if len(history) < 20:
                continue
            actual_digit = clean.iloc[i].get(pos_col)
            if actual_digit is None or pd.isna(actual_digit):
                continue
            actual_digit = int(actual_digit)

            feats = _build_ml_features_ordered(history, game_cfg, pos)
            if not feats:
                continue
            for d in digits:
                X_all.append(feats[d])
                y_all.append(1 if d == actual_digit else 0)

        if len(X_all) < 50:
            continue

        X = np.array(X_all)
        y = np.array(y_all)

        n_cal = max(20, int(n_train_draws * 0.2))
        pool_size = len(digits)
        cal_start = max(0, len(X) - n_cal * pool_size)
        X_tr, y_tr = X[:cal_start], y[:cal_start]
        X_cal, y_cal = X[cal_start:], y[cal_start:]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_cal_sc = scaler.transform(X_cal)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        xgb.fit(X_tr_sc, y_tr)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr_sc, y_tr)

        xgb_cal = IsotonicRegression(out_of_bounds="clip")
        xgb_cal.fit(xgb.predict_proba(X_cal_sc)[:, 1], y_cal)
        rf_cal = IsotonicRegression(out_of_bounds="clip")
        rf_cal.fit(rf.predict_proba(X_cal_sc)[:, 1], y_cal)

        avg_imp = np.mean([xgb.feature_importances_, rf.feature_importances_], axis=0)
        total = avg_imp.sum()
        feature_importance[pos] = {
            name: round(float(imp / total), 4) if total > 0 else 0.0
            for name, imp in zip(_ORDERED_FEATURE_NAMES, avg_imp)
        }

        position_models[pos] = {
            "xgb": xgb, "rf": rf,
            "scaler": scaler,
            "xgb_cal": xgb_cal, "rf_cal": rf_cal,
        }

    if not position_models:
        return {}

    return {
        "positions": position_models,
        "picks": game_cfg.picks,
        "pool": game_cfg.pool,
        "low_boundary": game_cfg.low_boundary,
        "feature_importance": feature_importance,
    }


def ml_predict_ordered(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble_ord: dict,
) -> dict[int, pd.DataFrame]:
    """
    Predict per-digit probability for each position.

    Returns {position: DataFrame(digit, xgb_prob, rf_prob, ensemble_prob, rank)}.
    """
    clean = _clean(df, game_cfg)
    digits = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    results: dict[int, pd.DataFrame] = {}

    for pos, models in ensemble_ord["positions"].items():
        feats = _build_ml_features_ordered(clean, game_cfg, pos)
        if not feats:
            continue

        X = np.array([feats[d] for d in digits])
        X_sc = models["scaler"].transform(X)

        xgb_raw = models["xgb"].predict_proba(X_sc)[:, 1]
        rf_raw  = models["rf"].predict_proba(X_sc)[:, 1]

        xgb_p = models["xgb_cal"].predict(xgb_raw)
        rf_p  = models["rf_cal"].predict(rf_raw)
        ens_p = (xgb_p + rf_p) / 2

        df_pos = pd.DataFrame({
            "digit": digits,
            "xgb_prob": np.round(xgb_p, 4),
            "rf_prob":  np.round(rf_p, 4),
            "ensemble_prob": np.round(ens_p, 4),
        })
        df_pos["rank"] = df_pos["ensemble_prob"].rank(ascending=False).astype(int)
        results[pos] = df_pos.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)

    return results


def ml_combinations_ordered(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble_ord: dict,
    n: int = 5,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate combinations for ordered games by picking the top-weighted
    digit per position according to the calibrated ensemble probability.
    """
    rng = np.random.default_rng(seed)
    preds = ml_predict_ordered(df, game_cfg, ensemble_ord)
    if not preds:
        return []

    combos = []
    for _ in range(n):
        combo = []
        for pos in range(1, game_cfg.picks + 1):
            df_pos = preds.get(pos)
            if df_pos is None or df_pos.empty:
                combo.append(int(rng.integers(game_cfg.low_boundary, game_cfg.pool + 1)))
                continue
            # Sample with probability proportional to ensemble_prob
            probs = df_pos["ensemble_prob"].values.astype(float)
            probs = np.clip(probs, 0, None)
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(len(probs)) / len(probs)
            chosen = int(rng.choice(df_pos["digit"].values, p=probs))
            combo.append(chosen)

        avg_prob = float(np.mean([
            float(preds[pos][preds[pos]["digit"] == combo[pos - 1]]["ensemble_prob"].values[0])
            for pos in range(1, game_cfg.picks + 1)
            if pos in preds
        ]))
        combos.append({"combination": combo, "avg_prob": round(avg_prob, 4)})

    return combos


# ===========================================================================
# SHAP Explainability
# ===========================================================================

def shap_explain_numbers(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble: dict,
    draw_time: str | None = None,
) -> pd.DataFrame:
    """
    Compute SHAP values averaged across all ensemble models (XGB + RF + LGB).

    Returns DataFrame (one row per number) with columns:
        number, total_shap, <feature_name>...
    Sorted descending by total_shap (highest = most positively predicted).
    Returns empty DataFrame if shap is not installed or ensemble has no xgb.
    """
    try:
        import shap as _shap
    except ImportError:
        return pd.DataFrame()

    if "xgb" not in ensemble or "scaler" not in ensemble:
        return pd.DataFrame()

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]

    feats = _build_ml_features(df, game_cfg)
    if not feats:
        return pd.DataFrame()

    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
    X = np.array([feats[num] for num in pool])
    X_scaled = ensemble["scaler"].transform(X)

    def _extract_shap(model):
        explainer = _shap.TreeExplainer(model)
        shap_out = explainer.shap_values(X_scaled)
        # list form: [neg_class, pos_class] — older SHAP + XGBoost/RF
        if isinstance(shap_out, list):
            sv = shap_out[1]
        else:
            sv = shap_out
        # newer SHAP returns 3D (n_samples, n_features, n_classes) for RF
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    shap_matrices = [_extract_shap(ensemble["xgb"])]
    shap_matrices.append(_extract_shap(ensemble["rf"]))
    if ensemble.get("has_lgb") and "lgb" in ensemble:
        shap_matrices.append(_extract_shap(ensemble["lgb"]))

    sv = np.mean(shap_matrices, axis=0)

    rows = []
    for idx, num in enumerate(pool):
        row: dict = {"number": num}
        for fi, fname in enumerate(ML_FEATURE_NAMES):
            row[fname] = round(float(sv[idx, fi]), 5)
        row["total_shap"] = round(float(sv[idx].sum()), 5)
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("total_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P0: Calibration diagnostics
# ---------------------------------------------------------------------------

def calibration_diagnostics(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble: dict,
    n_bins: int = 10,
    draw_time: str | None = None,
) -> dict:
    """
    Evaluate ensemble calibration quality on a temporal holdout.

    Returns reliability-diagram data, Expected Calibration Error (ECE),
    Brier score, and per-model metrics.
    """
    from sklearn.metrics import brier_score_loss

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    n_holdout = max(50, int(len(clean) * 0.15))
    start = len(clean) - n_holdout
    if start < 100:
        return {"error": "Not enough data for calibration diagnostics"}

    y_true_all: list[int] = []
    probs: dict[str, list[float]] = {"ensemble": [], "xgb": [], "rf": []}
    if ensemble.get("has_lgb"):
        probs["lgb"] = []

    for i in range(start, len(clean)):
        history = clean.iloc[:i]
        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        feats = _build_ml_features(history, game_cfg)
        if not feats:
            continue

        X = np.array([feats[num] for num in pool])
        X_sc = ensemble["scaler"].transform(X)

        xgb_raw = ensemble["xgb"].predict_proba(X_sc)[:, 1]
        rf_raw = ensemble["rf"].predict_proba(X_sc)[:, 1]

        xgb_p = (ensemble["xgb_calibrator"].predict(xgb_raw)
                 if "xgb_calibrator" in ensemble else xgb_raw)
        rf_p = (ensemble["rf_calibrator"].predict(rf_raw)
                if "rf_calibrator" in ensemble else rf_raw)

        model_ps = [xgb_p, rf_p]
        if ensemble.get("has_lgb") and "lgb" in ensemble:
            lgb_raw = ensemble["lgb"].predict_proba(X_sc)[:, 1]
            lgb_p = (ensemble["lgb_calibrator"].predict(lgb_raw)
                     if "lgb_calibrator" in ensemble else lgb_raw)
            model_ps.append(lgb_p)

        ens_p = np.mean(model_ps, axis=0)

        for j, num in enumerate(pool):
            y_true_all.append(1 if num in actual_nums else 0)
            probs["ensemble"].append(float(ens_p[j]))
            probs["xgb"].append(float(xgb_p[j]))
            probs["rf"].append(float(rf_p[j]))
            if "lgb" in probs:
                probs["lgb"].append(float(lgb_p[j]))

    y_true = np.array(y_true_all)

    def _reliability_and_ece(p_arr: np.ndarray) -> tuple[list[dict], float]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins_data = []
        ece = 0.0
        for b in range(n_bins):
            mask = (p_arr >= bin_edges[b]) & (p_arr < bin_edges[b + 1])
            if b == n_bins - 1:
                mask |= p_arr == bin_edges[b + 1]
            count = int(mask.sum())
            if count == 0:
                bins_data.append({
                    "bin_lo": round(float(bin_edges[b]), 3),
                    "bin_hi": round(float(bin_edges[b + 1]), 3),
                    "mean_predicted": None,
                    "fraction_positive": None,
                    "count": 0,
                })
                continue
            mean_pred = float(p_arr[mask].mean())
            frac_pos = float(y_true[mask].mean())
            ece += abs(frac_pos - mean_pred) * count
            bins_data.append({
                "bin_lo": round(float(bin_edges[b]), 3),
                "bin_hi": round(float(bin_edges[b + 1]), 3),
                "mean_predicted": round(mean_pred, 5),
                "fraction_positive": round(frac_pos, 5),
                "count": count,
            })
        ece /= len(p_arr)
        return bins_data, round(ece, 5)

    metrics = {}
    for model_name, p_list in probs.items():
        p_arr = np.array(p_list)
        reliability, ece = _reliability_and_ece(p_arr)
        brier = round(float(brier_score_loss(y_true, p_arr)), 5)
        metrics[model_name] = {
            "brier_score": brier,
            "ece": ece,
            "reliability": reliability,
        }

    return {
        "holdout_draws": n_holdout,
        "total_predictions": len(y_true_all),
        "base_rate": round(float(y_true.mean()), 5),
        "models": metrics,
    }


# ---------------------------------------------------------------------------
# P1: Optuna Bayesian hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    n_trials: int = 30,
    draw_time: str | None = None,
) -> dict:
    """
    Bayesian hyperparameter optimization per game using Optuna TPE sampler.

    Returns optimized parameters for XGB, LGB, and RF, plus best Brier score.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"error": "optuna not installed"}

    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    try:
        from lightgbm import LGBMClassifier
        _has_lgb = True
    except ImportError:
        _has_lgb = False

    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    n_rows = min(300, len(clean) - 50)
    if n_rows < 50:
        return {"error": "Not enough data for tuning"}

    start_idx = len(clean) - n_rows
    X_all: list[list[float]] = []
    y_all: list[int] = []

    for i in range(start_idx, len(clean)):
        history = clean.iloc[:i]
        if len(history) < 50:
            continue
        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))
        feats = _build_ml_features(history, game_cfg)
        if not feats:
            continue
        for num in pool:
            X_all.append(feats[num])
            y_all.append(1 if num in actual_nums else 0)

    X = np.array(X_all)
    y = np.array(y_all)

    pool_size = len(pool)
    n_cal = max(30, int(n_rows * 0.2))
    cal_start = len(X) - n_cal * pool_size
    cal_start = max(0, cal_start)
    X_tr, y_tr = X[:cal_start], y[:cal_start]
    X_val, y_val = X[cal_start:], y[cal_start:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    def objective(trial):
        xgb_params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 600),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
        }
        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 600),
            "max_depth": trial.suggest_int("rf_max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("rf_min_leaf", 5, 50),
        }

        xgb = XGBClassifier(
            **xgb_params, eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        xgb.fit(X_tr_sc, y_tr)
        rf = RandomForestClassifier(
            **rf_params, random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr_sc, y_tr)

        xgb_cal = IsotonicRegression(out_of_bounds="clip")
        xgb_cal.fit(xgb.predict_proba(X_val_sc)[:, 1], y_val)
        rf_cal = IsotonicRegression(out_of_bounds="clip")
        rf_cal.fit(rf.predict_proba(X_val_sc)[:, 1], y_val)

        model_ps = [
            xgb_cal.predict(xgb.predict_proba(X_val_sc)[:, 1]),
            rf_cal.predict(rf.predict_proba(X_val_sc)[:, 1]),
        ]

        if _has_lgb:
            lgb_params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 600),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 8),
                "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            }
            lgb = LGBMClassifier(
                **lgb_params, random_state=42, verbosity=-1, n_jobs=-1,
            )
            lgb.fit(X_tr_sc, y_tr)
            lgb_cal = IsotonicRegression(out_of_bounds="clip")
            lgb_cal.fit(lgb.predict_proba(X_val_sc)[:, 1], y_val)
            model_ps.append(lgb_cal.predict(lgb.predict_proba(X_val_sc)[:, 1]))

        ens_p = np.mean(model_ps, axis=0)
        return brier_score_loss(y_val, ens_p)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    return {
        "best_brier": round(study.best_value, 5),
        "n_trials": n_trials,
        "game": game_cfg.id,
        "xgb_params": {k.replace("xgb_", ""): v for k, v in best.items() if k.startswith("xgb_")},
        "rf_params": {k.replace("rf_", ""): v for k, v in best.items() if k.startswith("rf_")},
        "lgb_params": ({k.replace("lgb_", ""): v for k, v in best.items() if k.startswith("lgb_")}
                       if _has_lgb else None),
    }


# ---------------------------------------------------------------------------
# P2: Conformal prediction sets
# ---------------------------------------------------------------------------

def conformal_prediction_set(
    df: pd.DataFrame,
    game_cfg: GameConfig,
    ensemble: dict,
    alpha: float = 0.10,
    draw_time: str | None = None,
) -> dict:
    """
    Distribution-free prediction sets with coverage guarantees via split conformal.

    At confidence level (1 - alpha), the returned set is guaranteed to contain
    the drawn numbers at the specified coverage rate over exchangeable data.
    """
    if draw_time and "draw_time" in df.columns:
        df = df[df["draw_time"] == draw_time]
    clean = _clean(df, game_cfg)
    cols = _num_cols(game_cfg)
    pool = list(range(game_cfg.low_boundary, game_cfg.pool + 1))

    n_cal = max(50, int(len(clean) * 0.15))
    cal_start = len(clean) - n_cal
    if cal_start < 100:
        return {"error": "Not enough data for conformal prediction"}

    nonconformity_scores: list[float] = []

    for i in range(cal_start, len(clean)):
        history = clean.iloc[:i]
        actual_row = clean.iloc[i]
        actual_nums = set(int(actual_row[c]) for c in cols if pd.notna(actual_row[c]))

        feats = _build_ml_features(history, game_cfg)
        if not feats:
            continue

        X = np.array([feats[num] for num in pool])
        X_sc = ensemble["scaler"].transform(X)

        model_ps = []
        xgb_raw = ensemble["xgb"].predict_proba(X_sc)[:, 1]
        xgb_p = (ensemble["xgb_calibrator"].predict(xgb_raw)
                 if "xgb_calibrator" in ensemble else xgb_raw)
        model_ps.append(xgb_p)

        rf_raw = ensemble["rf"].predict_proba(X_sc)[:, 1]
        rf_p = (ensemble["rf_calibrator"].predict(rf_raw)
                if "rf_calibrator" in ensemble else rf_raw)
        model_ps.append(rf_p)

        if ensemble.get("has_lgb") and "lgb" in ensemble:
            lgb_raw = ensemble["lgb"].predict_proba(X_sc)[:, 1]
            lgb_p = (ensemble["lgb_calibrator"].predict(lgb_raw)
                     if "lgb_calibrator" in ensemble else lgb_raw)
            model_ps.append(lgb_p)

        ens_p = np.mean(model_ps, axis=0)
        prob_map = dict(zip(pool, ens_p))

        for num in actual_nums:
            if num in prob_map:
                nonconformity_scores.append(1.0 - prob_map[num])

    if len(nonconformity_scores) < 10:
        return {"error": "Insufficient calibration data for conformal prediction"}

    scores = np.array(sorted(nonconformity_scores))
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    threshold = float(np.quantile(scores, q_level))

    # Build prediction set for current data
    feats = _build_ml_features(clean, game_cfg)
    if not feats:
        return {"error": "Feature computation failed"}

    X = np.array([feats[num] for num in pool])
    X_sc = ensemble["scaler"].transform(X)

    model_ps = []
    xgb_raw = ensemble["xgb"].predict_proba(X_sc)[:, 1]
    xgb_p = (ensemble["xgb_calibrator"].predict(xgb_raw)
             if "xgb_calibrator" in ensemble else xgb_raw)
    model_ps.append(xgb_p)

    rf_raw = ensemble["rf"].predict_proba(X_sc)[:, 1]
    rf_p = (ensemble["rf_calibrator"].predict(rf_raw)
            if "rf_calibrator" in ensemble else rf_raw)
    model_ps.append(rf_p)

    if ensemble.get("has_lgb") and "lgb" in ensemble:
        lgb_raw = ensemble["lgb"].predict_proba(X_sc)[:, 1]
        lgb_p = (ensemble["lgb_calibrator"].predict(lgb_raw)
                 if "lgb_calibrator" in ensemble else lgb_raw)
        model_ps.append(lgb_p)

    ens_p = np.mean(model_ps, axis=0)

    prediction_set = []
    for j, num in enumerate(pool):
        if (1.0 - ens_p[j]) <= threshold:
            prediction_set.append({
                "number": num,
                "probability": round(float(ens_p[j]), 5),
            })

    prediction_set.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "alpha": alpha,
        "coverage_guarantee": round(1 - alpha, 2),
        "threshold": round(threshold, 5),
        "calibration_draws": n_cal,
        "calibration_scores": n,
        "prediction_set_size": len(prediction_set),
        "pool_size": len(pool),
        "prediction_set": prediction_set,
    }
