"""One descriptive logistic classifier per game/slot, not a lottery predictor.

No tuning, model ensembles, database access, or network calls. Evaluation uses
raw binary probabilities, not normalized sampling weights or ticket odds.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, log, sqrt
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .games import GameConfig

BURN_IN = 30
MIN_DRAWS = 100
SCORE_FLOOR = 1e-12
FEATURE_NAMES = ("frequency_last10", "frequency_last30", "frequency_all",
                 "gap_normalized", "trend", "recency_weight")
CAUTION = (
    "Historical scoring and an approximate draw-level t interval do not prove "
    "a winning advantage. The interval assumes independent draw-level errors; "
    "overlapping histories may violate this assumption. Sampling weights are "
    "not joint ticket probabilities."
)


def _validate_arguments(game_cfg: GameConfig, n: int, window: int | None) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, Integral) or window < 1
    ):
        raise ValueError("window must be a positive integer or None")
    values = (game_cfg.low_boundary, game_cfg.pool, game_cfg.picks)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("Game bounds and picks must be integers")
    size = game_cfg.pool - game_cfg.low_boundary + 1
    if size < 1 or game_cfg.picks < 1 or (not game_cfg.ordered and game_cfg.picks > size):
        raise ValueError("Invalid game bounds or pick count")


def _prepare_draws(
    df: pd.DataFrame, game_cfg: GameConfig,
    window: int | None = None, draw_time: str | None = None,
) -> pd.DataFrame:
    """Filter, validate, stably sort, then window; never mutate the caller.

    A date is required to establish chronology. Times, when present, must be
    HH:MM. Complete published positions take precedence *per row* for ordered
    games; incomplete published rows use all n columns (never mix sources).
    Identical duplicate draw identities collapse; conflicting identities are
    discarded rather than leaking the same draw across the train/test boundary.
    """
    cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    empty = pd.DataFrame(columns=["draw_date", "draw_time", *cols])
    if df.empty or "draw_date" not in df:
        return empty
    work = df.copy()
    if "game" in work:
        work = work.loc[work["game"].eq(game_cfg.id).fillna(False)].copy()
    if draw_time is not None:
        if "draw_time" not in work:
            return empty
        work = work.loc[work["draw_time"].eq(draw_time).fillna(False)].copy()
    work = work.reset_index(drop=True)
    numbers = work.reindex(columns=cols).copy()
    pub = [f"pub_{i}" for i in range(1, game_cfg.picks + 1)]
    if game_cfg.ordered and all(c in work for c in pub):
        complete = work[pub].notna().all(axis=1)
        published = work[pub].copy()
        published.columns = cols
        selected = numbers.to_numpy(dtype=object, copy=True)
        selected[complete.to_numpy()] = published.loc[complete].to_numpy(dtype=object)
        numbers = pd.DataFrame(selected, columns=cols)
    # Booleans are not draw integers; numeric strings and integral floats are OK.
    not_boolean = ~numbers.map(lambda v: isinstance(v, (bool, np.bool_))).any(axis=1)
    numeric = numbers.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    valid = (
        np.isfinite(values).all(axis=1)
        & (values >= game_cfg.low_boundary).all(axis=1)
        & (values <= game_cfg.pool).all(axis=1)
        & (values == np.floor(values)).all(axis=1)
        & not_boolean.to_numpy()
    )
    if not game_cfg.ordered:
        valid &= numeric.nunique(axis=1).eq(game_cfg.picks).to_numpy()
    dates = pd.to_datetime(work["draw_date"], format="mixed", errors="coerce", utc=True)
    times = (
        work["draw_time"].astype("string") if "draw_time" in work
        else pd.Series("", index=work.index, dtype="string")
    )
    if "draw_time" in work:
        valid &= times.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d").fillna(False).to_numpy(dtype=bool)
    valid &= dates.notna().to_numpy()
    clean = pd.DataFrame(values[valid].astype(np.int64), columns=cols)
    if not game_cfg.ordered:
        clean[cols] = np.sort(clean[cols].to_numpy(), axis=1)
    clean["draw_date"] = dates.loc[valid].dt.normalize().reset_index(drop=True)
    clean["draw_time"] = times.loc[valid].reset_index(drop=True)
    identity = ["draw_date", "draw_time"]
    clean = clean.drop_duplicates(subset=[*identity, *cols])
    clean = clean.loc[~clean.duplicated(subset=identity, keep=False)]
    clean = clean.sort_values(identity, kind="stable")
    if window is not None:
        clean = clean.tail(int(window))
    return clean.reset_index(drop=True)


@dataclass(frozen=True)
class HistoricalSnapshots:
    """Candidate rows remain grouped by draw: X[draw, candidate, feature]."""

    X: np.ndarray
    y: np.ndarray
    next_X: np.ndarray
    draw_indices: np.ndarray
    provenance: tuple[dict, ...]
    feature_names: tuple[str, ...]


def _build_snapshots(clean: pd.DataFrame, game_cfg: GameConfig) -> HistoricalSnapshots:
    """Build labels and strictly prefix-only features from prepared draws.

    Frequency smoothing adds ONE pseudo-draw at the constant baseline rate.
    Ordered frequencies are position-specific, with position one-hot indicators
    supplied to a SINGLE shared classifier. No candidate number ID is a feature.
    Exposed for leakage tests; callers must first use _prepare_draws.
    """
    size = game_cfg.pool - game_cfg.low_boundary + 1
    positions = game_cfg.picks if game_cfg.ordered else 1
    candidates = size * positions
    cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
    numbers = clean[cols].to_numpy(dtype=int) - game_cfg.low_boundary
    outcomes = np.zeros((len(clean), positions, size), dtype=float)
    for i, row in enumerate(numbers):
        if game_cfg.ordered:
            outcomes[i, np.arange(positions), row] = 1.0
        else:
            outcomes[i, 0, row] = 1.0
    prefix = np.concatenate((np.zeros((1, positions, size)), outcomes.cumsum(axis=0)))
    prior = (1 if game_cfg.ordered else game_cfg.picks) / size
    names = FEATURE_NAMES
    one_hot = np.repeat(np.eye(positions), size, axis=0)
    if game_cfg.ordered:
        names += tuple(f"position_{i}" for i in range(1, positions + 1))

    # Pre-compute last-seen draw index for gap/recency features.
    last_seen_at = np.full((len(clean) + 1, positions, size), -1.0)
    for i in range(len(clean)):
        last_seen_at[i + 1] = last_seen_at[i]
        if game_cfg.ordered:
            for pos_idx in range(positions):
                last_seen_at[i + 1, pos_idx, numbers[i, pos_idx]] = float(i)
        else:
            for num_val in numbers[i]:
                last_seen_at[i + 1, 0, num_val] = float(i)

    expected_gap = size / (1 if game_cfg.ordered else game_cfg.picks)

    def features(t: int) -> np.ndarray:
        frequencies = []
        for length in (10, 30, t):
            start = max(0, t - length)
            frequencies.append(((prefix[t] - prefix[start] + prior) / (t - start + 1)).reshape(-1))
        freq_features = np.column_stack(frequencies)

        # Gap: how many draws since last appearance, normalised by expected gap.
        raw_gaps = np.where(last_seen_at[t] >= 0, t - last_seen_at[t], t + 1).reshape(-1)
        gap_norm = np.minimum(raw_gaps / expected_gap, 3.0) / 3.0

        # Trend: recent momentum (positive = heating up).
        trend_vals = frequencies[0] - frequencies[1]

        # Recency: exponential decay shaped by game's expected gap.
        recency = np.exp(-raw_gaps / expected_gap)

        result = np.column_stack((freq_features, gap_norm, trend_vals, recency))
        return np.column_stack((result, one_hot)) if game_cfg.ordered else result

    indices = np.arange(BURN_IN, len(clean), dtype=int)
    X = np.array([features(int(t)) for t in indices]).reshape(-1, candidates, len(names))
    provenance = tuple({
        "draw_index": int(t),
        "draw_date": clean.iloc[t]["draw_date"].date().isoformat(),
        "draw_time": str(clean.iloc[t]["draw_time"]),
    } for t in indices)
    return HistoricalSnapshots(
        X, outcomes[BURN_IN:].reshape(len(indices), candidates),
        features(len(clean)), indices, provenance, names,
    )


def _fit_classifier(X: np.ndarray, y: np.ndarray) -> Pipeline | None:
    """Fit one scaler/classifier to whole draw groups; return None if degenerate."""
    labels = y.reshape(-1)
    if len(np.unique(labels)) < 2:
        return None
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=0)),
    ])
    model.fit(X.reshape(-1, X.shape[-1]), labels)
    return model


def _improvement_interval(improvements: np.ndarray) -> list[float]:
    """Two-sided approximate 95% t CI for paired DRAW-level improvements.

    Uses the existing SciPy dependency; independence assumptions remain approximate.
    """
    df = len(improvements) - 1
    if df < 1:
        raise ValueError("At least two draw-level improvements are required")
    t = float(student_t.ppf(0.975, df))
    mean = float(improvements.mean())
    margin = t * float(improvements.std(ddof=1)) / sqrt(len(improvements))
    return [mean - margin, mean + margin]


def _empty_evaluation(usable_draws: int) -> dict:
    return {
        "status": "insufficient_history", "test_draws": 0, "train_draws": 0,
        "usable_draws": usable_draws, "burn_in": BURN_IN,
        "model_brier": None, "baseline_brier": None, "brier_skill": None,
        "mean_brier_improvement": None, "improvement_ci95": [],
        "evidence_of_improvement": False, "training_end": None, "test_start": None,
        "caution": CAUTION,
    }


def _sample_combinations(
    game_cfg: GameConfig, scores: np.ndarray, n: int, seed: int | None,
) -> list[list[int]]:
    """Weighted draws with bounded retries, then finite prefix completion.

    Normally sample positions independently (ordered) or sequentially without
    replacement (unordered), using normalized scores. After 32 duplicate tries,
    choose a free prefix using score times unreturned-suffix count; this bounded
    heuristic avoids rejection hangs without enumerating the ticket space.
    Uniform scores give uniform unique tickets in both paths. Neither path's
    nonuniform weights are claimed to represent joint lottery probabilities.
    """
    size = game_cfg.pool - game_cfg.low_boundary + 1
    k = game_cfg.picks
    weights = np.asarray(scores, dtype=float).reshape(k if game_cfg.ordered else 1, size)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Sampling scores must be finite and non-negative")
    weights = np.maximum(weights, SCORE_FLOOR)
    normalized = weights / weights.max(axis=1, keepdims=True)
    normalized /= normalized.sum(axis=1, keepdims=True)
    total = size**k if game_cfg.ordered else comb(size, k)
    rng = np.random.default_rng(seed)
    used: dict[tuple[int, ...], int] = {}
    result: list[list[int]] = []
    for _ in range(min(int(n), total)):
        prefix: tuple[int, ...] = ()
        for _attempt in range(32):
            if game_cfg.ordered:
                chosen = [int(rng.choice(size, p=normalized[pos])) for pos in range(k)]
            else:
                chosen = sorted(int(v) for v in rng.choice(size, size=k, replace=False, p=normalized[0]))
            candidate = tuple(chosen)
            if candidate not in used:
                prefix = candidate
                break
        for pos in range(len(prefix), k):
            remaining = k - pos - 1
            start = 0 if game_cfg.ordered or not prefix else prefix[-1] + 1
            stop = size if game_cfg.ordered else size - remaining
            options, log_weights = [], []
            for number in range(start, stop):
                capacity = size**remaining if game_cfg.ordered else comb(size - number - 1, remaining)
                available = capacity - used.get((*prefix, number), 0)
                if available:
                    options.append(number)
                    log_weights.append(log(available) + log(weights[pos if game_cfg.ordered else 0, number]))
            logits = np.asarray(log_weights)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            number = int(rng.choice(options, p=probabilities))
            prefix += (number,)
        for length in range(1, k + 1):
            key = prefix[:length]
            used[key] = used.get(key, 0) + 1
        result.append([number + game_cfg.low_boundary for number in prefix])
    return result


def generate_ml_combinations(
    df: pd.DataFrame, game_cfg: GameConfig, n: int = 5,
    window: int | None = None, draw_time: str | None = None, seed: int | None = 42,
    bet_feedback: dict[int, float] | None = None,
) -> dict:
    """Score candidates, evaluate chronologically, and sample unique tickets.

    bet_feedback optionally maps a number to a multiplier learned from the
    user's scored past bets; it rescales the production scores (capped at 1)
    before sampling and is not part of the chronological evaluation.

    Requires >=100 validated, windowed draws, with the first 30 as burn-in.
    Multiple observed slots require an explicit draw_time (otherwise fallback);
    absent time columns represent one unspecified slot. Missing requested slots
    never silently use other draws. Bad/missing dates are unusable for chronology.

    Evaluation fits the first floor(80% of eligible DRAW groups), freezes that
    model/scaler, and tests the rest. Test features may use earlier test outcomes.
    A separate identical production pipeline then fits ALL eligible draws.
    training_draws counts production-fit target draws, excluding burn-in (zero
    on fallback); evaluation.usable_draws includes the burn-in history.

    scores are marginal binary probabilities (positive floor), in number order,
    or position-major order with 1-based positions for ordered games. They are
    normalized ONLY inside sampling. feature_weights are standardized-space
    logistic coefficients, including ordered position indicators, not importance
    or causal effects. Fallback exposes no model scores, weights, or metrics.
    """
    _validate_arguments(game_cfg, n, window)
    clean = _prepare_draws(df, game_cfg, window, draw_time)
    evaluation = _empty_evaluation(len(clean))
    size = game_cfg.pool - game_cfg.low_boundary + 1
    positions = game_cfg.picks if game_cfg.ordered else 1

    def fallback(reason: str) -> dict:
        return {
            "method": "uniform_fallback",
            "combinations": _sample_combinations(game_cfg, np.ones((positions, size)), n, seed),
            "scores": [], "evaluation": evaluation, "feature_weights": {},
            "reason": f"{reason} {CAUTION}", "training_draws": 0,
        }

    if clean["draw_time"].nunique() > 1:
        return fallback("Multiple draw slots remain; specify draw_time to fit one slot.")
    if len(clean) < MIN_DRAWS:
        return fallback(f"Need at least {MIN_DRAWS} usable draws; found {len(clean)}.")
    snapshots = _build_snapshots(clean, game_cfg)
    split = len(snapshots.X) * 4 // 5
    evaluation_model = _fit_classifier(snapshots.X[:split], snapshots.y[:split])
    if evaluation_model is None:
        return fallback("Evaluation training targets contain a single class.")
    test_X, test_y = snapshots.X[split:], snapshots.y[split:]
    predicted = evaluation_model.predict_proba(test_X.reshape(-1, test_X.shape[-1]))[:, 1]
    predicted = predicted.reshape(test_y.shape)
    baseline = (1 if game_cfg.ordered else game_cfg.picks) / size
    model_errors = ((predicted - test_y) ** 2).mean(axis=1)
    baseline_errors = ((baseline - test_y) ** 2).mean(axis=1)
    improvements = baseline_errors - model_errors
    interval = _improvement_interval(improvements)
    model_brier, baseline_brier = float(model_errors.mean()), float(baseline_errors.mean())

    production_model = _fit_classifier(snapshots.X, snapshots.y)
    if production_model is None:
        return fallback("Production training targets contain a single class.")
    evaluation.update({
        "status": "evaluated", "test_draws": len(test_y), "train_draws": split,
        "model_brier": model_brier, "baseline_brier": baseline_brier,
        "brier_skill": 1.0 - model_brier / baseline_brier,
        "mean_brier_improvement": float(improvements.mean()),
        "improvement_ci95": interval, "evidence_of_improvement": bool(interval[0] > 0),
        "training_end": snapshots.provenance[split - 1],
        "test_start": snapshots.provenance[split],
    })
    probabilities = np.maximum(production_model.predict_proba(snapshots.next_X)[:, 1], SCORE_FLOOR)
    feedback_applied = 0
    if bet_feedback:
        multipliers = np.ones(len(probabilities))
        for index in range(len(probabilities)):
            number = index % size + game_cfg.low_boundary
            if number in bet_feedback:
                multipliers[index] = float(bet_feedback[number])
                feedback_applied += 1
        probabilities = np.clip(probabilities * multipliers, SCORE_FLOOR, 1.0)
    scores = []
    for index, probability in enumerate(probabilities):
        record = {"number": index % size + game_cfg.low_boundary, "score": float(probability)}
        if game_cfg.ordered:
            record["position"] = index // size + 1
        scores.append(record)
    coefficients = production_model.named_steps["classifier"].coef_[0]
    generation = {
        "method": "logistic_regression",
        "combinations": _sample_combinations(game_cfg, probabilities, n, seed),
        "scores": scores, "evaluation": evaluation,
        "feature_weights": {name: float(value) for name, value in zip(snapshots.feature_names, coefficients)},
        "reason": f"Untuned logistic scoring; separate chronological evaluation and production fit. {CAUTION}",
        "training_draws": len(snapshots.X),
    }
    if feedback_applied:
        generation["bet_feedback_numbers"] = feedback_applied
    return generation