"""Crowd model: which 6-number combinations do other players hold?

A jackpot is split between everyone holding the winning combination. The draw
is random, but players are not: they over-pick calendar numbers (1-31). Draws
that happened to land on a popular combination show up in the data as draws
with a jackpot winner, so P(winner | drawn combination) is learnable from the
stored winner counts and gives a per-ticket crowd estimate.

Poisson reading: if the expected number of other holders of a combination is
lam, the jackpot you win by holding it is shared 1/(1+Poisson(lam)), so its
expected share is (1 - exp(-lam)) / lam. lam comes from the fitted P(winner>0)
as lam = -ln(1 - p).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .games import GAMES, GameConfig

FEATURE_NAMES = ("low31", "low12", "consecutive", "multiples_of_5", "odd",
                 "max_fraction", "spread_fraction")
MIN_WINNER_DRAWS = 30
BASELINE_SAMPLES = 4000
# Crowd-aware picks come from the least-crowded fifth of candidates, not the single
# extreme: the model is only trustworthy within the range real draws cover.
DEFAULT_CROWD_QUANTILE = 0.2


def crowd_games() -> list[GameConfig]:
    """Unordered six-number games: the ones with a shared jackpot."""
    return [g for g in GAMES.values() if not g.ordered and g.picks == 6]


def ticket_features(tickets: np.ndarray, pool: int) -> np.ndarray:
    """Popularity features for sorted six-number tickets, one row per ticket."""
    t = np.sort(np.asarray(tickets, dtype=int), axis=1)
    return np.column_stack([
        (t <= 31).sum(1), (t <= 12).sum(1), (np.diff(t, axis=1) == 1).sum(1),
        (t % 5 == 0).sum(1), (t % 2).sum(1), t[:, -1] / pool, (t[:, -1] - t[:, 0]) / pool,
    ]).astype(float)


def random_tickets(cfg: GameConfig, m: int, rng: np.random.Generator) -> np.ndarray:
    keys = rng.random((m, cfg.pool))
    return np.sort(np.argpartition(keys, cfg.picks, axis=1)[:, :cfg.picks] + 1, axis=1)


@dataclass
class CrowdModel:
    pipeline: object
    game_ids: tuple[str, ...]
    coefficients: dict[str, float]
    cv_auc: float
    game_only_auc: float
    draws: int
    winner_draws: int
    quintile_winner_rate: list[float]
    baseline_p: dict[str, float] = field(default_factory=dict)
    baseline_share: dict[str, float] = field(default_factory=dict)

    def _design(self, tickets: np.ndarray, game_id: str) -> np.ndarray:
        pool = GAMES[game_id].pool
        dummies = np.tile([float(g == game_id) for g in self.game_ids], (len(tickets), 1))
        return np.hstack([ticket_features(tickets, pool), dummies])

    def probability(self, tickets: np.ndarray, game_id: str) -> np.ndarray:
        """P(someone else holds the jackpot combination) for each ticket."""
        if game_id not in self.game_ids:
            raise ValueError(f"No crowd model for game {game_id!r}.")
        return self.pipeline.predict_proba(self._design(np.asarray(tickets), game_id))[:, 1]

    @staticmethod
    def _share(p: np.ndarray) -> np.ndarray:
        lam = -np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
        return (1 - np.exp(-lam)) / lam

    def share_factor(self, tickets: np.ndarray, game_id: str) -> np.ndarray:
        """Expected fraction of the jackpot you keep if the ticket wins."""
        return self._share(self.probability(tickets, game_id))

    def crowd_index(self, tickets: np.ndarray, game_id: str) -> np.ndarray:
        """Crowding relative to a random ticket (1.0 = average, lower = fewer rivals)."""
        return self.probability(tickets, game_id) / self.baseline_p[game_id]

    def share_gain_pct(self, tickets: np.ndarray, game_id: str) -> float:
        """Extra jackpot kept on a win versus an average random ticket, in percent."""
        return float((self.share_factor(tickets, game_id).mean()
                      / self.baseline_share[game_id] - 1) * 100)


def fit_crowd_model(frames: dict[str, pd.DataFrame], seed: int = 7) -> CrowdModel | None:
    """Fit on stored jackpot winner counts; None when there is too little signal."""
    rows, labels, ids = [], [], []
    for game_id, df in frames.items():
        cfg = GAMES.get(game_id)
        if cfg is None or cfg.ordered or cfg.picks != 6 or "winners_jackpot" not in df:
            continue
        cols = [f"n{i}" for i in range(1, 7)]
        usable = df.dropna(subset=cols + ["winners_jackpot"])
        if usable.empty:
            continue
        rows.append((game_id, ticket_features(usable[cols].astype(int).to_numpy(), cfg.pool)))
        labels.append((usable["winners_jackpot"].astype(float).to_numpy() > 0).astype(int))
        ids.append(game_id)
    if not rows:
        return None
    y = np.concatenate(labels)
    if y.sum() < MIN_WINNER_DRAWS or y.sum() == len(y):
        return None
    game_ids = tuple(ids)
    X = np.vstack([
        np.hstack([features, np.tile([float(g == game_id) for g in game_ids], (len(features), 1))])
        for game_id, features in rows
    ])

    def make():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))

    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv_p = cross_val_predict(make(), X, y, cv=folds, method="predict_proba")[:, 1]
    only_games = X[:, len(FEATURE_NAMES):]
    game_p = cross_val_predict(make(), only_games, y, cv=folds, method="predict_proba")[:, 1]
    pipeline = make().fit(X, y)
    in_sample = pipeline.predict_proba(X)[:, 1]
    quintile = pd.qcut(in_sample, 5, labels=False, duplicates="drop")
    model = CrowdModel(
        pipeline=pipeline, game_ids=game_ids,
        coefficients=dict(zip((*FEATURE_NAMES, *(f"game_{g}" for g in game_ids)),
                              map(float, pipeline[-1].coef_[0]))),
        cv_auc=float(roc_auc_score(y, cv_p)), game_only_auc=float(roc_auc_score(y, game_p)),
        draws=len(y), winner_draws=int(y.sum()),
        quintile_winner_rate=[float(y[quintile == q].mean()) for q in sorted(set(quintile))],
    )
    rng = np.random.default_rng(seed)
    for game_id in game_ids:
        sample = random_tickets(GAMES[game_id], BASELINE_SAMPLES, rng)
        p = model.probability(sample, game_id)
        model.baseline_p[game_id] = float(p.mean())
        model.baseline_share[game_id] = float(model._share(p).mean())
    return model


def crowd_avoiding_tickets(
    model: CrowdModel, cfg: GameConfig, n: int = 5, candidates: int = 20000,
    seed: int | None = 42, weights: np.ndarray | None = None,
    max_overlap: int | None = None, quantile: float | None = None,
) -> list[list[int]]:
    """The n least crowded tickets among random candidates.

    quantile, when set, restricts the choice to that fraction of least crowded
    candidates and picks n of them at random (keeping them ordered by crowding).
    The single most extreme tickets sit outside the range real draws cover,
    where the crowd model is only extrapolating.

    weights optionally biases the candidate pool (for example ML scores, one per
    number) so the least crowded tickets are chosen from a preferred pool.
    max_overlap, when set, skips any ticket sharing more than that many numbers
    with a ticket already chosen, so the set is diverse.
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        pool = random_tickets(cfg, candidates, rng)
    else:
        w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 1e-9)
        keys = np.log(w) + rng.gumbel(size=(candidates, len(w)))
        pool = np.sort(np.argpartition(-keys, cfg.picks, axis=1)[:, :cfg.picks] + 1, axis=1)
    pool = np.unique(pool, axis=0)
    probability = model.probability(pool, cfg.id)
    order = np.argsort(probability, kind="stable")
    if quantile is not None:
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1].")
        eligible = order[:max(int(n), int(len(order) * quantile))]
        order = eligible[rng.permutation(len(eligible))]
    if max_overlap is None:
        chosen_rows = list(order[:n])
    else:
        chosen_rows = []
        for i in order:
            ticket = set(int(v) for v in pool[i])
            if all(len(ticket & set(int(v) for v in pool[j])) <= max_overlap for j in chosen_rows):
                chosen_rows.append(i)
                if len(chosen_rows) == n:
                    break
    chosen_rows.sort(key=lambda i: probability[i])
    return [[int(v) for v in pool[i]] for i in chosen_rows]
