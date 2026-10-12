"""Crowd model and Strategy Arena: synthetic data only, no database or network."""
import numpy as np
import pandas as pd
import pytest

from src.lotto import arena
from src.lotto.crowd import (
    crowd_avoiding_tickets, crowd_games, fit_crowd_model, random_tickets, ticket_features,
)
from src.lotto.games import GAMES
from tests.conftest import _fair_draws


def _crowded_frames(seed=3, draws=900):
    """Draws where a jackpot winner appears far more often on calendar-heavy draws."""
    rng = np.random.default_rng(seed)
    frames = {}
    for game_id in ("655", "649"):
        cfg = GAMES[game_id]
        frame = _fair_draws(cfg, n=draws, seed=seed)
        tickets = frame[[f"n{i}" for i in range(1, 7)]].to_numpy()
        calendar = (tickets <= 31).sum(1)
        p = np.where(calendar >= 5, 0.30, 0.04)
        frame["winners_jackpot"] = (rng.random(len(frame)) < p).astype(int)
        frames[game_id] = frame
    return frames


@pytest.fixture(scope="module")
def crowd_model():
    return fit_crowd_model(_crowded_frames())


def test_ticket_features_count_calendar_and_patterns():
    row = ticket_features(np.array([[3, 4, 5, 20, 35, 50]]), pool=55)[0]
    assert list(row[:5]) == [4, 3, 2, 4, 3]  # low31, low12, consecutive pairs, mult of 5, odd
    assert row[5] == pytest.approx(50 / 55) and row[6] == pytest.approx(47 / 55)


def test_crowd_model_learns_the_planted_calendar_effect(crowd_model):
    assert crowd_model.cv_auc > crowd_model.game_only_auc + 0.08
    assert crowd_model.coefficients["low31"] > 0
    rates = crowd_model.quintile_winner_rate
    assert rates[-1] > 2 * rates[0]
    calendar, high = np.array([[1, 5, 9, 14, 22, 30]]), np.array([[33, 38, 41, 44, 50, 54]])
    assert crowd_model.crowd_index(calendar, "655")[0] > 1 > crowd_model.crowd_index(high, "655")[0]
    assert crowd_model.share_factor(high, "655")[0] > crowd_model.share_factor(calendar, "655")[0]
    assert crowd_model.share_gain_pct(high, "655") > 0 > crowd_model.share_gain_pct(calendar, "655")


def test_crowd_model_needs_enough_winner_draws():
    frames = _crowded_frames(draws=200)
    for frame in frames.values():
        frame["winners_jackpot"] = 0
    assert fit_crowd_model(frames) is None
    assert fit_crowd_model({}) is None


def test_crowd_model_rejects_games_it_was_not_fitted_on(crowd_model):
    with pytest.raises(ValueError):
        crowd_model.probability(np.array([[1, 2, 3, 4, 5, 6]]), "642")


def test_crowd_avoiding_tickets_are_valid_and_less_crowded(crowd_model):
    cfg = GAMES["655"]
    tickets = crowd_avoiding_tickets(crowd_model, cfg, n=8, seed=5)
    assert len({tuple(t) for t in tickets}) == 8
    for ticket in tickets:
        assert ticket == sorted(set(ticket)) and len(ticket) == 6
        assert all(1 <= v <= cfg.pool for v in ticket)
    random = random_tickets(cfg, 500, np.random.default_rng(1))
    assert crowd_model.crowd_index(np.array(tickets), "655").mean() < \
        crowd_model.crowd_index(random, "655").mean() * 0.5
    assert tickets == crowd_avoiding_tickets(crowd_model, cfg, n=8, seed=5)


def test_crowd_avoiding_tickets_respect_weights(crowd_model):
    cfg = GAMES["655"]
    weights = np.full(cfg.pool, 1e-6)
    weights[:12] = 1.0  # only numbers 1-12 are plausible
    ticket = crowd_avoiding_tickets(crowd_model, cfg, n=1, candidates=3000, seed=2, weights=weights)[0]
    assert all(v <= 15 for v in ticket)


def test_crowd_games_are_the_unordered_six_number_games():
    assert {g.id for g in crowd_games()} == {"658", "655", "649", "645", "642"}


def _fair(game_id, draws=260, seed=0):
    return _fair_draws(GAMES[game_id], n=draws, seed=seed)


def test_expected_matches_match_the_game_rules():
    assert arena._expected_matches(GAMES["655"]) == pytest.approx(36 / 55)
    assert arena._expected_matches(GAMES["3d"]) == pytest.approx(0.3)
    assert arena._expected_matches(GAMES["2d"]) == pytest.approx(2 / 31)


@pytest.mark.parametrize("game_id", ["655", "3d", "2d"])
def test_sampled_tickets_are_valid_distinct_and_weighted(game_id):
    cfg = GAMES[game_id]
    size = cfg.pool - cfg.low_boundary + 1
    rng = np.random.default_rng(4)
    weights = np.ones((cfg.picks if cfg.ordered else 1, size))
    tickets = arena.sample_tickets(cfg, weights, 12, rng)
    assert len({tuple(t) for t in tickets}) == 12
    for ticket in tickets:
        assert len(ticket) == cfg.picks
        assert all(cfg.low_boundary <= v <= cfg.pool for v in ticket)
        assert cfg.ordered or ticket == sorted(set(ticket))
    heavy = np.full_like(weights, 1e-9)
    heavy[..., 2] = 1.0
    skewed = arena.sample_tickets(cfg, heavy, 1, np.random.default_rng(4))[0]
    assert cfg.low_boundary + 2 in skewed


def test_sampler_caps_requests_at_the_ticket_space():
    cfg = GAMES["655"]
    from dataclasses import replace
    tiny = replace(cfg, pool=4, picks=3)
    tickets = arena.sample_tickets(tiny, np.ones((1, 4)), 50, np.random.default_rng(0))
    assert len(tickets) == 4


def test_arena_only_uses_history_before_each_draw(monkeypatch):
    seen = []
    real = arena._counts

    def spy(history, cfg, last):
        seen.append(len(history))
        return real(history, cfg, last)

    monkeypatch.setattr(arena, "_counts", spy)
    frame = _fair("655", draws=200)
    result = arena.run_arena(frame, GAMES["655"], strategies=["hot"], test_draws=40)
    assert seen == list(range(160, 200))
    assert result.meta["test_draws"] == 40


def test_fair_draws_show_no_significant_strategy():
    result = arena.run_arena(_fair("655", draws=420, seed=11), GAMES["655"],
                             strategies=["uniform", "hot", "overdue"], test_draws=300)
    assert result.conclusion.startswith("No strategy beat random")
    assert (result.table["p_adjusted"] >= 0.05).all()
    assert {"edge_pct", "z", "p_value", "p_adjusted", "draws_with_3plus"} <= set(result.table.columns)


def test_arena_detects_a_planted_number_bias():
    rng = np.random.default_rng(8)
    frame = _fair("655", draws=420, seed=5)
    cols = [f"n{i}" for i in range(1, 7)]
    for index in frame.index:
        favoured = rng.choice(np.arange(1, 7), size=3, replace=False)
        rest = rng.choice(np.arange(7, 56), size=3, replace=False)
        frame.loc[index, cols] = np.sort(np.concatenate([favoured, rest]))
    result = arena.run_arena(frame, GAMES["655"], strategies=["uniform", "hot"], test_draws=300)
    hot = result.table.set_index("key").loc["hot"]
    assert hot["z"] > 5 and hot["p_adjusted"] < 0.001 and hot["edge_pct"] > 20
    assert "Hot numbers" in result.conclusion
    assert abs(result.table.set_index("key").loc["uniform", "z"]) < 3.5


def test_arena_runs_ordered_games_and_multislot_filtering():
    frame = _fair("3d", draws=220, seed=2)
    other = frame.copy()
    other["draw_time"] = "14:00"
    result = arena.run_arena(pd.concat([frame, other]), GAMES["3d"],
                             strategies=["uniform", "hot", "overdue"], test_draws=100,
                             draw_time="21:00")
    assert result.meta["expected_matches"] == pytest.approx(0.3)
    assert set(result.table["key"]) == {"uniform", "hot", "overdue"}
    assert (result.table["draws"] == 100).all()


def test_arena_with_ml_and_crowd_strategies(crowd_model):
    frame = _crowded_frames(draws=260)["655"]
    result = arena.run_arena(frame, GAMES["655"], test_draws=30, tickets_per_draw=3,
                             crowd=crowd_model, ml_refit_every=15)
    table = result.table.set_index("key")
    assert set(table.index) == {"uniform", "hot", "overdue", "ml", "lucky", "hybrid", "crowd", "ml_crowd"}
    assert table.loc["crowd", "share_gain_pct"] > 0
    assert table.loc["crowd", "jackpot_share_kept"] > table.loc["uniform", "jackpot_share_kept"]


def test_arena_validates_inputs(crowd_model):
    frame = _fair("655", draws=200)
    with pytest.raises(ValueError, match="Unknown strategies"):
        arena.run_arena(frame, GAMES["655"], strategies=["magic"])
    with pytest.raises(ValueError, match="crowd model"):
        arena.run_arena(frame, GAMES["655"], strategies=["crowd"])
    with pytest.raises(ValueError, match="at least"):
        arena.run_arena(frame.head(40), GAMES["655"], strategies=["uniform"])
    assert "crowd" not in arena.available_strategies(GAMES["3d"], crowd_model)
    assert "crowd" in arena.available_strategies(GAMES["655"], crowd_model)
    assert "crowd" not in arena.available_strategies(GAMES["655"], None)


@pytest.mark.parametrize("strategy", ["uniform", "hot", "overdue", "ml", "crowd", "ml_crowd"])
def test_next_draw_tickets_are_valid_for_every_strategy(crowd_model, strategy):
    cfg = GAMES["655"]
    frame = _crowded_frames(draws=260)["655"]
    tickets = arena.next_draw_tickets(frame, cfg, strategy, n=4, crowd=crowd_model)
    assert len({tuple(t) for t in tickets}) == 4
    assert all(t == sorted(set(t)) and len(t) == 6 and 1 <= min(t) and max(t) <= 55 for t in tickets)
    assert tickets == arena.next_draw_tickets(frame, cfg, strategy, n=4, crowd=crowd_model)


def test_next_draw_tickets_reject_unavailable_strategy():
    with pytest.raises(ValueError, match="unavailable"):
        arena.next_draw_tickets(_fair("3d", draws=120), GAMES["3d"], "crowd", n=3)


def test_overlap_limit_makes_the_crowd_set_diverse(crowd_model):
    cfg = GAMES["655"]
    plain = crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=5)
    diverse = crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=5, max_overlap=2)
    def widest(tickets):
        return max(len(set(a) & set(b)) for i, a in enumerate(tickets) for b in tickets[i + 1:])
    assert widest(plain) > 2 and len(diverse) == 10 and widest(diverse) <= 2
    assert diverse[0] == plain[0]


def test_ai_shortlist_is_diverse_and_carries_context(crowd_model):
    cfg = GAMES["655"]
    frame = _crowded_frames(draws=260)["655"]
    menu = arena.ai_shortlist(frame, cfg, crowd_model, n_candidates=15, max_overlap=3)
    tickets = [c["ticket"] for c in menu["candidates"]]
    assert len(tickets) == 15 and menu["max_overlap"] == 3
    assert all(len(set(a) & set(b)) <= 3 for i, a in enumerate(tickets) for b in tickets[i + 1:])
    indexes = [c["crowd_index"] for c in menu["candidates"]]
    assert indexes == sorted(indexes) and indexes[0] < 1
    assert all(0 < c["share_kept"] <= 1 and c["ml_support"] > 0 for c in menu["candidates"])
    numbers = {v for t in tickets for v in t}
    assert set(menu["number_context"]) == numbers
    sample = menu["number_context"][min(numbers)]
    assert set(sample) == {"ml_vs_average", "freq_last100", "draws_since_seen"}


def test_ai_shortlist_needs_a_crowd_model_and_unordered_game(crowd_model):
    with pytest.raises(ValueError, match="unavailable"):
        arena.ai_shortlist(_fair("655", draws=120), GAMES["655"], None)
    with pytest.raises(ValueError, match="unavailable"):
        arena.ai_shortlist(_fair("3d", draws=120), GAMES["3d"], crowd_model)


def test_quantile_selection_spreads_tickets_instead_of_hugging_the_top(crowd_model):
    cfg = GAMES["655"]
    extreme = crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=3)
    spread = crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=3, quantile=0.2)
    random = random_tickets(cfg, 500, np.random.default_rng(1))
    calendar = lambda tickets: float((np.array(tickets) <= 31).sum(1).mean())
    assert calendar(extreme) <= 1.0 < calendar(spread) < calendar(random)
    assert len({v for t in spread for v in t}) > len({v for t in extreme for v in t})
    index = crowd_model.crowd_index(np.array(spread), "655")
    assert (index < 0.8).all() and list(index) == sorted(index)
    assert spread != crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=4, quantile=0.2)
    assert spread == crowd_avoiding_tickets(crowd_model, cfg, n=10, seed=3, quantile=0.2)


def test_quantile_must_be_a_fraction(crowd_model):
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="quantile"):
            crowd_avoiding_tickets(crowd_model, GAMES["655"], n=3, quantile=bad)


def test_arena_crowd_strategies_cover_the_pool(crowd_model):
    cfg = GAMES["655"]
    frame = _crowded_frames(draws=260)["655"]
    for strategy in ("crowd", "ml_crowd"):
        tickets = arena.next_draw_tickets(frame, cfg, strategy, n=10, crowd=crowd_model)
        assert (np.array(tickets) <= 31).sum(1).mean() > 1.0
        assert len({v for t in tickets for v in t}) > 25
    menu = arena.ai_shortlist(frame, cfg, crowd_model, n_candidates=30)
    assert len({v for c in menu["candidates"] for v in c["ticket"]}) > 35


def test_strategies_do_not_share_sampling_noise(crowd_model):
    cfg = GAMES["655"]
    frame = _crowded_frames(draws=260)["655"]
    names = ["uniform", "hot", "overdue", "ml", "crowd", "ml_crowd"]
    sets = {s: arena.next_draw_tickets(frame, cfg, s, n=11, crowd=crowd_model, seed=42) for s in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not [t for t in sets[a] if t in sets[b]], f"{a} and {b} returned identical tickets"
            overlap = np.mean([max(len(set(x) & set(y)) for y in sets[b]) for x in sets[a]])
            assert overlap < 3.2, f"{a} and {b} look like the same tickets"


def test_next_draw_tickets_vary_with_seed_and_repeat_for_the_same_seed(crowd_model):
    cfg = GAMES["655"]
    frame = _crowded_frames(draws=260)["655"]
    for strategy in ("uniform", "hot", "ml", "crowd", "ml_crowd"):
        first = arena.next_draw_tickets(frame, cfg, strategy, n=5, crowd=crowd_model, seed=1)
        assert first == arena.next_draw_tickets(frame, cfg, strategy, n=5, crowd=crowd_model, seed=1)
        assert first != arena.next_draw_tickets(frame, cfg, strategy, n=5, crowd=crowd_model, seed=2)
