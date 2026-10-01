"""Synthetic-only tests: no database, network, or hosted model calls."""
from dataclasses import replace
import json
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from src.lotto import lean_ml as ml
from src.lotto.games import GAMES


def _draws(game, count=150, seed=1):
    rng = np.random.default_rng(seed)
    pool = np.arange(game.low_boundary, game.pool + 1)
    values = np.array([rng.choice(pool, game.picks, replace=game.ordered) for _ in range(count)])
    frame = pd.DataFrame(values, columns=[f"n{i}" for i in range(1, game.picks + 1)])
    frame["draw_date"] = pd.date_range("2020-01-01", periods=count).strftime("%Y-%m-%d")
    frame["draw_time"] = "21:00"
    frame["game"] = game.id
    return frame


def _tickets_valid(tickets, game, count):
    assert len(tickets) == len({tuple(t) for t in tickets}) == count
    for ticket in tickets:
        assert len(ticket) == game.picks
        assert all(type(number) is int and game.low_boundary <= number <= game.pool for number in ticket)
        if not game.ordered:
            assert ticket == sorted(set(ticket))


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
def test_all_games_finite_interface_and_determinism(game):
    draws = _draws(game, 100)
    original = draws.copy(deep=True)
    result = ml.generate_ml_combinations(draws, game, n=10)
    assert set(result) == {"method", "combinations", "scores", "evaluation", "feature_weights", "reason", "training_draws"}
    assert result == ml.generate_ml_combinations(draws.sample(frac=1, random_state=3), game, n=10)
    pd.testing.assert_frame_equal(draws, original)
    assert result["method"] == "logistic_regression"
    assert result["training_draws"] == 70
    _tickets_valid(result["combinations"], game, 10)
    size = game.pool - game.low_boundary + 1
    assert len(result["scores"]) == size * (game.picks if game.ordered else 1)
    for row in result["scores"]:
        assert ml.SCORE_FLOOR <= row["score"] <= 1
        assert type(row["number"]) is int
        assert ("position" in row) == game.ordered
    expected_features = set(ml.FEATURE_NAMES)
    if game.ordered:
        expected_features |= {f"position_{i}" for i in range(1, game.picks + 1)}
    assert set(result["feature_weights"]) == expected_features
    assert np.isfinite(list(result["feature_weights"].values())).all()
    evaluation = result["evaluation"]
    assert evaluation["status"] == "evaluated"
    assert evaluation["test_draws"] == 14
    assert evaluation["train_draws"] == 56
    assert evaluation["usable_draws"] == 100
    assert evaluation["training_end"]["draw_index"] == 85
    assert evaluation["test_start"]["draw_index"] == 86
    prior = (1 if game.ordered else game.picks) / size
    assert evaluation["baseline_brier"] == pytest.approx(prior * (1 - prior))
    assert evaluation["mean_brier_improvement"] == pytest.approx(evaluation["baseline_brier"] - evaluation["model_brier"])
    assert evaluation["brier_skill"] == pytest.approx(1 - evaluation["model_brier"] / evaluation["baseline_brier"])
    assert evaluation["evidence_of_improvement"] == (evaluation["improvement_ci95"][0] > 0)
    assert "do not prove a winning advantage" in result["reason"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("count", [0, 29, 30, 99])
def test_short_history_has_no_misleading_scores(count):
    game = GAMES["655"]
    draws = _draws(game, count) if count else pd.DataFrame()
    result = ml.generate_ml_combinations(draws, game)
    assert result["method"] == "uniform_fallback"
    assert result["scores"] == []
    assert result["feature_weights"] == {}
    assert result["training_draws"] == 0
    evaluation = result["evaluation"]
    assert evaluation["status"] == "insufficient_history"
    assert evaluation["test_draws"] == 0
    assert evaluation["model_brier"] is evaluation["baseline_brier"] is None
    assert evaluation["brier_skill"] is evaluation["mean_brier_improvement"] is None
    assert evaluation["improvement_ci95"] == []
    assert evaluation["training_end"] is evaluation["test_start"] is None
    assert not evaluation["evidence_of_improvement"]
    _tickets_valid(result["combinations"], game, 5)
    assert result["combinations"] == ml.generate_ml_combinations(pd.DataFrame(), game)["combinations"]


@pytest.mark.parametrize("ordered", [False, True])
def test_small_space_exhaustion_with_extreme_scores(ordered):
    game = replace(GAMES["3d"] if ordered else GAMES["655"], pool=1 if ordered else 3, picks=2)
    weights = np.array([[1, 0], [0, 1]]) if ordered else np.array([[1, 0, 0]])
    result = ml._sample_combinations(game, weights, 100, 4)
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)} if ordered else {(1, 2), (1, 3), (2, 3)}
    assert {tuple(t) for t in result} == expected
    fallback = ml.generate_ml_combinations(pd.DataFrame(), game, n=100)
    assert {tuple(t) for t in fallback["combinations"]} == expected


def test_scores_actually_weight_generation():
    game = replace(GAMES["655"], pool=4, picks=1)
    biased = [ml._sample_combinations(game, np.array([1, 0, 0, 0]), 1, seed)[0][0] for seed in range(80)]
    uniform = [ml._sample_combinations(game, np.ones(4), 1, seed)[0][0] for seed in range(80)]
    assert biased.count(1) == 80
    assert uniform.count(1) < 40
    assert ml._sample_combinations(game, np.array([0, 0, 0, 1]), 1, 42) == [[4]]


def test_unordered_weighting_includes_high_number_not_just_first_position():
    game = replace(GAMES["655"], pool=8, picks=2)
    weights = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    for seed in range(20):
        assert ml._sample_combinations(game, weights, 1, seed) == [[7, 8]]


def test_learned_scores_drive_ordered_generation_end_to_end(monkeypatch):
    game = replace(GAMES["3d"], pool=2)
    draws = _draws(game, 120)
    draws[["n1", "n2", "n3"]] = [2, 0, 2]
    sample = ml._sample_combinations
    received = []

    def spy(cfg, scores, n, seed):
        received.append(np.asarray(scores).copy())
        return sample(cfg, scores, n, seed)

    monkeypatch.setattr(ml, "_sample_combinations", spy)
    result = ml.generate_ml_combinations(draws, game, n=1)
    assert result["method"] == "logistic_regression"
    np.testing.assert_array_equal(received[0], [row["score"] for row in result["scores"]])
    assert received[0].reshape(3, 3).argmax(axis=1).tolist() == [2, 0, 2]
    assert result["combinations"] == [[2, 0, 2]]
    assert result["evaluation"]["evidence_of_improvement"]


def test_ordered_position_weights_and_repeats():
    game = replace(GAMES["3d"], pool=2)
    scores = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1]])
    assert ml._sample_combinations(game, scores, 1, 1) == [[2, 0, 2]]


@pytest.mark.parametrize("game", [GAMES["655"], GAMES["3d"]])
def test_features_ignore_current_and_future_outcomes(game):
    draws = _draws(game)
    changed = draws.copy()
    cols = [f"n{i}" for i in range(1, game.picks + 1)]
    changed.loc[100:, cols] = _draws(game, 50, seed=88)[cols].to_numpy()
    first = ml._build_snapshots(ml._prepare_draws(draws, game), game)
    second = ml._build_snapshots(ml._prepare_draws(changed, game), game)
    # Features for draw 100 cannot see that draw's altered outcome.
    np.testing.assert_array_equal(first.X[:71], second.X[:71])
    np.testing.assert_array_equal(first.y[:70], second.y[:70])
    assert not np.array_equal(first.X[71:], second.X[71:])
    assert not np.array_equal(first.next_X, second.next_X)
    prefix = ml._build_snapshots(ml._prepare_draws(draws.iloc[:100], game), game)
    np.testing.assert_array_equal(prefix.next_X, first.X[70])


def test_three_smoothed_features_and_position_one_hot():
    game = replace(GAMES["3d"], picks=2, pool=2)
    draws = _draws(game, 40)
    draws[["n1", "n2"]] = [2, 0]
    snapshots = ml._build_snapshots(ml._prepare_draws(draws, game), game)
    assert snapshots.X.shape == (10, 6, 5)
    np.testing.assert_array_equal(snapshots.y[0], [0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(snapshots.X[0, 2, :3], [(10 + 1 / 3) / 11, (30 + 1 / 3) / 31, (30 + 1 / 3) / 31])
    np.testing.assert_allclose(snapshots.X[0, 0, :3], [(1 / 3) / 11, (1 / 3) / 31, (1 / 3) / 31])
    np.testing.assert_array_equal(snapshots.X[0, :, 3:], [[1, 0]] * 3 + [[0, 1]] * 3)
    assert len(snapshots.feature_names) == 5


def test_unordered_frequencies_use_inclusion_not_flattened_digit_frequency():
    game = replace(GAMES["655"], pool=4, picks=2)
    draws = _draws(game, 40)
    draws[["n1", "n2"]] = [1, 4]
    snapshots = ml._build_snapshots(ml._prepare_draws(draws, game), game)
    assert snapshots.X.shape == (10, 4, 3)
    np.testing.assert_allclose(snapshots.X[0, 0], [10.5 / 11, 30.5 / 31, 30.5 / 31])
    np.testing.assert_array_equal(snapshots.y[0], [1, 0, 0, 1])


def test_chronological_groups_frozen_scaler_and_separate_production_fit(monkeypatch):
    game = GAMES["3d"]
    draws = _draws(game, 150)
    snapshots = ml._build_snapshots(ml._prepare_draws(draws, game), game)
    original_fit = ml._fit_classifier
    calls = []

    def spy(X, y):
        model = original_fit(X, y)
        calls.append((X.copy(), y.copy(), model))
        return model

    monkeypatch.setattr(ml, "_fit_classifier", spy)
    result = ml.generate_ml_combinations(draws, game)
    assert len(calls) == 2  # One shared classifier each for evaluation and production.
    train_X, train_y, evaluation_model = calls[0]
    full_X, full_y, production_model = calls[1]
    assert evaluation_model is not production_model
    assert train_X.shape == (96, 30, 6)
    np.testing.assert_array_equal(train_X, snapshots.X[:96])
    np.testing.assert_array_equal(train_y, snapshots.y[:96])
    np.testing.assert_array_equal(full_X, snapshots.X)
    np.testing.assert_array_equal(full_y, snapshots.y)
    scaler = evaluation_model.named_steps["scaler"]
    assert scaler.n_samples_seen_ == 96 * 30
    np.testing.assert_allclose(scaler.mean_, train_X.reshape(-1, 6).mean(axis=0))
    test_X, test_y = snapshots.X[96:], snapshots.y[96:]
    prediction = evaluation_model.predict_proba(test_X.reshape(-1, 6))[:, 1].reshape(test_y.shape)
    errors = ((prediction - test_y) ** 2).mean(axis=1)
    baseline_errors = ((0.1 - test_y) ** 2).mean(axis=1)
    evaluation = result["evaluation"]
    assert evaluation["model_brier"] == pytest.approx(errors.mean())
    np.testing.assert_allclose(evaluation["improvement_ci95"], ml._improvement_interval(baseline_errors - errors))
    assert evaluation["training_end"] == snapshots.provenance[95]
    assert evaluation["test_start"] == snapshots.provenance[96]
    # Returned scores come from all-history production fit, NOT the held-out fit.
    raw_scores = production_model.predict_proba(snapshots.next_X)[:, 1]
    np.testing.assert_allclose([s["score"] for s in result["scores"]], np.maximum(raw_scores, ml.SCORE_FLOOR))


def test_test_outcomes_do_not_change_evaluation_fit(monkeypatch):
    game = GAMES["655"]
    draws = _draws(game, 100)
    changed = draws.copy()
    cols = [f"n{i}" for i in range(1, 7)]
    changed.loc[86:, cols] = [1, 2, 3, 4, 5, 6]
    models = []
    fit = ml._fit_classifier

    def spy(X, y):
        model = fit(X, y)
        models.append(model)
        return model

    monkeypatch.setattr(ml, "_fit_classifier", spy)
    ml.generate_ml_combinations(draws, game)
    ml.generate_ml_combinations(changed, game)
    np.testing.assert_array_equal(models[0].named_steps["scaler"].mean_, models[2].named_steps["scaler"].mean_)
    np.testing.assert_array_equal(models[0].named_steps["classifier"].coef_, models[2].named_steps["classifier"].coef_)
    assert not np.array_equal(models[1].named_steps["classifier"].coef_, models[3].named_steps["classifier"].coef_)


def test_draw_level_t_interval_not_candidate_level_normal_interval():
    improvements = np.linspace(-0.1, 0.2, 14)
    low, high = ml._improvement_interval(improvements)
    margin = 2.160369 * improvements.std(ddof=1) / sqrt(14)
    assert low == pytest.approx(improvements.mean() - margin, abs=1e-5)
    assert high == pytest.approx(improvements.mean() + margin, abs=1e-5)
    assert ml._improvement_interval(np.zeros(14)) == [0, 0]


def test_game_slot_filtering_before_window_and_stable_chronology():
    game = GAMES["3d"]
    wanted = _draws(game, 120)
    other_slot = wanted.assign(draw_time="14:00")
    other_game = wanted.assign(game="655")
    mixed = pd.concat([wanted, other_slot, other_game], ignore_index=True).sample(frac=1, random_state=8)
    result = ml.generate_ml_combinations(mixed, game, window=100, draw_time="21:00")
    assert result == ml.generate_ml_combinations(wanted.tail(100), game, draw_time="21:00")
    clean = ml._prepare_draws(mixed, game)
    assert clean["draw_time"].iloc[:4].tolist() == ["14:00", "21:00", "14:00", "21:00"]
    fallback = ml.generate_ml_combinations(mixed, game)
    assert fallback["method"] == "uniform_fallback"
    assert "Multiple draw slots" in fallback["reason"]
    for frame, slot in [(mixed, "17:00"), (wanted.drop(columns="draw_time"), "21:00")]:
        assert ml.generate_ml_combinations(frame, game, draw_time=slot)["evaluation"]["usable_draws"] == 0
    assert ml.generate_ml_combinations(wanted, game, window=99)["method"] == "uniform_fallback"
    assert ml.generate_ml_combinations(wanted.drop(columns=["game", "draw_time"]), game)["method"] == "logistic_regression"


def test_bad_rows_nullable_numbers_and_missing_chronology():
    game = replace(GAMES["655"], pool=5, picks=2)
    bad = pd.DataFrame({
        "draw_date": pd.date_range("2020-01-01", periods=10).strftime("%Y-%m-%d"),
        "n1": [1, 1, 0, 1.2, np.inf, pd.NA, True, "2", 1, 1],
        "n2": [2, 1, 3, 3, 3, 3, 3, "4", 9, 3],
    })
    bad.loc[9, "draw_date"] = "not-a-date"
    clean = ml._prepare_draws(bad, game)
    assert clean[["n1", "n2"]].values.tolist() == [[1, 2], [2, 4]]
    nullable = _draws(game, 100)
    nullable[["n1", "n2"]] = nullable[["n1", "n2"]].astype("Int64")
    nullable.loc[0, "n1"] = pd.NA
    result = ml.generate_ml_combinations(nullable, game)
    assert result["evaluation"]["usable_draws"] == 99
    assert result["method"] == "uniform_fallback"
    for frame in [nullable.drop(columns="draw_date"), nullable.drop(columns="n2"), nullable.assign(draw_time="25:00")]:
        assert ml.generate_ml_combinations(frame, game)["evaluation"]["usable_draws"] == 0


def test_published_positions_complete_per_row_otherwise_fallback():
    game = replace(GAMES["3d"], picks=2)
    draws = _draws(game, 5)
    draws[["n1", "n2"]] = [1, 2]
    draws["pub_1"] = [9, 8, pd.NA, 10, 7]
    draws["pub_2"] = [0, pd.NA, 6, 1, 7]
    clean = ml._prepare_draws(draws, game)
    assert clean[["n1", "n2"]].values.tolist() == [[9, 0], [1, 2], [1, 2], [7, 7]]
    published_only = ml._prepare_draws(draws.drop(columns=["n1", "n2"]), game)
    assert published_only[["n1", "n2"]].values.tolist() == [[9, 0], [7, 7]]
    incomplete_columns = ml._prepare_draws(draws.drop(columns="pub_2"), game)
    assert incomplete_columns[["n1", "n2"]].values.tolist() == [[1, 2]] * 5


def test_duplicate_identity_cannot_cross_draw_group_boundary():
    game = replace(GAMES["655"], pool=5, picks=2)
    draws = _draws(game, 100)
    duplicate = pd.concat([draws, draws.iloc[[85, 86]]], ignore_index=True)
    assert ml.generate_ml_combinations(duplicate, game) == ml.generate_ml_combinations(draws, game)
    reversed_draw = draws.iloc[[85]].copy()
    reversed_draw[["n1", "n2"]] = reversed_draw[["n2", "n1"]].to_numpy()
    assert len(ml._prepare_draws(pd.concat([draws, reversed_draw], ignore_index=True), game)) == 100
    conflict = draws.iloc[[85]].copy()
    conflict[["n1", "n2"]] = [1, 5] if draws.loc[85, ["n1", "n2"]].tolist() != [1, 5] else [2, 4]
    clean = ml._prepare_draws(pd.concat([draws, conflict], ignore_index=True), game)
    assert len(clean) == 99
    assert clean["draw_date"].dt.strftime("%Y-%m-%d").ne(draws.loc[85, "draw_date"]).all()


@pytest.mark.parametrize("ordered", [False, True])
def test_single_class_training_falls_back(ordered):
    game = replace(GAMES["3d"] if ordered else GAMES["655"], pool=0 if ordered else 2, picks=2)
    result = ml.generate_ml_combinations(_draws(game, 100), game, n=10)
    assert result["method"] == "uniform_fallback"
    assert "single class" in result["reason"]
    assert result["scores"] == []
    assert result["feature_weights"] == {}
    assert result["evaluation"]["model_brier"] is None
    assert len(result["combinations"]) == 1


@pytest.mark.parametrize("kwargs", [{"n": -1}, {"n": 1.5}, {"n": True}, {"window": 0}, {"window": -1}, {"window": 3.5}, {"window": False}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ml.generate_ml_combinations(pd.DataFrame(), GAMES["655"], **kwargs)


@pytest.mark.parametrize("game", [replace(GAMES["655"], pool=0), replace(GAMES["655"], picks=0), replace(GAMES["655"], pool=5), replace(GAMES["655"], picks=2.5)])
def test_invalid_game_configuration(game):
    with pytest.raises(ValueError):
        ml.generate_ml_combinations(pd.DataFrame(), game)


def test_zero_requested_tickets_and_seed_changes_only_sampling():
    game = GAMES["655"]
    draws = _draws(game, 100)
    first = ml.generate_ml_combinations(draws, game, seed=1)
    second = ml.generate_ml_combinations(draws, game, seed=9)
    assert first["combinations"] != second["combinations"]
    for key in ("scores", "feature_weights", "evaluation"):
        assert first[key] == second[key]
    assert ml.generate_ml_combinations(draws, game, n=0)["combinations"] == []
    assert ml.generate_ml_combinations(pd.DataFrame(), game, n=0)["combinations"] == []