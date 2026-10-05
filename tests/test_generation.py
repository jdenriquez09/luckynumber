"""Generation contracts: valid tickets, no learned weights, no live AI calls."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.lotto.azure_ai import AzureAIClient
from src.lotto.games import GAMES
from src.lotto.predict import DISCLAIMER, number_heat_map, random_combinations, suggest_combinations
from tests.conftest import _fair_draws


@pytest.mark.parametrize("game", GAMES.values(), ids=GAMES.keys())
def test_random_ticket_contract(game):
    tickets = random_combinations(game, n=50, seed=42)
    assert tickets == random_combinations(game, n=50, seed=42)
    assert len(tickets) == len({tuple(t) for t in tickets}) == 50
    for ticket in tickets:
        assert len(ticket) == game.picks
        assert all(game.low_boundary <= number <= game.pool for number in ticket)
        if not game.ordered:
            assert ticket == sorted(set(ticket))


def test_ordered_ticket_space_includes_zero_repeats_and_order():
    game = replace(GAMES["3d"], pool=1, picks=2)
    assert {tuple(t) for t in random_combinations(game, n=10, seed=2)} == {
        (0, 0), (0, 1), (1, 0), (1, 1)
    }


def test_unordered_ticket_space_exhaustion():
    game = replace(GAMES["655"], pool=3, picks=2)
    assert {tuple(t) for t in random_combinations(game, n=10, seed=2)} == {
        (1, 2), (1, 3), (2, 3)
    }


def test_history_never_weights_explicitly_seeded_sampling(fair_draws_655):
    biased = fair_draws_655.copy()
    for pos in range(1, 7):
        biased[f"n{pos}"] = pos
    assert suggest_combinations(biased, GAMES["655"], seed=9) == suggest_combinations(
        fair_draws_655, GAMES["655"], seed=9
    )


def test_snapshot_seed_and_heat_are_chronological(fair_draws_655):
    shuffled = fair_draws_655.sample(frac=1, random_state=5)
    assert suggest_combinations(shuffled, GAMES["655"]) == suggest_combinations(
        fair_draws_655, GAMES["655"]
    )
    pd.testing.assert_frame_equal(
        number_heat_map(shuffled, GAMES["655"], window=20),
        number_heat_map(fair_draws_655, GAMES["655"], window=20),
    )


def test_intraday_ordering_and_empty_slot():
    game = replace(GAMES["655"], pool=3, picks=1)
    draws = pd.DataFrame({"draw_date": ["2026-01-01"] * 3,
                          "draw_time": ["21:00", "14:00", "17:00"], "n1": [3, 1, 2]})
    heat = number_heat_map(draws, game, window=1).set_index("number")
    assert heat.loc[3, "current_gap"] == 0
    assert heat.loc[1, "current_gap"] == 2
    assert suggest_combinations(draws, game, draw_time="12:00") == []
    assert suggest_combinations(pd.DataFrame(), game) == []


def test_fair_fixtures_follow_game_rules():
    unordered = _fair_draws(GAMES["655"], n=100)
    assert (unordered[[f"n{i}" for i in range(1, 7)]].nunique(axis=1) == 6).all()
    ordered = _fair_draws(GAMES["3d"], n=100)
    assert (ordered[["n1", "n2", "n3"]].nunique(axis=1) < 3).any()
    assert (ordered["n1"] > ordered["n2"]).any()


def _client(monkeypatch, text, finish_reason="stop"):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test")
    client = AzureAIClient()
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=text)
    )])
    monkeypatch.setattr(client, "_client", lambda: sdk)
    return client, sdk


def _slot(label="Evening", ordered=False):
    rows = [{"number": i, "total_draws": i + 1} for i in range(10)]
    if ordered:
        rows = [dict(row, position=pos) for pos in range(1, 4) for row in rows]
    return {"label": label, "heat_map_rows": rows, "total_draws": 100}


def test_ai_cannot_replace_tickets_or_disclaimer(monkeypatch):
    client, sdk = _client(monkeypatch,
        "SLOT: Evening\nCOMBINATION: 999, 999, 999\nREASONING: Uniform random example.\n"
        "DISCLAIMER: Guaranteed winner!")
    result = client.interpret_lotto_stats("Example", [_slot()], "uniform", "gaps", 3,
                                        prior_run_feedback=[{"secret": "NEAR_MISS_SENTINEL"}],
                                        ml_scores={999: 1.0})["Evening"]
    assert len(result["combination"]) == len(set(result["combination"])) == 3
    assert all(0 <= n <= 9 for n in result["combination"])
    assert result["disclaimer"] == DISCLAIMER
    assert "NEAR_MISS_SENTINEL" not in client.last_prompt
    assert "ml_prob" not in client.last_prompt
    assert "Chi-square summary (descriptive): uniform" in client.last_prompt
    assert sdk.chat.completions.create.call_count == 1


def test_ai_ordered_sampling_retains_repeats(monkeypatch):
    client, _ = _client(monkeypatch, "SLOT: Evening\nREASONING: Random digits.")
    slot = _slot(ordered=True)
    slot["heat_map_rows"] = [r for r in slot["heat_map_rows"] if r["number"] == 0]
    result = client.interpret_lotto_stats("Digits", [slot], "uniform", "repeat", 3)
    assert result["Evening"]["combination"] == [0, 0, 0]


@pytest.mark.parametrize("text,finish", [
    ("", "stop"), ("SLOT: Evening\nDISCLAIMER: test", "stop"),
    ("SLOT: Other\nREASONING: test", "stop"),
    ("SLOT: Evening\nREASONING: partial", "length"),
])
def test_ai_rejects_incomplete_explanations(monkeypatch, text, finish):
    client, _ = _client(monkeypatch, text, finish)
    with pytest.raises(RuntimeError):
        client.interpret_lotto_stats("Example", [_slot()], "uniform", "gaps", 3)


def test_ai_multislot_results(monkeypatch):
    client, _ = _client(monkeypatch,
        "SLOT: Morning\nREASONING: Random example.\nSLOT: Evening\nREASONING: Another example.")
    result = client.interpret_lotto_stats("Example", [_slot("Morning"), _slot()], "ok", "ok", 3)
    assert set(result) == {"Morning", "Evening"}
    assert all(len(r["combination"]) == 3 for r in result.values())


def _generation(ordered=False, fallback=False):
    return {
        "method": "uniform_fallback" if fallback else "logistic_regression",
        "combinations": [[7, 0, 7] if ordered else [7, 0, 3], [1, 2, 3]],
        "scores": [] if fallback else [
            {"number": number, **({"position": position} if ordered else {}),
             "score": (number + position) / 100}
            for position in (range(1, 4) if ordered else [1])
            for number in range(10)
        ],
        "evaluation": {
            "status": "insufficient_history" if fallback else "evaluated",
            "test_draws": 0 if fallback else 14,
            "model_brier": None if fallback else 0.19,
            "baseline_brier": None if fallback else 0.21,
            "brier_skill": None if fallback else 0.095,
            "improvement_ci95": [] if fallback else [-0.01, 0.05],
            "evidence_of_improvement": False,
            "caution": "DRAW_LEVEL_INTERVAL_CAUTION",
        },
        "feature_weights": {} if fallback else {
            "frequency_last10": 0.31, "frequency_last30": -0.22, "frequency_all": 0.07,
            "gap_normalized": 0.15, "trend": -0.08, "recency_weight": 0.12,
        },
        "reason": "Need at least 100 usable draws; found 40." if fallback else "Untuned logistic scoring.",
        "training_draws": 0 if fallback else 70,
    }


@pytest.mark.parametrize("evidence", [False, True])
def test_ai_supplied_ml_evidence_reaches_prompt_and_cannot_replace_ticket(monkeypatch, evidence):
    from copy import deepcopy
    import json

    client, sdk = _client(monkeypatch,
        "SLOT: Evening\nCOMBINATION: 1, 2, 3\nREASONING: Historical scoring only.\n"
        "DISCLAIMER: Guaranteed winner!")
    generation = _generation()
    generation["evaluation"]["evidence_of_improvement"] = evidence
    if evidence:
        generation["evaluation"]["improvement_ci95"] = [0.01, 0.03]
    original = deepcopy(generation)
    # Supplied generation must not even instantiate a local sampler.
    monkeypatch.setattr("numpy.random.default_rng", Mock(side_effect=AssertionError("resampled")))
    result = client.interpret_lotto_stats(
        "Example", [dict(_slot(), generation=generation)], "ok", "gaps", 3,
        latest_draw={"date": "LEGACY_SENTINEL"},
        trend_summary={"heating_up": [11, 22, 33], "cooling_down": [44, 55]},
        ml_scores={999: 0.99}, shap_summary={999: "LEGACY_SENTINEL"},
        ml_top_combinations=[[999]], feature_importance={"LEGACY_SENTINEL": 9.0},
        pagerank_tiers={"LEGACY_SENTINEL": [999]},
        prior_run_feedback=[{"secret": "LEGACY_SENTINEL"}],
    )["Evening"]
    assert result["combination"] == [7, 0, 3]  # Preserve even unordered display order.
    assert result["disclaimer"] == DISCLAIMER
    assert result["generation"] == generation == original
    prompt = client.last_prompt
    assert "Generator method: logistic_regression" in prompt
    assert 'Selected number model scores: [{"number":7,"score":0.08},' in prompt
    assert '{"number":0,"score":0.01},{"number":3,"score":0.04}]' in prompt
    assert '{"number":9,"score":0.1}' not in prompt  # Selected scores only.
    assert json.dumps(generation["evaluation"], separators=(",", ":")) in prompt
    assert json.dumps(generation["feature_weights"], separators=(",", ":")) in prompt
    assert "Training draws: 70" in prompt
    assert "standardized associations, not causal effects" in prompt
    assert "not calibrated odds" in prompt
    assert "Brier is not winning odds" in prompt
    assert "does not prove an edge" in prompt
    assert "explain why the model weighted the chosen numbers" in prompt
    assert "RNG still chooses a score-weighted sample, not guaranteed highest-ranked numbers" in prompt
    # trend_summary is used (not ignored); other legacy params are still ignored
    assert "trending hotter" in prompt
    assert "[11, 22, 33]" in prompt
    assert "trending cooler" in prompt
    assert "[44, 55]" in prompt
    assert "LEGACY_SENTINEL" not in prompt
    assert "Any supplied combination was generated locally by uniform" not in prompt
    call = sdk.chat.completions.create.call_args.kwargs
    assert call == {"model": "test", "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 1024}


@pytest.mark.parametrize("ticket", [
    [], [1, 2], [1, 2, 3, 4], [True, 2, 3], [1.0, 2, 3], [1.5, 2, 3],
    ["1", 2, 3], [None, 2, 3], [-1, 2, 3], [10, 2, 3], [1, 1, 2],
    [float("nan"), 2, 3], [float("inf"), 2, 3], "123", None,
])
@pytest.mark.parametrize("fallback", [False, True])
def test_ai_rejects_malformed_supplied_first_ticket(monkeypatch, ticket, fallback):
    client, sdk = _client(monkeypatch, "SLOT: Evening\nREASONING: Should not be called.")
    generation = _generation(fallback=fallback)
    generation["combinations"][0] = ticket
    monkeypatch.setattr("numpy.random.default_rng", Mock(side_effect=AssertionError("resampled")))
    with pytest.raises(ValueError, match="combination"):
        client.interpret_lotto_stats("Example", [dict(_slot(), generation=generation)], "ok", "ok", 3)
    sdk.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("generation", [None, {}, [], {"method": "unknown"},
    {"method": "logistic_regression", "combinations": []},
    {"method": "uniform_fallback", "combinations": None}])
def test_ai_invalid_generation_is_not_silently_resampled(monkeypatch, generation):
    client, sdk = _client(monkeypatch, "SLOT: Evening\nREASONING: Should not be called.")
    with pytest.raises(ValueError):
        client.interpret_lotto_stats("Example", [dict(_slot(), generation=generation)], "ok", "ok", 3)
    sdk.chat.completions.create.assert_not_called()


def test_ai_supplied_ordered_repeats_use_position_specific_scores(monkeypatch):
    client, _ = _client(monkeypatch,
        "SLOT: Evening\nCOMBINATION: 1, 2, 3\nREASONING: Position-specific historical scores.")
    generation = _generation(ordered=True)
    result = client.interpret_lotto_stats(
        "Digits", [dict(_slot(ordered=True), generation=generation)], "ok", "ok", 3,
    )["Evening"]
    assert result["combination"] == [7, 0, 7]
    assert result["generation"] == generation
    assert ('Selected number model scores: [{"number":7,"position":1,"score":0.08},'
            '{"number":0,"position":2,"score":0.02},'
            '{"number":7,"position":3,"score":0.1}]') in client.last_prompt


@pytest.mark.parametrize("score", [None, True, "0.1", -0.1, 1.1, float("nan"), float("inf")])
def test_ai_supplied_scores_must_be_finite_numeric_probabilities(monkeypatch, score):
    client, sdk = _client(monkeypatch, "SLOT: Evening\nREASONING: Should not be called.")
    generation = _generation()
    generation["scores"][0]["score"] = score
    with pytest.raises(ValueError, match="Scores"):
        client.interpret_lotto_stats("Example", [dict(_slot(), generation=generation)], "ok", "ok", 3)
    sdk.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("scores,ordered", [
    ([], False), ([{"number": 0, "score": 0.1}] * 2, False),
    ([{"number": 10, "score": 0.1}], False), ([{"number": True, "score": 0.1}], False),
    ([{"number": 0, "position": 1, "score": 0.1}], False),
    ([{"number": 7, "score": 0.1}], True),
    ([{"number": 7, "position": True, "score": 0.1}], True),
    ([{"number": 7, "position": 4, "score": 0.1}], True),
    ([{"number": 7, "position": 1.0, "score": 0.1}], True),
    ([{"number": 7, "position": 1, "score": 0.1}], True),
    ([None], False), ({}, False),
])
def test_ai_rejects_missing_ambiguous_or_invalid_score_keys(monkeypatch, scores, ordered):
    client, sdk = _client(monkeypatch, "SLOT: Evening\nREASONING: Should not be called.")
    generation = _generation(ordered=ordered)
    generation["scores"] = scores
    with pytest.raises(ValueError):
        client.interpret_lotto_stats(
            "Example", [dict(_slot(ordered=ordered), generation=generation)], "ok", "ok", 3,
        )
    sdk.chat.completions.create.assert_not_called()


def test_ai_uniform_fallback_prompt_is_truthful_and_metadata_is_per_slot(monkeypatch):
    import json

    client, _ = _client(monkeypatch,
        "SLOT: Morning\nCOMBINATION: 1, 2, 3\nREASONING: Uniform fallback.\n"
        "SLOT: Evening\nREASONING: Local uniform example.")
    generation = _generation(fallback=True)
    result = client.interpret_lotto_stats(
        "Example", [dict(_slot("Morning"), generation=generation), _slot()], "ok", "ok", 3,
    )
    assert result["Morning"]["combination"] == [7, 0, 3]
    assert result["Morning"]["generation"] == generation
    assert "generation" not in result["Evening"]
    assert "Generator method: uniform_fallback" in client.last_prompt
    assert "Fallback reason: Need at least 100 usable draws; found 40." in client.last_prompt
    assert "Uniform random sampling, no learned weights" in client.last_prompt
    assert "do not fabricate ML explanations" in client.last_prompt
    assert "Generator method: uniform_random" in client.last_prompt
    assert "Selected number model scores:" not in client.last_prompt
    assert "Feature coefficients (" not in client.last_prompt
    assert json.dumps(generation["evaluation"], separators=(",", ":")) in client.last_prompt