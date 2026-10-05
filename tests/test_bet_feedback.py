"""Bet-feedback loop: scored bets -> number multipliers -> ML scores and AI prompt."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api import db
from src.lotto import lean_ml as ml
from src.lotto.azure_ai import AzureAIClient
from src.lotto.games import GAMES
from tests.test_lean_ml import _draws


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", str(tmp_path / "bets.duckdb"))
    return db


def _scored_bet(combo, actual, game="655", bet_date="2026-01-01"):
    played = [int(p) for p in combo.split("-")]
    conn = db.get_connection()
    try:
        conn.execute(
            """INSERT INTO saved_combinations
               (id, game, combination, bet_date, actual_draw, match_count)
               VALUES (nextval('saved_combinations_seq'), ?, ?, ?, ?, ?)""",
            [game, combo, bet_date, json.dumps(actual), sum(p in actual for p in played)],
        )
    finally:
        conn.close()


def test_feedback_needs_enough_scored_history(temp_db):
    for _ in range(4):
        _scored_bet("01-02-03-04-05-06", [1, 2, 40, 41, 42, 43])
    assert db.bet_number_feedback("655") == {}


def test_feedback_without_any_hit_is_empty(temp_db):
    for _ in range(10):
        _scored_bet("01-02-03-04-05-06", [40, 41, 42, 43, 44, 45])
    assert db.bet_number_feedback("655") == {}


def test_feedback_multipliers_are_bounded_and_directional(temp_db):
    for _ in range(10):
        _scored_bet("01-02-03-04-05-06", [1, 2, 40, 41, 42, 43])
    feedback = db.bet_number_feedback("655")
    assert feedback[1] == feedback[2] == pytest.approx(1.25)
    assert feedback[3] == feedback[6] == pytest.approx(0.75)
    assert db.bet_number_feedback("642") == {}


def test_bet_history_reports_matches_and_miss_distance(temp_db):
    _scored_bet("01-02-03", [1, 5, 9], game="3d")
    record = db.load_bet_history("3d")[0]
    assert record["ai_combination"] == [1, 2, 3]
    assert record["actual_draw"] == [1, 5, 9]
    assert record["match_count"] == 1
    assert record["total_distance"] == 3  # |2-1| + |3-1|


def test_unscored_bets_are_ignored(temp_db):
    conn = db.get_connection()
    try:
        conn.execute(
            """INSERT INTO saved_combinations (id, game, combination)
               VALUES (nextval('saved_combinations_seq'), '655', '01-02-03-04-05-06')"""
        )
    finally:
        conn.close()
    assert db.load_bet_history("655") == []


def test_latest_run_restores_stored_prompt(temp_db):
    assert db.latest_ml_run("655") is None
    first = db.save_ml_run("655", None, "2026-01-01", 70, "v1", ai_prompt_text="old prompt")
    second = db.save_ml_run("655", None, "2026-01-02", 70, "v1", ai_prompt_text="new prompt")
    latest = db.latest_ml_run("655")
    assert second > first
    assert latest["prompt"] == "new prompt" and latest["id"] == second
    assert db.latest_ml_run("642") is None


def test_ml_feedback_rescales_scores_and_flags_numbers():
    game = GAMES["655"]
    draws = _draws(game, 100)
    base = ml.generate_ml_combinations(draws, game, n=5)
    adjusted = ml.generate_ml_combinations(draws, game, n=5, bet_feedback={7: 1.25, 8: 0.75})
    assert "bet_feedback_numbers" not in base
    assert adjusted["bet_feedback_numbers"] == 2
    before = {row["number"]: row["score"] for row in base["scores"]}
    after = {row["number"]: row["score"] for row in adjusted["scores"]}
    assert after[7] == pytest.approx(min(1.0, before[7] * 1.25))
    assert after[8] == pytest.approx(before[8] * 0.75)
    assert after[9] == pytest.approx(before[9])
    assert adjusted["evaluation"] == base["evaluation"]


def _client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test")
    client = AzureAIClient()
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="SLOT: Evening\nREASONING: ok\nDISCLAIMER: none"),
    )])
    monkeypatch.setattr(client, "_client", lambda: sdk)
    return client


def _slot():
    rows = [{"number": i, "total_draws": i + 1} for i in range(10)]
    return {"label": "Evening", "heat_map_rows": rows, "total_draws": 100}


def test_past_bet_results_reach_prompt_and_junk_is_skipped(monkeypatch):
    client = _client(monkeypatch)
    history = [
        {"run_date": "2026-01-05", "ai_combination": [1, 2, 3], "actual_draw": [1, 5, 9],
         "match_count": 1, "total_distance": 3},
        {"secret": "JUNK_SENTINEL"},
        {"ai_combination": [True, 2, 3], "actual_draw": [1, 2, 3], "match_count": 3},
    ]
    client.interpret_lotto_stats("Example", [_slot()], "uniform", "gaps", 3,
                                 prior_run_feedback=history)
    prompt = client.last_prompt
    assert "Scored past bets (most recent first):" in prompt
    assert "2026-01-05: played [1, 2, 3], drawn [1, 5, 9], matches 1" in prompt
    assert "JUNK_SENTINEL" not in prompt
    assert prompt.count("played [") == 1
    assert "review of how far those earlier tickets landed" in prompt


def test_no_history_means_no_history_block(monkeypatch):
    client = _client(monkeypatch)
    client.interpret_lotto_stats("Example", [_slot()], "uniform", "gaps", 3,
                                 prior_run_feedback=[])
    assert "Scored past bets" not in client.last_prompt


def test_rescaling_is_disclosed_to_the_model(monkeypatch):
    client = _client(monkeypatch)
    generation = {
        "method": "logistic_regression", "combinations": [[1, 2, 3]],
        "scores": [{"number": n, "score": 0.1} for n in range(10)],
        "feature_weights": {"frequency_all": 0.1}, "training_draws": 70,
        "evaluation": {"status": "evaluated"}, "reason": "r", "bet_feedback_numbers": 4,
    }
    slot = dict(_slot(), generation=generation)
    client.interpret_lotto_stats("Example", [slot], "uniform", "gaps", 3)
    assert "rescaled (x0.75 to x1.25) for 4 numbers" in client.last_prompt


def _arena_rows():
    return [
        {"strategy": "Crowd-avoiding random", "mean_matches": 0.74, "edge_pct": 13.1, "z": 2.13,
         "p_value": 0.033, "p_adjusted": 0.2, "jackpot_share_kept": 0.995, "share_gain_pct": 2.3},
        {"strategy": "Uniform random", "mean_matches": 0.673, "edge_pct": 2.9, "z": 0.76,
         "p_value": 0.45, "p_adjusted": 1.0, "jackpot_share_kept": 0.973, "share_gain_pct": 0.0},
    ]


def test_arena_analysis_prompt_uses_only_supplied_results(monkeypatch):
    client = _client(monkeypatch)
    client._client = lambda: Mock(chat=Mock(completions=Mock(create=Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="VERDICT: ok"))])))))
    text = client.interpret_arena(
        "6/55 Grand Lotto", _arena_rows(), "No strategy beat random.", 300, 5, 0.6545,
        crowd={"cv_auc": 0.698, "game_only_auc": 0.619, "winner_draws": 243, "draws": 2498,
               "quintile_winner_rate": [0.026, 0.218]},
        bet_history=[{"run_date": "2026-01-05", "ai_combination": [1, 2], "actual_draw": [1, 9],
                      "match_count": 1}, {"secret": "JUNK_SENTINEL"}],
        next_tickets=[[3, 9, 20, 31, 40, 50]],
    )
    prompt = client.last_prompt
    assert text == "VERDICT: ok"
    assert "seasoned gambler" in prompt
    assert "Crowd-avoiding random: mean matches 0.7400 (+13.10% vs random), z +2.13" in prompt
    assert "adjusted p 0.200" in prompt and "jackpot share kept 0.9950 (+2.30% vs uniform)" in prompt
    assert "cross-validated AUC 0.698 versus 0.619" in prompt and "2.6% / 21.8%" in prompt
    assert "played [1, 2], drawn [1, 9]" in prompt and "JUNK_SENTINEL" not in prompt
    assert "[[3, 9, 20, 31, 40, 50]]" in prompt
    assert "adjusted p-value is below 0.05" in prompt
    assert "do not create, change, rank or recommend tickets" in prompt


def test_arena_analysis_rejects_bad_responses(monkeypatch):
    client = _client(monkeypatch)
    for finish, content, message in [("length", "partial", "truncated"), ("stop", "  ", "empty")]:
        client._client = lambda f=finish, c=content: Mock(chat=Mock(completions=Mock(create=Mock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=f, message=SimpleNamespace(content=c))])))))
        with pytest.raises(RuntimeError, match=message):
            client.interpret_arena("G", _arena_rows(), "c", 10, 1, 0.5)
    with pytest.raises(ValueError):
        client.interpret_arena("G", [], "c", 10, 1, 0.5)


MENU = [
    {"ticket": [32, 37, 38, 42, 54, 58], "crowd_index": 0.16, "share_kept": 0.998, "ml_support": 1.1},
    {"ticket": [32, 37, 38, 42, 50, 56], "crowd_index": 0.17, "share_kept": 0.998, "ml_support": 1.0},
    {"ticket": [2, 34, 46, 48, 55, 57], "crowd_index": 0.18, "share_kept": 0.997, "ml_support": 0.9},
    {"ticket": [35, 36, 40, 44, 51, 53], "crowd_index": 0.19, "share_kept": 0.997, "ml_support": 1.2},
]
CONTEXT = {32: {"ml_vs_average": 1.1, "freq_last100": 5, "draws_since_seen": 4}}


def _selector(monkeypatch, content, finish="stop"):
    client = _client(monkeypatch)
    client._client = lambda: Mock(chat=Mock(completions=Mock(create=Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])))))
    return client


def test_ai_picks_valid_shortlisted_tickets(monkeypatch):
    reply = ('```json\n{"tickets": [[35, 36, 40, 44, 51, 53], [2, 34, 46, 48, 55, 57]], '
             '"reasoning": "Low crowding with spread."}\n```')
    client = _selector(monkeypatch, reply)
    result = client.select_tickets("6/55", MENU, CONTEXT, n=2, max_overlap=3)
    assert result["tickets"] == [[35, 36, 40, 44, 51, 53], [2, 34, 46, 48, 55, 57]]
    assert result["ai_picked"] == 2 and result["filled"] == 0
    assert result["reasoning"] == "Low crowding with spread."
    prompt = client.last_prompt
    assert "seasoned gambler" in prompt and "Choose exactly 2 tickets" in prompt
    assert "[32, 37, 38, 42, 54, 58] crowd index 0.160" in prompt
    assert "32: ML 1.10x average, drawn 5 times in last 100 draws" in prompt
    assert "Copy each chosen ticket exactly" in prompt


def test_invented_duplicate_and_overlapping_picks_are_dropped_and_filled(monkeypatch):
    reply = ('{"tickets": [[1, 2, 3, 4, 5, 6], [32, 37, 38, 42, 54, 58], [32, 37, 38, 42, 54, 58], '
             '[32, 37, 38, 42, 50, 56]], "reasoning": "x"}')
    client = _selector(monkeypatch, reply)
    result = client.select_tickets("6/55", MENU, CONTEXT, n=3, max_overlap=3)
    assert result["tickets"][0] == [32, 37, 38, 42, 54, 58]
    assert [32, 37, 38, 42, 50, 56] not in result["tickets"]  # shares four numbers with the first
    assert [1, 2, 3, 4, 5, 6] not in result["tickets"]
    assert result["ai_picked"] == 1 and result["filled"] == 2 and len(result["tickets"]) == 3
    assert all(t in [m["ticket"] for m in MENU] for t in result["tickets"])


def test_unparseable_reply_without_reasoning_is_an_error_but_prose_is_filled(monkeypatch):
    with pytest.raises(RuntimeError, match="no usable selection"):
        _selector(monkeypatch, "I like 7").select_tickets("G", MENU, CONTEXT, n=2, max_overlap=3)
    with pytest.raises(RuntimeError, match="truncated"):
        _selector(monkeypatch, "{}", "length").select_tickets("G", MENU, CONTEXT, n=2, max_overlap=3)
    result = _selector(monkeypatch, '{"tickets": "none", "reasoning": "Nothing stood out."}') \
        .select_tickets("G", MENU, CONTEXT, n=2, max_overlap=3)
    assert result["ai_picked"] == 0 and result["filled"] == 2


def test_selection_validates_its_inputs(monkeypatch):
    client = _selector(monkeypatch, "{}")
    with pytest.raises(ValueError, match="fewer candidates"):
        client.select_tickets("G", MENU[:1], CONTEXT, n=3, max_overlap=3)
    with pytest.raises(ValueError, match="integer ticket"):
        client.select_tickets("G", [{"ticket": ["a"]}], CONTEXT, n=1, max_overlap=3)
    with pytest.raises(ValueError, match="positive"):
        client.select_tickets("G", MENU, CONTEXT, n=0, max_overlap=3)
