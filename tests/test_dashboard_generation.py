"""Offline AppTest coverage of the actual Probability & AI page.

Extract the entire page branch and shared generation helper from app.py's AST;
never import/execute full-app startup (which can auto-scrape or write DuckDB).
This checks real Streamlit rendering, ML, prompt construction and response parsing,
not navigation, startup, the load-data UI, persistence, or a live Azure service.
"""
import ast
from datetime import date
import json
from pathlib import Path
import re
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import duckdb
import numpy as np
import openai
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from api import db
from src.lotto import lean_ml, scrape
from src.lotto.azure_ai import AzureAIClient
from src.lotto.games import GAMES
from src.lotto.predict import DISCLAIMER


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
HARNESS_MODULE = "_dashboard_generation_test_fixtures"
TIME_LABELS = {"14:00": "2:00 PM", "17:00": "5:00 PM", "21:00": "9:00 PM"}


def _page_source():
    """Fail closed if the page/helper structure changes; omit all startup code."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    helpers = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.dump(node.test) == ast.dump(ast.parse('page == "Probability & AI"', mode="eval").body)
    ]
    assert len(branches) == 1, "Probability & AI page must exist exactly once"
    # Retain actual preceding imports (st, pd, px, go, GAMES, DB alias, etc.).
    imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node.lineno < helpers["_prob_generation"].lineno
    ]
    setup = ast.parse(
        f"from {HARNESS_MODULE} import load_draws, game_selector\n"
        "assert st.session_state['_auto_updated'] is True\n"
        "_last_dates = _get_last_dates()\n"
        "st.session_state['_test_last_dates'] = _last_dates\n"
    ).body
    observe = ast.parse(
        "st.session_state['_test_generation'] = generation_by_label\n"
        "st.session_state['_test_slots'] = _build_slots_data()\n"
    ).body
    module = ast.Module(body=[
        *imports, helpers["_prob_generation"], helpers["_get_last_dates"],
        *setup, *branches[0].body, *observe,
    ], type_ignores=[])
    source = ast.unparse(ast.fix_missing_locations(module))
    # Streamlit 1.40 writes from_string scripts with Windows' locale encoding.
    # Escape Unicode literals without changing the executed AST or UI text.
    portable = source.encode("ascii", "backslashreplace").decode("ascii")
    assert ast.dump(ast.parse(portable)) == ast.dump(ast.parse(source))
    return portable


def _synthetic_draws(game):
    rng = np.random.default_rng(27)
    frames = []
    for slot in game.draw_times:
        values = np.array([
            rng.choice(np.arange(game.low_boundary, game.pool + 1), game.picks,
                       replace=game.ordered)
            for _ in range(120)
        ])
        if game.ordered:
            # Published order deliberately differs from sorted storage order.
            # Position 1 and 3 repeat 7, while position 2 is always zero.
            values[:] = [7, 0, 7]
        frame = pd.DataFrame({
            "draw_date": pd.date_range(end=date.today(), periods=120).strftime("%Y-%m-%d"),
            "draw_time": slot, "game": game.id,
        })
        for pos in range(1, game.picks + 1):
            frame[f"pub_{pos}"] = pd.array(values[:, pos - 1], dtype="Int64")
            frame[f"n{pos}"] = pd.array(np.sort(values, axis=1)[:, pos - 1], dtype="Int64")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=8)


@pytest.fixture
def dashboard(monkeypatch):
    """Block external access even if a future edit accidentally reaches startup."""
    blocked = Mock(side_effect=AssertionError("Live DB/network/scraper/SDK access forbidden"))
    monkeypatch.setattr(duckdb, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(openai, "AzureOpenAI", blocked)
    for name in ("scrape_pcso_gov", "scrape_pick6", "fetch_page", "_fetch_pcso_gov_chunk"):
        monkeypatch.setattr(scrape, name, blocked)
    monkeypatch.setattr(db, "compare_with_actual", blocked)
    for name, value in {
        "AZURE_OPENAI_ENDPOINT": "https://offline.invalid",
        "AZURE_OPENAI_API_KEY": "offline-test-only",
        "AZURE_OPENAI_DEPLOYMENT": "offline-test",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
    }.items():
        monkeypatch.setenv(name, value)

    sdk = Mock()
    client_factory = Mock(return_value=sdk)
    monkeypatch.setattr(AzureAIClient, "_client", client_factory)
    connections = []
    st.cache_data.clear()

    def start(game_id):
        game = GAMES[game_id]
        draws = _synthetic_draws(game)
        labels = [TIME_LABELS[slot] for slot in game.draw_times]
        sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="\n".join(
                f"SLOT: {label}\nREASONING: Offline explanation for {label}; no winning edge."
                for label in labels
            )),
        )])

        def get_connection(*args, **kwargs):
            conn = Mock()

            def execute(sql, *params):
                assert " ".join(sql.split()) == (
                    "SELECT game, MAX(draw_date) AS last_date FROM draws GROUP BY game"
                ), "Only _get_last_dates may query the fake DB"
                return conn

            conn.execute.side_effect = execute
            conn.fetchdf.return_value = pd.DataFrame({
                "game": [game_id], "last_date": [date.today()],
            })
            connections.append(conn)
            return conn

        monkeypatch.setattr(db, "get_connection", get_connection)
        module = ModuleType(HARNESS_MODULE)

        def load_draws(requested_game):
            assert requested_game == game_id
            return draws.copy(deep=True)

        loader = Mock(side_effect=load_draws)
        module.load_draws = loader
        module.game_selector = lambda label="Game": st.selectbox(label, [game_id])
        monkeypatch.setitem(sys.modules, HARNESS_MODULE, module)
        app = AppTest.from_string(_page_source(), default_timeout=60)
        # Required before the FIRST run, not after startup has already executed.
        app.session_state["_auto_updated"] = True
        return SimpleNamespace(app=app, game=game, draws=draws, loader=loader,
                               sdk=sdk, client_factory=client_factory, labels=labels)

    yield start
    st.cache_data.clear()
    blocked.assert_not_called()  # Detect even swallowed errors in dashboard code.
    for conn in connections:
        conn.execute.assert_called_once()
        conn.close.assert_called_once()


def _assert_clean(app):
    assert not app.exception, [element.message for element in app.exception]
    assert not app.error, [element.value for element in app.error]


def _first_displayed(app, game):
    first = [element.value for element in app.markdown if element.value.startswith("**#1**")]
    assert len(first) == len(game.draw_times)
    return [list(map(int, re.findall(r"`(\d+)`", value))) for value in first]


def _assert_ai_click(run):
    app, game = run.app, run.game
    displayed = _first_displayed(app, game)
    slots = app.session_state["_test_slots"]
    run.client_factory.assert_not_called()  # AI must remain opt-in.
    app.button(key="regen_ai").click().run()
    _assert_clean(app)
    run.client_factory.assert_called_once()
    run.sdk.chat.completions.create.assert_called_once()
    call = run.sdk.chat.completions.create.call_args.kwargs
    assert call["model"] == "offline-test"
    assert call["max_completion_tokens"] > 0
    assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"
    assert "temperature" not in call
    prompt = call["messages"][0]["content"]
    assert any(element.value == prompt for element in app.code)
    assert _first_displayed(app, game) == displayed
    results = [element.value for element in app.success if "Combination — AI Explanation" in element.value]
    assert len(results) == len(run.labels)
    for label, ticket, slot, rendered in zip(run.labels, displayed, slots, results):
        assert slot["label"] == label
        assert slot["generation"]["combinations"][0] == ticket
        assert f"SLOT: {label}\nDraw count: {slot['total_draws']}\n" in prompt
        block = prompt.split(f"SLOT: {label}\n", 1)[1].split("\nSLOT:", 1)[0]
        assert f"Exact preselected combination: {ticket}" in block
        assert json.dumps(slot["generation"]["evaluation"], separators=(",", ":")) in block
        assert " — ".join(f"**{number:02d}**" for number in ticket) in rendered
        assert any(f"Offline explanation for {label}" in element.value for element in app.info)
    assert sum(DISCLAIMER in element.value for element in app.caption) == len(run.labels)
    # An ordinary rerun must reuse the explanation without another SDK request.
    app.run()
    _assert_clean(app)
    run.sdk.chat.completions.create.assert_called_once()
    return prompt


@pytest.mark.parametrize("game_id", ["655", "3d"])
def test_probability_default_ml_renders_scores_evaluation_and_exact_ai_ticket(dashboard, game_id):
    run = dashboard(game_id)
    app = run.app.run()
    _assert_clean(app)
    assert app.title[0].value == "Probability & AI Analysis"
    assert app.selectbox(key="prob_generation_mode").value == "ML-assisted"
    assert app.session_state["_test_last_dates"] == {game_id: date.today().isoformat()}
    assert run.loader.call_count >= 2
    generations = app.session_state["_test_generation"]
    assert list(generations) == run.labels
    score_tables = [element.value for element in app.dataframe if "score" in element.value.columns]
    weight_tables = [element.value for element in app.dataframe if "coefficient" in element.value.columns]
    assert len(score_tables) == len(weight_tables) == len(run.labels)
    for result, scores, weights in zip(generations.values(), score_tables, weight_tables):
        assert result["method"] == "logistic_regression"
        assert result["evaluation"]["status"] == "evaluated"
        assert result["evaluation"]["usable_draws"] == 120
        assert result["training_draws"] == 90
        pd.testing.assert_frame_equal(scores, pd.DataFrame(result["scores"]))
        assert dict(zip(weights["feature"], weights["coefficient"])) == result["feature_weights"]
    for label, key in [("Test model Brier", "model_brier"), ("Uniform Brier", "baseline_brier"),
                       ("Test draws", "test_draws")]:
        expected = [str(r["evaluation"][key]) if key == "test_draws"
                    else f"{r['evaluation'][key]:.6f}" for r in generations.values()]
        assert [m.value for m in app.metric if m.label == label] == expected
    assert sum("Evaluation status: evaluated" in c.value for c in app.caption) == len(run.labels)
    assert sum("95% CI" in c.value for c in app.caption) == len(run.labels)

    prompt = _assert_ai_click(run)
    assert "Selected number model scores:" in prompt
    if run.game.ordered:
        assert _first_displayed(app, run.game) == [[7, 0, 7]] * 3
        for position in range(1, 4):
            assert any(tab.label == f"Position {position}" for tab in app.tabs)
            assert f"Position {position}:" in prompt
        for slot in app.session_state["_test_slots"]:
            rows = slot["heat_map_rows"]
            assert len(rows) == 30
            assert {(r["position"], r["number"]) for r in rows} == {
                (pos, num) for pos in range(1, 4) for num in range(10)
            }
            assert [r["number"] for r in rows if r["total_draws"] == 120] == [7, 0, 7]
        assert '"number":7,"position":1' in prompt
        assert '"number":0,"position":2' in prompt
        assert '"number":7,"position":3' in prompt


@pytest.mark.parametrize("game_id", ["655", "3d"])
def test_probability_uniform_bypasses_generator_and_trainer(dashboard, monkeypatch, game_id):
    run = dashboard(game_id)
    forbidden = Mock(side_effect=AssertionError("Uniform selection must not train"))
    monkeypatch.setattr(lean_ml, "generate_ml_combinations", forbidden)
    monkeypatch.setattr(lean_ml, "_fit_classifier", forbidden)
    # Render the actual selectbox in uniform mode without fitting an initial model.
    run.app.session_state["prob_generation_mode"] = "Uniform random"
    app = run.app.run()
    _assert_clean(app)
    assert app.selectbox(key="prob_generation_mode").value == "Uniform random"
    for result in app.session_state["_test_generation"].values():
        assert result["method"] == "uniform_fallback"
        assert result["scores"] == [] and result["feature_weights"] == {}
        assert result["training_draws"] == 0
        assert result["evaluation"]["status"] == "not_requested"
    for label in ("Test model Brier", "Uniform Brier"):
        assert [m.value for m in app.metric if m.label == label] == ["N/A"] * len(run.labels)
    assert not any("score" in element.value.columns for element in app.dataframe)
    assert sum("no model was trained or evaluated" in w.value for w in app.warning) == len(run.labels)
    prompt = _assert_ai_click(run)
    assert "Generator method: uniform_fallback" in prompt
    assert "Selected number model scores:" not in prompt
    assert "do not fabricate ML explanations" in prompt
    forbidden.assert_not_called()