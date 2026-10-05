"""Offline AppTest of the actual Strategy Arena page branch from dashboard/app.py."""
import ast
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import duckdb
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from api import db
from src.lotto.crowd import crowd_games
from tests.conftest import _fair_draws

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
HARNESS = "_arena_page_test_fixtures"


def _page_source():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    test = ast.dump(ast.parse('page == "Strategy Arena"', mode="eval").body)
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.dump(n.test) == test]
    assert len(branches) == 1, "Strategy Arena page must exist exactly once"
    helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
               and n.lineno < helpers["_prob_generation"].lineno]
    setup = ast.parse(f"from {HARNESS} import load_draws, game_selector\n").body
    module = ast.Module(body=[*imports, *setup, *branches[0].body], type_ignores=[])
    source = ast.unparse(ast.fix_missing_locations(module))
    return source.encode("ascii", "backslashreplace").decode("ascii")


def _frames():
    rng = np.random.default_rng(5)
    frames = {}
    for cfg in crowd_games():
        frame = _fair_draws(cfg, n=320, seed=cfg.pool)
        calendar = (frame[[f"n{i}" for i in range(1, 7)]].to_numpy() <= 31).sum(1)
        frame["winners_jackpot"] = (rng.random(len(frame)) < np.where(calendar >= 5, 0.3, 0.05)).astype(int)
        frames[cfg.id] = frame
    return frames


@pytest.fixture
def arena_app(monkeypatch):
    blocked = Mock(side_effect=AssertionError("Live DB access forbidden"))
    monkeypatch.setattr(duckdb, "connect", blocked)
    conn = Mock()
    monkeypatch.setattr(db, "get_connection", lambda *a, **k: conn)
    frames = _frames()
    module = ModuleType(HARNESS)
    module.load_draws = lambda game_id: frames[game_id].copy()
    module.game_selector = lambda label="Game": st.selectbox(label, ["655"])
    monkeypatch.setitem(sys.modules, HARNESS, module)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_string(_page_source(), default_timeout=120)
    yield app, conn
    st.cache_resource.clear()
    blocked.assert_not_called()


def _button(app, prefix):
    return next(b for b in app.button if b.label.startswith(prefix))


def test_arena_page_runs_strategies_and_shows_crowd_model(arena_app):
    app, _ = arena_app
    app.run()
    assert not app.exception and not app.error
    assert app.title[0].value.endswith("Strategy Arena")
    assert any(m.label == "Cross-validated AUC" for m in app.metric)
    assert any("Winner rate by predicted-crowding quintile" in c.value for c in app.caption)
    app.multiselect[0].set_value(["uniform", "hot", "crowd"])
    app.slider[0].set_value(100)
    _button(app, "▶ Run Arena").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    table = next(d.value for d in app.dataframe if "p_adjusted" in d.value.columns)
    assert set(table["strategy"]) == {"Uniform random", "Hot numbers (last 100 draws)",
                                      "Crowd-avoiding random"}
    assert (table["draws"] == 100).all()
    assert "share_gain_pct" in table.columns
    assert any("strategy" in i.value.lower() or "random" in i.value.lower() for i in app.info)


def test_arena_page_next_tickets_show_crowding_and_save_bets(arena_app):
    app, conn = arena_app
    app.run()
    app.selectbox(key="arena_next_strategy").set_value("crowd")
    app.slider(key="arena_next_count").set_value(3)
    app.button(key="arena_next_go").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    shown = next(d.value for d in app.dataframe if "ticket" in d.value.columns)
    assert len(shown) == 3
    assert (shown["crowd index (1.0 = average)"] < 1).all()
    assert (shown["jackpot share kept if won"] > 0).all()
    app.button(key="arena_save").click().run()
    assert not app.exception and not app.error
    inserts = [c for c in conn.execute.call_args_list if "INSERT INTO saved_combinations" in c.args[0]]
    assert len(inserts) == 3
    params = inserts[0].args[1]
    assert params[0] == "655" and params[1] is None
    assert params[3] == "Strategy Arena: Crowd-avoiding random"
    conn.close.assert_called_once()


def test_arena_page_ai_analysis_button(arena_app, monkeypatch):
    from types import SimpleNamespace
    from src.lotto.azure_ai import AzureAIClient

    app, _ = arena_app
    for name, value in {"AZURE_OPENAI_ENDPOINT": "https://offline.invalid",
                        "AZURE_OPENAI_API_KEY": "offline-test-only",
                        "AZURE_OPENAI_DEPLOYMENT": "offline-test"}.items():
        monkeypatch.setenv(name, value)
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="VERDICT: offline analysis text"))])
    monkeypatch.setattr(AzureAIClient, "_client", Mock(return_value=sdk))
    history = Mock(return_value=[{"run_date": "2026-01-05", "ai_combination": [1, 2, 3, 4, 5, 6],
                                  "actual_draw": [1, 2, 9, 10, 11, 12], "match_count": 2}])
    monkeypatch.setattr(db, "load_bet_history", history)
    app.run()
    assert not any(b.key == "arena_ai" for b in app.button)  # needs a finished run first
    app.multiselect[0].set_value(["uniform", "crowd"])
    app.slider[0].set_value(100)
    _button(app, "▶ Run Arena").click().run()
    sdk.chat.completions.create.assert_not_called()
    app.button(key="arena_ai").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    sdk.chat.completions.create.assert_called_once()
    prompt = sdk.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Crowd-avoiding random: mean matches" in prompt and "Uniform random: mean matches" in prompt
    assert "played [1, 2, 3, 4, 5, 6], drawn [1, 2, 9, 10, 11, 12]" in prompt
    assert "cross-validated AUC" in prompt
    assert any(i.value == "VERDICT: offline analysis text" for i in app.info)
    assert any(c.value == prompt for c in app.code)
    history.assert_called_with("655")
    app.run()  # the stored analysis survives a rerun without a second request
    sdk.chat.completions.create.assert_called_once()
    assert any(i.value == "VERDICT: offline analysis text" for i in app.info)


def test_arena_page_ai_picks_tickets_from_the_shortlist(arena_app, monkeypatch):
    import json
    import re
    from types import SimpleNamespace
    from src.lotto.azure_ai import AzureAIClient

    app, conn = arena_app
    for name, value in {"AZURE_OPENAI_ENDPOINT": "https://offline.invalid",
                        "AZURE_OPENAI_API_KEY": "offline-test-only",
                        "AZURE_OPENAI_DEPLOYMENT": "offline-test"}.items():
        monkeypatch.setenv(name, value)

    def reply(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        menu = [json.loads(m) for m in re.findall(r"^\d+\. (\[[\d, ]+\]) crowd index", prompt, re.M)]
        picks = [menu[1], menu[0]]
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps({"tickets": picks + [[1, 2, 3, 4, 5, 6]], "reasoning": "Offline reasoning."})))])

    sdk = Mock()
    sdk.chat.completions.create.side_effect = reply
    monkeypatch.setattr(AzureAIClient, "_client", Mock(return_value=sdk))
    monkeypatch.setattr(db, "load_bet_history", Mock(return_value=[]))
    app.run()
    app.slider(key="arena_next_count").set_value(2)
    app.button(key="arena_ai_pick").click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    sdk.chat.completions.create.assert_called_once()
    prompt = sdk.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Choose exactly 2 tickets" in prompt and "Number context:" in prompt
    assert any("AI chose 2 of 2 tickets" in s.value for s in app.success)
    assert any(i.value == "Offline reasoning." for i in app.info)
    shown = next(d.value for d in app.dataframe if "ticket" in d.value.columns)
    assert len(shown) == 2 and (shown["crowd index (1.0 = average)"] < 1).all()
    assert any(c.value == prompt for c in app.code)
    app.button(key="arena_save").click().run()
    inserts = [c for c in conn.execute.call_args_list if "INSERT INTO saved_combinations" in c.args[0]]
    assert len(inserts) == 2
    assert inserts[0].args[1][3] == "Strategy Arena: AI-curated (ML + crowd shortlist)"


def test_arena_page_each_click_draws_fresh_tickets_with_a_visible_seed(arena_app):
    app, _ = arena_app
    app.run()
    app.selectbox(key="arena_next_strategy").set_value("hot")
    app.button(key="arena_next_go").click().run()
    assert not app.exception and not app.error
    first = app.session_state["arena_next"]
    app.button(key="arena_next_go").click().run()
    second = app.session_state["arena_next"]
    assert first["seed"] != second["seed"] and first["tickets"] != second["tickets"]
    assert any(f"Sampling seed {second['seed']}" in c.value for c in app.caption)
