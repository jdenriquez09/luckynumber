"""Offline regression tests for generation routes (no app startup or live services)."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import analysis as routes
from src.lotto.azure_ai import AzureAIClient
from src.lotto.games import GAMES


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(routes, "get_connection", Mock(side_effect=AssertionError("Live DB forbidden")))
    monkeypatch.setattr(AzureAIClient, "_client", Mock(side_effect=AssertionError("Live Azure forbidden")))


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ai(monkeypatch):
    mock = Mock(spec=AzureAIClient)
    mock.is_configured.return_value = True
    mock.interpret_lotto_stats.return_value = {}
    monkeypatch.setattr("src.lotto.azure_ai.AzureAIClient", Mock(return_value=mock))
    return mock


def draws(game_id, entries):
    """Keep public order distinct from sorted n-columns to catch position loss."""
    rows = []
    for index, (time, numbers) in enumerate(entries):
        row = {"draw_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index), "draw_time": time}
        row.update({f"pub_{i}": value for i, value in enumerate(numbers, 1)})
        sorted_numbers = sorted(numbers, key=lambda value: -1 if value is None else value)
        row.update({f"n{i}": value for i, value in enumerate(sorted_numbers, 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_probability_route_reaches_endpoint(client, monkeypatch):
    frame = pd.DataFrame({"draw_time": ["21:00"]})
    load = Mock(return_value=frame)
    heat = Mock(return_value=pd.DataFrame([{"number": 1, "total_draws": 1}]))
    generation = model_generation([[1, 2, 3, 4, 5, 6]])
    generate = Mock(return_value=generation)
    monkeypatch.setattr(routes, "_load_draws", load)
    monkeypatch.setattr(routes.pred, "number_heat_map", heat)
    monkeypatch.setattr(routes.lean_ml, "generate_ml_combinations", generate)
    monkeypatch.setattr(routes.pred, "heat_map_summary", Mock(return_value="History only"))

    response = client.get("/analysis/655/probability?window=2&n_suggestions=1&draw_time=21:00")

    assert response.status_code == 200
    assert response.json()["suggestions"] == [[1, 2, 3, 4, 5, 6]]
    assert response.json()["window"] == 2
    assert response.json()["generation"] == generation
    assert response.json()["by_draw_time"] == {}
    assert response.json()["game"] == "655"
    assert response.json()["draw_time"] == "21:00"
    assert response.json()["total_draws"] == 1
    assert response.json()["heat_map"] == heat.return_value.to_dict(orient="records")
    assert response.json()["summary"] == "History only"
    assert response.json()["disclaimer"] == routes.pred.DISCLAIMER
    load.assert_called_once_with(GAMES["655"])
    heat.assert_called_once_with(frame, GAMES["655"], window=2, draw_time="21:00")
    generate.assert_called_once_with(frame, GAMES["655"], n=1, window=2, draw_time="21:00", seed=42)


@pytest.mark.parametrize("game_id", ["3d", "4d", "6d", "2d"])
def test_ordered_ai_payload_and_sampling(client, monkeypatch, game_id):
    cfg = GAMES[game_id]
    chosen = [cfg.low_boundary, cfg.pool] + [cfg.low_boundary] * (cfg.picks - 2)
    # 2D has no zero but still permits repeated numbers.
    if game_id == "2d":
        chosen = [1, 1]
    times = cfg.draw_times[:2]
    entries = [(time, [cfg.pool] * cfg.picks) for time in times]
    entries += [(time, chosen) for time in times]
    frame = draws(game_id, entries)
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))
    position_heat = Mock(wraps=routes.pred.position_heat_map)
    monkeypatch.setattr(routes.pred, "position_heat_map", position_heat)
    monkeypatch.setattr(routes.pred, "number_heat_map", Mock(side_effect=AssertionError("Not ordered")))
    uniformity = Mock(return_value=SimpleNamespace(conclusion="Digit uniformity"))
    repeats = Mock(return_value=SimpleNamespace(conclusion="Digit repeats"))
    monkeypatch.setattr(routes.an, "digit_uniformity", uniformity)
    monkeypatch.setattr(routes.an, "digit_repeat_rate", repeats)
    for name in ("freq_chi_square", "gap_analysis"):
        monkeypatch.setattr(routes.an, name, Mock(side_effect=AssertionError("Not pick-6")))

    ai_client = AzureAIClient()
    monkeypatch.setattr(ai_client, "is_configured", Mock(return_value=True))
    labels = [{"14:00": "2:00 PM", "17:00": "5:00 PM", "21:00": "9:00 PM"}[t] for t in times]
    content = "\n".join(f"SLOT: {label}\nREASONING: Uniform random sampling.\nDISCLAIMER: Fair draws." for label in labels)
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content),
    )]))
    monkeypatch.setattr(ai_client, "_client", Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )))
    interpret = Mock(wraps=ai_client.interpret_lotto_stats)
    monkeypatch.setattr(ai_client, "interpret_lotto_stats", interpret)
    monkeypatch.setattr("src.lotto.azure_ai.AzureAIClient", Mock(return_value=ai_client))
    rng = Mock()
    rng.choice.return_value = np.array(chosen)
    monkeypatch.setattr(np.random, "default_rng", Mock(return_value=rng))
    trainer = Mock(side_effect=AssertionError("Uniform must bypass ML generation"))
    monkeypatch.setattr(routes.lean_ml, "generate_ml_combinations", trainer)

    response = client.post(f"/analysis/{game_id}/ai_summary", json={
        "window": 1, "method": "uniform", "n_suggestions": 1,
    })

    assert response.status_code == 200, response.text
    assert list(response.json()["results"]) == labels
    for result in response.json()["results"].values():
        assert result["combination"] == chosen
        assert result["disclaimer"] == routes.pred.DISCLAIMER
        assert result["generation"]["combinations"][0] == chosen
        assert result["generation"]["evaluation"]["status"] == "not_requested"
    kwargs = interpret.call_args.kwargs
    assert kwargs["picks"] == cfg.picks
    assert kwargs["game_name"] == cfg.name
    assert kwargs["chi_sq_result"] == "Digit uniformity"
    assert kwargs["gap_result"] == "Digit repeats"
    assert position_heat.call_count == len(times)
    for call in position_heat.call_args_list:
        assert len(call.args[0]) == 1
        assert call.args[1] is cfg
    for slot in kwargs["slots"]:
        assert slot["total_draws"] == 1
        assert len(slot["heat_map_rows"]) == cfg.picks * (cfg.pool - cfg.low_boundary + 1)
        for row in slot["heat_map_rows"]:
            assert set(row) == {"number", "position", "total_draws"}
            assert all(type(value) is int for value in row.values())
            assert row["total_draws"] == int(row["number"] == chosen[row["position"] - 1])
    for call in rng.choice.call_args_list:
        assert len(call.args) == 1
        np.testing.assert_array_equal(call.args[0], np.arange(cfg.low_boundary, cfg.pool + 1))
        assert call.kwargs == {"size": cfg.picks, "replace": True}
    assert rng.choice.call_count == len(times)
    trainer.assert_not_called()
    create.assert_called_once()
    uniformity.assert_called_once()
    repeats.assert_called_once()


def model_generation(combinations, ordered=False):
    """Grounded fake evidence accepted by the real AI payload validator."""
    return {
        "method": "logistic_regression", "combinations": combinations,
        "scores": [
            {"number": number, "score": 0.15, **({"position": pos} if ordered else {})}
            for pos, number in enumerate(combinations[0], 1)
        ],
        "feature_weights": {"frequency_last10": 0.25}, "training_draws": 90,
        "reason": "Untuned logistic scoring.",
        "evaluation": {"status": "evaluated", "model_brier": 0.1,
                       "baseline_brier": 0.12, "evidence_of_improvement": False},
    }


@pytest.fixture
def grounded_ai(monkeypatch):
    """Exercise the real prompt/response adapter with only network I/O mocked."""
    ai_client = AzureAIClient()
    monkeypatch.setattr(ai_client, "is_configured", Mock(return_value=True))
    content = "\n".join(
        f"SLOT: {label}\nREASONING: Local generation evidence.\nDISCLAIMER: Fair draws."
        for label in ("2:00 PM", "5:00 PM", "9:00 PM")
    )
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content),
    )]))
    monkeypatch.setattr(ai_client, "_client", Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )))
    interpret = Mock(wraps=ai_client.interpret_lotto_stats)
    monkeypatch.setattr(ai_client, "interpret_lotto_stats", interpret)
    monkeypatch.setattr("src.lotto.azure_ai.AzureAIClient", Mock(return_value=ai_client))
    return interpret, create


@pytest.mark.parametrize("game_id", ["655", "3d"])
def test_ml_default_ai_explains_exact_probability_ticket(client, monkeypatch, grounded_ai, game_id):
    cfg = GAMES[game_id]
    chosen = [9, 0, 9] if cfg.ordered else [6, 5, 4, 3, 2, 1]
    frame = draws(game_id, [(dt, chosen) for dt in cfg.draw_times] * 3)
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))
    generation = model_generation([chosen] * 5, ordered=cfg.ordered)
    generate = Mock(return_value=generation)
    monkeypatch.setattr(routes.lean_ml, "generate_ml_combinations", generate)
    monkeypatch.setattr(routes.pred, "suggest_combinations", Mock(side_effect=AssertionError("No resampling")))
    for name in ("freq_chi_square", "gap_analysis", "digit_uniformity", "digit_repeat_rate"):
        monkeypatch.setattr(routes.an, name, Mock(return_value=SimpleNamespace(conclusion=name)))

    probability = client.get(f"/analysis/{game_id}/probability?draw_time=21:00&window=1")
    response = client.post(f"/analysis/{game_id}/ai_summary", json={"window": 1})

    assert probability.status_code == response.status_code == 200, response.text
    assert probability.json()["suggestions"] == generation["combinations"]
    for result in response.json()["results"].values():
        assert result["combination"] == probability.json()["suggestions"][0]
        assert result["generation"] == probability.json()["generation"] == generation
    interpret, create = grounded_ai
    assert len(interpret.call_args.kwargs["slots"]) == len(cfg.draw_times)
    for slot in interpret.call_args.kwargs["slots"]:
        assert slot["generation"] is generation
    assert generate.call_count == 1 + len(cfg.draw_times)
    assert [call.kwargs["draw_time"] for call in generate.call_args_list[1:]] == list(cfg.draw_times)
    for call in generate.call_args_list:
        assert call.args[0] is frame  # Never pass the window-truncated heat-map frame.
        assert call.args[1] is cfg
        assert call.kwargs == {"n": 5, "window": 1, "draw_time": call.kwargs["draw_time"], "seed": 42}
    prompt = str(create.call_args.kwargs["messages"])
    assert "frequency_last10" in prompt
    assert "model_brier" in prompt
    assert "Training draws: 90" in prompt


@pytest.mark.parametrize("endpoint", ["probability", "ai_summary"])
def test_multi_slot_ml_never_pools_training(client, monkeypatch, ai, endpoint):
    cfg = GAMES["3d"]
    # Combined history exceeds the training minimum; each real slot is too short.
    times = cfg.draw_times[:2]
    frame = draws("3d", [(dt, [9, 0, 9]) for _ in range(60) for dt in times])
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))
    generate = Mock(wraps=routes.lean_ml.generate_ml_combinations)
    monkeypatch.setattr(routes.lean_ml, "generate_ml_combinations", generate)
    fit = Mock(side_effect=AssertionError("Pooling would incorrectly train a model"))
    monkeypatch.setattr(routes.lean_ml, "_fit_classifier", fit)
    for name in ("digit_uniformity", "digit_repeat_rate"):
        monkeypatch.setattr(routes.an, name, Mock(return_value=SimpleNamespace(conclusion=name)))

    if endpoint == "probability":
        response = client.get("/analysis/3d/probability?n_suggestions=2")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["suggestions"] == []
        assert payload["generation"] is None
        assert list(payload["by_draw_time"]) == list(times)
        generations = list(payload["by_draw_time"].values())
    else:
        response = client.post("/analysis/3d/ai_summary", json={"n_suggestions": 2})
        assert response.status_code == 200, response.text
        generations = [slot["generation"] for slot in ai.interpret_lotto_stats.call_args.kwargs["slots"]]
    assert [call.kwargs["draw_time"] for call in generate.call_args_list] == list(times)
    for call in generate.call_args_list:
        assert call.args[0] is frame
        assert call.kwargs == {"n": 2, "window": None, "draw_time": call.kwargs["draw_time"], "seed": 42}
    for generation in generations:
        assert generation["method"] == "uniform_fallback"
        assert generation["training_draws"] == 0
        assert generation["evaluation"]["usable_draws"] == 60
        assert len(generation["combinations"]) == 2
    fit.assert_not_called()


@pytest.mark.parametrize("game_id", ["655", "3d"])
def test_uniform_probability_and_ai_return_identical_seeded_payload(client, monkeypatch, grounded_ai, game_id):
    cfg = GAMES[game_id]
    chosen = [9, 0, 9] if cfg.ordered else [1, 2, 3, 4, 5, 6]
    frame = draws(game_id, [(dt, chosen) for dt in cfg.draw_times] * 2)
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))
    trainer = Mock(side_effect=AssertionError("Uniform must bypass ML"))
    monkeypatch.setattr(routes.lean_ml, "generate_ml_combinations", trainer)
    suggest = Mock(wraps=routes.pred.suggest_combinations)
    monkeypatch.setattr(routes.pred, "suggest_combinations", suggest)
    for name in ("freq_chi_square", "gap_analysis", "digit_uniformity", "digit_repeat_rate"):
        monkeypatch.setattr(routes.an, name, Mock(return_value=SimpleNamespace(conclusion=name)))

    probability = client.get(f"/analysis/{game_id}/probability?method=uniform&n_suggestions=3&window=1&draw_time=21:00")
    response = client.post(f"/analysis/{game_id}/ai_summary", json={
        "method": "uniform", "n_suggestions": 3, "window": 1,
    })

    assert probability.status_code == response.status_code == 200, response.text
    payload = probability.json()
    generation = payload["generation"]
    assert len(payload["suggestions"]) == 3
    assert payload["suggestions"] == generation["combinations"]
    assert generation["method"] == "uniform_fallback"
    assert generation["scores"] == []
    assert generation["feature_weights"] == {}
    assert generation["training_draws"] == 0
    assert generation["evaluation"]["status"] == "not_requested"
    assert "Uniform requested" in generation["reason"]
    result = response.json()["results"]["9:00 PM"]
    assert result["generation"] == generation
    assert result["combination"] == payload["suggestions"][0]
    assert suggest.call_count == 1 + len(cfg.draw_times)
    for call in suggest.call_args_list:
        assert call.args[0] is frame
        assert call.kwargs == {"n": 3, "window": 1, "draw_time": call.kwargs["draw_time"], "seed": 42}
    trainer.assert_not_called()


@pytest.mark.parametrize("endpoint", ["probability", "ai_summary"])
@pytest.mark.parametrize("field,value", [("method", "invalid"), ("n_suggestions", 0), ("n_suggestions", 21)])
def test_generation_arguments_validated(client, monkeypatch, endpoint, field, value):
    load = Mock(side_effect=AssertionError("Validation must precede loading"))
    monkeypatch.setattr(routes, "_load_draws", load)
    if endpoint == "probability":
        response = client.get(f"/analysis/655/{endpoint}", params={field: value})
    else:
        response = client.post(f"/analysis/655/{endpoint}", json={field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    load.assert_not_called()


def test_unordered_ai_uses_number_heat_map(client, monkeypatch, ai):
    frame = draws("655", [("21:00", [1, 2, 3, 4, 5, 6])])
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))
    heat = Mock(return_value=pd.DataFrame([{"number": n, "total_draws": 0} for n in range(1, 56)]))
    monkeypatch.setattr(routes.pred, "number_heat_map", heat)
    monkeypatch.setattr(routes.pred, "position_heat_map", Mock(side_effect=AssertionError("Not ordered")))
    for name in ("freq_chi_square", "gap_analysis"):
        monkeypatch.setattr(routes.an, name, Mock(return_value=SimpleNamespace(conclusion=name)))
    for name in ("digit_uniformity", "digit_repeat_rate"):
        monkeypatch.setattr(routes.an, name, Mock(side_effect=AssertionError("Not digits")))

    response = client.post("/analysis/655/ai_summary", json={"window": 1})

    assert response.status_code == 200
    heat.assert_called_once()
    assert heat.call_args.kwargs["window"] == 1
    slot = ai.interpret_lotto_stats.call_args.kwargs["slots"][0]
    assert slot["label"] == "9:00 PM"
    assert slot["heat_map_rows"] == heat.return_value.to_dict(orient="records")
    assert all("position" not in row for row in slot["heat_map_rows"])


@pytest.mark.parametrize("game_id", ["3d", "4d", "655"])
@pytest.mark.parametrize("empty_kind", ["no_draws", "unusable_draws", "unknown_slot"])
def test_empty_slots_return_404(client, monkeypatch, ai, game_id, empty_kind):
    cfg = GAMES[game_id]
    frame = draws(game_id, [("21:00", [None] * cfg.picks)])
    if empty_kind == "no_draws":
        frame = frame.iloc[:0]
    elif empty_kind == "unknown_slot":
        frame = draws(game_id, [("unknown", [cfg.low_boundary] * cfg.picks)])
    monkeypatch.setattr(routes, "_load_draws", Mock(return_value=frame))

    response = client.post(f"/analysis/{game_id}/ai_summary", json={})

    assert response.status_code == 404
    ai.interpret_lotto_stats.assert_not_called()


@pytest.mark.parametrize("window", [0, -1])
@pytest.mark.parametrize("endpoint", ["probability", "ai_summary"])
def test_invalid_window_rejected_before_loading(client, monkeypatch, window, endpoint):
    load = Mock(side_effect=AssertionError("Validation must precede loading"))
    monkeypatch.setattr(routes, "_load_draws", load)
    if endpoint == "probability":
        response = client.get(f"/analysis/655/probability?window={window}")
    else:
        response = client.post("/analysis/655/ai_summary", json={"window": window})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "window"
    load.assert_not_called()


@pytest.mark.parametrize("operation", ["load", "analysis"])
@pytest.mark.parametrize("failure", ["execute", "fetchdf", None])
def test_connections_always_closed(monkeypatch, operation, failure):
    conn = Mock()
    conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{"n1": 1}])
    if failure == "execute":
        conn.execute.side_effect = RuntimeError("query failed")
    elif failure == "fetchdf":
        conn.execute.return_value.fetchdf.side_effect = RuntimeError("query failed")
    monkeypatch.setattr(routes, "get_connection", Mock(return_value=conn))
    result = SimpleNamespace(name="freq_chi_square", conclusion="OK", table=pd.DataFrame(), meta={})
    monkeypatch.setitem(routes._PICK6_TESTS, "freq_chi_square", Mock(return_value=result))
    def invoke():
        if operation == "load":
            return routes._load_draws(GAMES["655"])
        return routes.run_analysis("655", "freq_chi_square")
    if failure:
        with pytest.raises(RuntimeError, match="query failed"):
            invoke()
    else:
        invoke()
    conn.close.assert_called_once()


def test_generic_analysis_route_still_registered(client, monkeypatch):
    conn = Mock()
    conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{"n1": 1}])
    monkeypatch.setattr(routes, "get_connection", Mock(return_value=conn))
    result = SimpleNamespace(name="freq_chi_square", conclusion="OK", table=pd.DataFrame(), meta={})
    monkeypatch.setitem(routes._PICK6_TESTS, "freq_chi_square", Mock(return_value=result))
    response = client.get("/analysis/655/freq_chi_square")
    assert response.status_code == 200
    assert response.json()["test"] == "freq_chi_square"
    conn.close.assert_called_once()