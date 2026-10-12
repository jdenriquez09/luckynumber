"""Lucky-number engine and the ML + lucky hybrid: synthetic data only."""
import datetime as dt

import numpy as np
import pytest

from src.lotto import arena, lucky
from src.lotto.games import GAMES
from tests.conftest import _fair_draws
from tests.test_lean_ml import _draws, _tickets_valid


def day_pillar(d):
    return lucky.day_profile(d)["day_pillar"]


def test_day_pillar_matches_known_calendar_dates():
    assert day_pillar(dt.date(2000, 1, 7)) == "Jia-Zi"    # the usual sexagenary epoch
    assert day_pillar(dt.date(2000, 1, 1)) == "Wu-Wu"
    assert day_pillar(dt.date(2024, 2, 10)) == "Jia-Chen"  # Chinese New Year 2024
    assert day_pillar(dt.date(2024, 2, 11)) == "Yi-Si"


def test_year_star_and_year_pillar_turn_over_at_li_chun():
    assert lucky.day_profile(dt.date(2024, 6, 1))["year_star"] == 3
    assert lucky.day_profile(dt.date(2025, 6, 1))["year_star"] == 2
    assert lucky.day_profile(dt.date(2026, 6, 1))["year_star"] == 1
    before, after = lucky.day_profile(dt.date(2026, 2, 3)), lucky.day_profile(dt.date(2026, 2, 4))
    assert before["year_pillar"] == "Yi-Si" and after["year_pillar"] == "Bing-Wu"
    assert after["year_animal"] == "Horse"


def test_moon_age_is_one_on_a_known_new_moon():
    assert lucky.day_profile(dt.date(2024, 2, 10))["moon_age_day"] == 1
    assert 12 <= lucky.day_profile(dt.date(2024, 2, 24))["moon_age_day"] <= 16  # full moon 24 Feb


def test_scores_cover_the_pool_and_explain_themselves():
    cfg = GAMES["658"]
    result = lucky.lucky_scores(cfg, dt.date(2026, 10, 9))
    assert result.numbers == list(range(1, 59)) and len(result.scores) == 58
    assert (result.scores >= 0).all()
    assert all(row["reasons"] for row in result.top(10))
    assert result.top(3)[0]["score"] >= result.top(3)[-1]["score"]
    payload = result.to_dict()
    assert payload["profile"]["day_pillar"] == "Bing-Chen" and len(payload["scores"]) == 58


def test_scores_depend_on_the_date_only():
    cfg = GAMES["655"]
    first = lucky.lucky_scores(cfg, dt.date(2026, 3, 4)).scores
    assert np.array_equal(first, lucky.lucky_scores(cfg, dt.date(2026, 3, 4)).scores)
    assert not np.array_equal(first, lucky.lucky_scores(cfg, dt.date(2026, 3, 5)).scores)


def test_fours_are_discounted_and_eights_are_favoured():
    cfg = GAMES["658"]
    result = lucky.lucky_scores(cfg, dt.date(2026, 10, 9))  # Dragon day: digits 1, 6, 7 score
    assert result.scores[13] > 0 and "contains 4 (taboo)" in result.reasons[14]
    assert "contains 4 (taboo)" not in result.reasons[13]
    assert result.scores[13] < result.scores[10]  # 14 is discounted below 11, which shares its digit 1
    assert "auspicious digits 8" in result.reasons[8]


def test_personal_and_own_numbers_raise_their_scores():
    cfg = GAMES["658"]
    day = dt.date(2026, 10, 9)
    base = lucky.lucky_scores(cfg, day)
    mine = lucky.lucky_scores(cfg, day, birthdate=dt.date(1990, 7, 23), extra_numbers=[50])
    assert mine.scores[49] == pytest.approx(base.scores[49] + lucky.EXTRA_NUMBER_WEIGHT)
    assert mine.scores[22] > base.scores[22]  # birth day 23
    assert mine.scores[6] > base.scores[6]    # birth month 7
    assert "your own lucky number" in mine.reasons[50]
    assert mine.personal["life_path"] == lucky.digit_root(sum(int(c) for c in "19900723"))


def test_parse_extra_numbers_keeps_pool_numbers_in_order_without_repeats():
    cfg = GAMES["658"]
    assert lucky.parse_extra_numbers("7, 23 and 41; 7 plus 99 and 0", cfg) == [7, 23, 41]
    assert lucky.parse_extra_numbers("", cfg) == [] and lucky.parse_extra_numbers(None, cfg) == []
    assert lucky.parse_extra_numbers("0 3 9 10", GAMES["3d"]) == [0, 3, 9]


def test_next_draw_date_counts_today_until_the_last_slot_starts():
    cfg = GAMES["658"]  # Tue / Fri / Sun at 21:00
    friday = dt.datetime(2026, 10, 9, 15, 0)
    assert lucky.next_draw_date(cfg, friday) == dt.date(2026, 10, 9)
    assert lucky.next_draw_date(cfg, friday.replace(hour=21, minute=5)) == dt.date(2026, 10, 11)
    assert lucky.next_draw_date(GAMES["3d"], dt.datetime(2026, 10, 9, 15)) == dt.date(2026, 10, 9)


@pytest.mark.parametrize("game_id", ["658", "645", "6d", "4d", "3d", "2d"])
def test_hybrid_generation_splits_every_ticket_between_ml_and_lucky(game_id):
    game = GAMES[game_id]
    result = lucky.hybrid_generation(_draws(game, 150), game, n=6, draw_date=dt.date(2026, 10, 9))
    assert result["method"] == "hybrid_ml_lucky" and result["ml_method"] == "logistic_regression"
    _tickets_valid(result["combinations"], game, 6)
    assert result["ml_count"] == lucky.default_ml_count(game)
    assert result["ml_count"] + result["lucky_count"] == game.picks
    for ticket in result["tickets"]:
        assert len(ticket["ml"]) == result["ml_count"] and len(ticket["lucky"]) == result["lucky_count"]
        assert sorted(ticket["ml"] + ticket["lucky"]) == sorted(ticket["numbers"])
    assert result["evaluation"]["status"] == "evaluated" and result["lucky"]["top_numbers"]


def test_hybrid_is_three_plus_three_on_six_number_games_and_never_overlaps():
    game = GAMES["658"]
    result = lucky.hybrid_generation(_draws(game, 150), game, n=20, draw_date=dt.date(2026, 10, 9))
    assert (result["ml_count"], result["lucky_count"]) == (3, 3)
    for ticket in result["tickets"]:
        assert len(set(ticket["ml"]) | set(ticket["lucky"])) == 6


def test_first_ticket_uses_strongest_lucky_numbers_not_taken_by_ml():
    game = GAMES["658"]
    day = dt.date(2026, 10, 9)
    result = lucky.hybrid_generation(_draws(game, 150), game, n=3, draw_date=day)
    first = result["tickets"][0]
    ranked = [row["number"] for row in lucky.lucky_scores(game, day).top(58)
              if row["number"] not in first["ml"]]
    assert first["lucky"] == sorted(ranked[:3])


def test_hybrid_is_reproducible_per_date_and_changes_with_the_date():
    game = GAMES["649"]
    draws = _draws(game, 120)
    a = lucky.hybrid_generation(draws, game, n=5, draw_date=dt.date(2026, 10, 9))
    assert a["combinations"] == lucky.hybrid_generation(draws, game, n=5, draw_date=dt.date(2026, 10, 9))["combinations"]
    assert a["combinations"] != lucky.hybrid_generation(draws, game, n=5, draw_date=dt.date(2026, 10, 13))["combinations"]


def test_hybrid_falls_back_to_random_ml_half_without_enough_history():
    game = GAMES["658"]
    result = lucky.hybrid_generation(_draws(game, 20), game, n=4, draw_date=dt.date(2026, 10, 9))
    assert result["ml_method"] == "uniform_fallback" and result["method"] == "hybrid_ml_lucky"
    _tickets_valid(result["combinations"], game, 4)


def test_ml_count_extremes_give_pure_lucky_or_pure_ml_tickets():
    game = GAMES["658"]
    draws, day = _draws(game, 120), dt.date(2026, 10, 9)
    none = lucky.hybrid_generation(draws, game, n=3, draw_date=day, ml_count=0)
    assert all(t["ml"] == [] and len(t["lucky"]) == 6 for t in none["tickets"])
    assert none["tickets"][0]["lucky"] == sorted(r["number"] for r in lucky.lucky_scores(game, day).top(6))
    full = lucky.hybrid_generation(draws, game, n=3, draw_date=day, ml_count=6)
    assert all(t["lucky"] == [] and len(t["ml"]) == 6 for t in full["tickets"])


def test_arena_scores_lucky_and_hybrid_strategies_walk_forward():
    game = GAMES["655"]
    frame = _fair_draws(game, n=220, seed=5)
    result = arena.run_arena(frame, game, strategies=["uniform", "lucky", "hybrid"], test_draws=40,
                             tickets_per_draw=3, ml_window=120, ml_refit_every=20)
    assert set(result.table["key"]) == {"uniform", "lucky", "hybrid"}
    assert (result.table["draws"] == 40).all()
    assert {"Lucky numbers (feng shui + numerology)", "Hybrid (half ML, half lucky numbers)"} <= set(result.table["strategy"])
    assert {"lucky", "hybrid"} <= set(arena.available_strategies(game, None))


def test_generate_tickets_is_the_one_standard_path():
    from src.lotto.tickets import STANDARD_METHOD, generate_tickets

    game = GAMES["658"]
    draws, day = _draws(game, 150), dt.date(2026, 10, 9)
    standard = generate_tickets(draws, game, n=4, draw_date=day, extra_numbers=[7])
    assert standard["method"] == STANDARD_METHOD == "hybrid_ml_lucky"
    same = lucky.hybrid_generation(draws, game, n=4, draw_date=day, extra_numbers=[7])
    assert standard == same  # no second recipe hiding behind the entry point
    assert [(len(t["ml"]), len(t["lucky"])) for t in standard["tickets"]] == [(3, 3)] * 4
    assert all("crowd_index" not in t for t in standard["tickets"])  # only with a crowd model


def test_generate_tickets_adds_crowd_info_for_six_number_games_only():
    from src.lotto.crowd import fit_crowd_model
    from src.lotto.tickets import generate_tickets
    from tests.test_arena import _crowded_frames

    model = fit_crowd_model(_crowded_frames())
    game, day = GAMES["655"], dt.date(2026, 10, 9)
    result = generate_tickets(_draws(game, 150), game, n=3, draw_date=day, crowd=model)
    assert all(t["crowd_index"] > 0 and 0 < t["share_kept"] <= 1 for t in result["tickets"])
    ordered = generate_tickets(_draws(GAMES["3d"], 150), GAMES["3d"], n=3, draw_date=day, crowd=model)
    assert all("crowd_index" not in t for t in ordered["tickets"])
    assert generate_tickets(_draws(game, 150), game, n=3, draw_date=day) ["tickets"][0]["numbers"] == \
        result["tickets"][0]["numbers"]  # crowd info annotates, it never changes the tickets


def _ai_client(monkeypatch, content, finish="stop"):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from src.lotto.azure_ai import AzureAIClient

    for name, value in {"AZURE_OPENAI_ENDPOINT": "https://offline.invalid",
                        "AZURE_OPENAI_API_KEY": "offline-test-only",
                        "AZURE_OPENAI_DEPLOYMENT": "offline-test"}.items():
        monkeypatch.setenv(name, value)
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])
    monkeypatch.setattr(AzureAIClient, "_client", Mock(return_value=sdk))
    return AzureAIClient(), sdk


def test_ai_lucky_numbers_are_validated_and_get_the_calendar_facts(monkeypatch):
    game = GAMES["658"]
    reply = ('Here you go:\n```json\n{"numbers":[{"number":8,"reason":"8 sounds like prosper"},'
             '{"number":8,"reason":"duplicate"},{"number":99,"reason":"out of range"},'
             '{"number":true,"reason":"bool"},{"number":"7","reason":"string"},'
             '{"number":36,"reason":"Fire day, 6 is smooth"}]}\n```')
    client, sdk = _ai_client(monkeypatch, reply)
    found = lucky.ask_ai_lucky_numbers(client, game, dt.date(2026, 10, 9), dt.date(1990, 7, 23))
    assert found == {8: "8 sounds like prosper", 36: "Fire day, 6 is smooth"}
    prompt = sdk.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Bing-Chen" in prompt and '"year_star":1' in prompt and "from 1 to 58" in prompt
    assert '"life_path"' in prompt and "not predictions" in prompt
    assert sdk.chat.completions.create.call_args.kwargs["max_completion_tokens"] > 0
    assert "temperature" not in sdk.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize("content,finish", [("no json at all", "stop"), ('{"numbers":[]}', "stop"),
                                            ('{"numbers":[{"number":99,"reason":"x"}]}', "stop"),
                                            ('{"numbers":[{"number":5,"reason":"x"}]}', "length")])
def test_ai_lucky_numbers_reject_unusable_replies(monkeypatch, content, finish):
    client, _ = _ai_client(monkeypatch, content, finish)
    with pytest.raises(RuntimeError):
        lucky.ask_ai_lucky_numbers(client, GAMES["658"], dt.date(2026, 10, 9))


def test_ai_numbers_add_one_more_scored_source_with_their_reason():
    game, day = GAMES["658"], dt.date(2026, 10, 9)
    base = lucky.lucky_scores(game, day)
    with_ai = lucky.lucky_scores(game, day, ai_numbers={50: "water brings wealth"})
    assert with_ai.scores[49] == pytest.approx(base.scores[49] + lucky.AI_NUMBER_WEIGHT)
    assert "AI suggestion: water brings wealth" in with_ai.reasons[50]
    assert with_ai.personal["ai_numbers"] == [50]
    assert lucky.AI_NUMBER_WEIGHT < lucky.EXTRA_NUMBER_WEIGHT  # your own numbers still count more
    assert np.array_equal(np.delete(with_ai.scores, 49), np.delete(base.scores, 49))
    result = lucky.hybrid_generation(_draws(game, 120), game, n=2, draw_date=day,
                                     ai_numbers={50: "water brings wealth"})
    assert "AI suggestion: water brings wealth" in result["lucky"]["reasons"]["50"]


def _year_of_tops(game, start=dt.date(2026, 10, 13), k=6):
    days = [start + dt.timedelta(days=i) for i in range(365)]
    days = [d for d in days if d.strftime("%a") in game.draw_days]
    return days, [[r["number"] for r in lucky.lucky_scores(game, d).top(k)] for d in days]


@pytest.mark.parametrize("game_id", ["658", "645", "642"])
def test_lucky_numbers_follow_the_draw_date_not_a_fixed_favourite_list(game_id):
    import collections

    days, tops = _year_of_tops(GAMES[game_id])
    most_common = collections.Counter(n for t in tops for n in t).most_common(1)[0][1]
    assert most_common / len(days) < 0.45  # no number sits in the top six on most draw dates
    overlap = np.mean([len(set(a[:3]) & set(b[:3])) for a, b in zip(tops, tops[1:])])
    assert overlap < 0.4  # consecutive draws rarely share their strongest three
    assert len({tuple(t[:3]) for t in tops}) > 0.9 * len(days)


def test_evergreen_sources_are_only_tie_breakers_next_to_date_sources():
    assert lucky.AUSPICIOUS_CAP + lucky.YEAR_STAR_WEIGHT < 1.0  # day of month alone outweighs both


def test_day_pillar_numbers_and_he_tu_digits_come_with_reasons():
    game = GAMES["658"]
    day = dt.date(2026, 10, 9)  # Bing-Chen: stem Bing = 3, branch Chen = 5, element Fire (He Tu 2, 7)
    profile = lucky.day_profile(day)
    assert (profile["day_stem_number"], profile["day_branch_number"]) == (3, 5)
    result = lucky.lucky_scores(game, day)
    assert "day stem Bing = 3" in result.reasons[3] and "day branch Chen = 5" in result.reasons[5]
    for number in (2, 7, 17, 27, 37, 47, 57, 12, 22, 32, 42, 52):
        assert "He Tu number of Fire" in result.reasons[number]
    assert "He Tu number of Fire" not in result.reasons[8]
    water = lucky.lucky_scores(game, dt.date(2026, 10, 15))  # Ren-Xu: a Water day, He Tu numbers 1 and 6
    assert water.profile["day_element"] == "Water" and "He Tu number of Water" in water.reasons[6]
    assert "He Tu number of Fire" not in water.reasons[7] and "He Tu number of Water" not in water.reasons[7]


def test_ai_shortlist_is_where_the_lucky_half_is_drawn_from():
    game, day = GAMES["658"], dt.date(2026, 10, 9)
    shortlist = {n: f"reason {n}" for n in (4, 11, 22, 33, 44, 50, 55, 57)}
    result = lucky.hybrid_generation(_draws(game, 150), game, n=15, draw_date=day, ai_numbers=shortlist)
    assert result["lucky_source"] == "ai_shortlist+date_rules"
    for ticket in result["tickets"]:
        assert set(ticket["lucky"]) <= set(shortlist)  # every ticket, not just the first
    assert lucky.hybrid_generation(_draws(game, 150), game, n=2, draw_date=day)["lucky_source"] == "date_rules"


def test_ai_shortlist_ranks_by_the_date_rules_and_keeps_own_numbers():
    game, day = GAMES["658"], dt.date(2026, 10, 9)
    shortlist = {n: "x" for n in (4, 11, 22, 33, 44, 50, 55, 57)}
    ranked = lucky.lucky_scores(game, day, ai_numbers=shortlist)
    in_list = sorted(shortlist, key=lambda n: (-ranked.scores[n - 1], n))
    first = lucky.hybrid_generation(_draws(game, 150), game, n=1, draw_date=day, ai_numbers=shortlist,
                                    ml_count=0)["tickets"][0]
    assert first["lucky"] == sorted(in_list[:6])  # the strongest shortlisted numbers by date score
    own = lucky.hybrid_generation(_draws(game, 150), game, n=6, draw_date=day, ai_numbers=shortlist,
                                  extra_numbers=[1], ml_count=0)
    assert all(set(t["lucky"]) <= set(shortlist) | {1} for t in own["tickets"])
    assert 1 in own["tickets"][0]["lucky"]  # your own number joins the shortlist


def test_a_small_ai_shortlist_is_topped_up_by_the_date_rules():
    game, day = GAMES["658"], dt.date(2026, 10, 9)
    result = lucky.hybrid_generation(_draws(game, 150), game, n=4, draw_date=day, ml_count=0,
                                     ai_numbers={50: "only one"})
    for ticket in result["tickets"]:
        assert len(ticket["lucky"]) == 6 and 50 in ticket["lucky"]
    rules = [r["number"] for r in lucky.lucky_scores(game, day, ai_numbers={50: "only one"}).top(58)
             if r["number"] != 50][:5]
    assert result["tickets"][0]["lucky"] == sorted(rules + [50])


def test_ai_shortlist_works_for_ordered_games():
    game = GAMES["3d"]
    result = lucky.hybrid_generation(_draws(game, 150), game, n=6, draw_date=dt.date(2026, 10, 9),
                                     ai_numbers={3: "a", 7: "b", 8: "c"})
    for ticket in result["tickets"]:
        assert all(d in {3, 7, 8} for d in ticket["lucky"])
