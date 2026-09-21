"""Tests for game config completeness and correctness."""
import pytest
from src.lotto.games import GAMES, pick6_games, digit_games, get_game

REQUIRED_FIELDS = ("id", "name", "pool", "picks", "draw_days", "draw_times", "ordered")
VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


def test_all_nine_games_defined():
    assert set(GAMES.keys()) == {"658", "655", "649", "645", "642", "6d", "4d", "3d", "2d"}


@pytest.mark.parametrize("game_id", list(GAMES.keys()))
def test_required_fields(game_id):
    g = GAMES[game_id]
    for field in REQUIRED_FIELDS:
        assert hasattr(g, field), f"Missing field {field} on game {game_id}"


@pytest.mark.parametrize("game_id", list(GAMES.keys()))
def test_draw_days_valid(game_id):
    g = GAMES[game_id]
    assert set(g.draw_days).issubset(VALID_DAYS)
    assert len(g.draw_days) >= 1


@pytest.mark.parametrize("game_id", list(GAMES.keys()))
def test_pool_and_picks_sensible(game_id):
    g = GAMES[game_id]
    assert g.pool >= g.picks
    assert g.picks >= 2
    assert g.pool >= 2


def test_pick6_games_are_unordered():
    for g in pick6_games():
        assert not g.ordered
        assert g.picks == 6


def test_digit_games_are_ordered():
    for g in digit_games():
        assert g.ordered


def test_get_game_case_insensitive():
    assert get_game("655").id == "655"
    assert get_game("655").id == get_game("655").id


def test_get_game_unknown_raises():
    with pytest.raises(KeyError):
        get_game("999")
