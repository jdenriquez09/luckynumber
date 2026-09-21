"""Tests for data validation and cleaning rules."""
import pytest
from src.lotto.clean import validate_row
from src.lotto.games import GAMES

G655 = GAMES["655"]  # pick 6 from 1-55, Mon/Wed/Sat


def _base_row(**overrides):
    """Valid 6/55 row for a Monday."""
    row = {
        "draw_date": "2025-01-06",  # Monday
        "weekday": "Mon",
        "game": "655",
        "draw_time": "21:00",
        "pub_1": 5, "pub_2": 12, "pub_3": 23, "pub_4": 34, "pub_5": 45, "pub_6": 55,
        "jackpot_php": 50_000_000.0,
        "notes": None,
    }
    row.update(overrides)
    return row


class TestValidRow:
    def test_valid_row_passes(self):
        result = validate_row(_base_row(), G655)
        assert result["notes"] is None
        assert result["n1"] == 5
        assert result["n6"] == 55

    def test_numbers_are_sorted_in_n_columns(self):
        row = _base_row(pub_1=55, pub_2=45, pub_3=34, pub_4=23, pub_5=12, pub_6=5)
        result = validate_row(row, G655)
        assert result["n1"] == 5 and result["n6"] == 55


class TestWrongWeekday:
    def test_tuesday_flagged_for_655(self):
        # 2025-01-07 is a Tuesday; 6/55 draws Mon/Wed/Sat
        row = _base_row(draw_date="2025-01-07", weekday="Tue")
        result = validate_row(row, G655)
        assert result["notes"] is not None
        assert "Tue" in result["notes"] or "draw_date" in result["notes"].lower() or "draws on" in result["notes"]

    def test_draw_date_not_blanked_on_wrong_weekday(self):
        row = _base_row(draw_date="2025-01-07", weekday="Tue")
        result = validate_row(row, G655)
        assert result["draw_date"] == "2025-01-07"  # flag only, not blanked


class TestOutOfRangeNumbers:
    def test_number_above_pool_blanked(self):
        row = _base_row(pub_6=56)  # max is 55
        result = validate_row(row, G655)
        assert result["n1"] is None
        assert "out of range" in (result["notes"] or "")

    def test_number_zero_blanked(self):
        row = _base_row(pub_1=0)  # min is 1
        result = validate_row(row, G655)
        assert result["n1"] is None

    def test_number_negative_blanked(self):
        row = _base_row(pub_1=-1)
        result = validate_row(row, G655)
        assert result["n1"] is None


class TestDuplicateNumbers:
    def test_duplicate_number_blanked(self):
        row = _base_row(pub_1=5, pub_2=5)  # duplicate
        result = validate_row(row, G655)
        assert result["n1"] is None
        assert "duplicate" in (result["notes"] or "")


class TestBadJackpot:
    def test_negative_jackpot_blanked(self):
        row = _base_row(jackpot_php=-1000)
        result = validate_row(row, G655)
        assert result["jackpot_php"] is None
        assert result["notes"] is not None

    def test_none_jackpot_allowed(self):
        row = _base_row(jackpot_php=None)
        result = validate_row(row, G655)
        assert result["jackpot_php"] is None
        assert result["notes"] is None

    def test_valid_jackpot_preserved(self):
        row = _base_row(jackpot_php=50_000_000.0)
        result = validate_row(row, G655)
        assert result["jackpot_php"] == 50_000_000.0


class TestMissingNumbers:
    def test_missing_number_blanks_all(self):
        row = _base_row(pub_6=None)
        result = validate_row(row, G655)
        assert result["n1"] is None
        assert "missing" in (result["notes"] or "")
