"""Tests for the scraper: caching, rate-limiting, parsing."""
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.lotto.scrape import fetch_page, parse_pick6_page, _cache_path
from src.lotto.games import GAMES

G655 = GAMES["655"]

SAMPLE_HTML = """
<html><body>
<table>
  <tr><th>Date</th><th>Result</th><th>Jackpot</th></tr>
  <tr><td>Monday, January 06, 2025</td><td>05-12-23-34-45-55</td><td>50,000,000.00</td></tr>
  <tr><td>Wednesday, January 08, 2025</td><td>01-07-14-28-42-50</td><td>49,000,000.00</td></tr>
  <tr><td>Saturday, January 11, 2025</td><td>03-15-22-31-40-53</td><td>51,000,000.00</td></tr>
</table>
</body></html>
"""


class TestParser:
    def test_parses_three_rows(self):
        rows = parse_pick6_page(SAMPLE_HTML, G655)
        assert len(rows) == 3

    def test_row_has_correct_date(self):
        rows = parse_pick6_page(SAMPLE_HTML, G655)
        assert rows[0]["draw_date"] == "2025-01-06"

    def test_row_has_six_numbers(self):
        rows = parse_pick6_page(SAMPLE_HTML, G655)
        nums = [rows[0][f"pub_{i}"] for i in range(1, 7)]
        assert nums == [5, 12, 23, 34, 45, 55]

    def test_jackpot_parsed(self):
        rows = parse_pick6_page(SAMPLE_HTML, G655)
        assert rows[0]["jackpot_php"] == pytest.approx(50_000_000.0)

    def test_empty_table_returns_empty(self):
        rows = parse_pick6_page("<html><body><table></table></body></html>", G655)
        assert rows == []


class TestCaching:
    def test_cache_path_is_deterministic(self):
        url = "https://example.com/page"
        p1 = _cache_path(url)
        p2 = _cache_path(url)
        assert p1 == p2

    def test_different_urls_different_paths(self):
        assert _cache_path("https://a.com/1") != _cache_path("https://a.com/2")

    @pytest.mark.asyncio
    async def test_fetch_uses_cache(self, tmp_path, monkeypatch):
        url = "https://example.com/cached"
        cache_p = _cache_path(url)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(SAMPLE_HTML, encoding="utf-8")

        # If cache is hit, no HTTP call should be made
        called = []
        original_import = __import__

        html, p = await fetch_page(url)
        assert html == SAMPLE_HTML
        cache_p.unlink(missing_ok=True)
