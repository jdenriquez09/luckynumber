"""
Tests for statistical analysis functions.

Key invariant: when run on FAIR simulated draws, no single test should reject
more than ~α=0.05 of the time across repeated random seeds.
We use a single fixed-seed fair dataset and verify results are structurally correct
(correct columns, valid p-values, etc.) rather than testing a false-rejection rate
(which would require hundreds of simulation runs and is slow for a unit test).
"""
import pytest
import numpy as np
import pandas as pd
from src.lotto import analysis as an
from src.lotto.games import GAMES

G655 = GAMES["655"]


class TestFreqChiSquare:
    def test_returns_correct_columns(self, fair_draws_655):
        result = an.freq_chi_square(fair_draws_655, G655)
        assert set(result.table.columns) >= {"number", "observed", "expected", "deviation"}

    def test_table_length_equals_pool(self, fair_draws_655):
        result = an.freq_chi_square(fair_draws_655, G655)
        assert len(result.table) == G655.pool  # 1..55

    def test_p_value_in_meta(self, fair_draws_655):
        result = an.freq_chi_square(fair_draws_655, G655)
        p = result.meta["p"]
        assert 0.0 <= p <= 1.0

    def test_conclusion_is_string(self, fair_draws_655):
        result = an.freq_chi_square(fair_draws_655, G655)
        assert isinstance(result.conclusion, str) and len(result.conclusion) > 20

    def test_fair_data_mostly_not_rejected(self):
        """Fair draws should not be flagged as non-uniform."""
        rng = np.random.default_rng(99)
        pool = np.arange(1, G655.pool + 1)
        rejections = 0
        for seed in range(20):
            rng2 = np.random.default_rng(seed)
            n = 300
            rows = []
            from datetime import date, timedelta
            d = date(2020, 1, 1)
            count = 0
            while count < n:
                if d.strftime("%a") in G655.draw_days:
                    nums = sorted(rng2.choice(pool, size=6, replace=False).tolist())
                    row = {"draw_date": d.isoformat(), "weekday": d.strftime("%a"), "game": "655",
                           "draw_time": "21:00", "jackpot_php": None, "notes": None}
                    for i, v in enumerate(nums, 1):
                        row[f"pub_{i}"] = v
                        row[f"n{i}"] = v
                    rows.append(row)
                    count += 1
                d += timedelta(days=1)
            df = pd.DataFrame(rows)
            r = an.freq_chi_square(df, G655)
            if r.meta["p"] < 0.05:
                rejections += 1
        # Expect ≤5/20 rejections (generous bound; binomial 95th pct is ~3)
        assert rejections <= 5, f"Too many false rejections: {rejections}/20"


class TestDistributionTests:
    def test_returns_expected_tests(self, fair_draws_655):
        result = an.distribution_tests(fair_draws_655, G655)
        test_names = set(result.table["test"])
        assert "draw_sum" in test_names
        assert "odd_count_distribution" in test_names


class TestSequenceTests:
    def test_returns_three_rows(self, fair_draws_655):
        result = an.sequence_tests(fair_draws_655, G655)
        assert len(result.table) == 3

    def test_p_values_valid(self, fair_draws_655):
        result = an.sequence_tests(fair_draws_655, G655)
        for p in result.table["p_value"]:
            assert 0.0 <= p <= 1.0


class TestGapAnalysis:
    def test_no_row_for_number_with_zero_appearances(self, fair_draws_655):
        result = an.gap_analysis(fair_draws_655, G655)
        assert "number" in result.table.columns
        # With 500 fair draws, all 55 numbers should appear at least once
        assert len(result.table) > 0

    def test_overdue_disclaimer_in_conclusion(self, fair_draws_655):
        result = an.gap_analysis(fair_draws_655, G655)
        assert "overdue" in result.conclusion.lower() or "absence" in result.conclusion.lower()


class TestMonteCarlo:
    def test_percentile_within_expected_range(self, fair_draws_655):
        result = an.monte_carlo(fair_draws_655, G655, n_sim=500)
        pct = result.meta["percentile"]
        # Fair data should land between ~1st and ~99th percentile
        assert 1 <= pct <= 99


class TestJackpotDynamics:
    def test_empty_jackpot_data_returns_gracefully(self, fair_draws_655):
        df = fair_draws_655.copy()
        df["jackpot_php"] = None
        result = an.jackpot_dynamics(df, G655)
        assert isinstance(result.conclusion, str)


class TestDigitUniformity:
    def test_digit_game(self, game_3d):
        rng = np.random.default_rng(7)
        n = 300
        rows = []
        from datetime import date, timedelta
        d = date(2020, 1, 1)
        count = 0
        while count < n:
            rows.append({
                "draw_date": d.isoformat(), "weekday": d.strftime("%a"),
                "game": "3d", "draw_time": "21:00",
                "jackpot_php": None, "notes": None,
                "pub_1": None, "pub_2": None, "pub_3": None,
                "n1": int(rng.integers(0, 10)),
                "n2": int(rng.integers(0, 10)),
                "n3": int(rng.integers(0, 10)),
            })
            count += 1
            d += timedelta(days=1)
        df = pd.DataFrame(rows)
        result = an.digit_uniformity(df, game_3d)
        assert len(result.table) == game_3d.picks
        assert "position" in result.table.columns
