"""Tests for attribution diagnostics."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.diagnostics.attribution import AttributionReport, SeriesAdjustment


def make_wide_dfs(n_periods=6, n_series=3):
    """Build simple before/after DataFrames."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    cols = [f"series_{i}" for i in range(n_series)]
    before = pd.DataFrame(rng.uniform(0.02, 0.05, (n_periods, n_series)), index=idx, columns=cols)
    # After: small random adjustments
    after = before + rng.normal(0, 0.002, (n_periods, n_series))
    after = after.clip(lower=0.001)
    return before, after


class TestSeriesAdjustment:
    def test_str(self):
        adj = SeriesAdjustment(
            series_name="Fire",
            mean_before=0.025,
            mean_after=0.022,
            mean_adjustment=-0.003,
            mean_adjustment_pct=-12.0,
            max_abs_adjustment=0.005,
            total_ep=5e6,
            ep_weighted_lc_impact=-15000.0,
        )
        s = str(adj)
        assert "Fire" in s
        assert "0.025" in s


class TestAttributionReport:
    def test_from_before_after(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        assert report.n_series == 3
        assert len(report.adjustments) == 3

    def test_with_ep(self):
        before, after = make_wide_dfs()
        ep = before * 1e8  # Arbitrary EP
        report = AttributionReport.from_before_after(before, after, earned_premium_df=ep)
        assert report.portfolio_lc_before is not None
        assert report.portfolio_lc_after is not None

    def test_no_ep(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        assert report.portfolio_lc_before is None

    def test_to_dataframe(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        df = report.to_dataframe()
        assert "series_name" in df.columns
        assert "mean_adjustment_pct" in df.columns
        assert len(df) == 3

    def test_to_string(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        s = report.to_string()
        assert "Attribution Report" in s

    def test_largest_adjustments(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        largest = report.largest_adjustments(n=2)
        assert len(largest) <= 2
        # Should be sorted by abs pct
        if len(largest) == 2:
            assert abs(largest[0].mean_adjustment_pct) >= abs(largest[1].mean_adjustment_pct)

    def test_repr(self):
        before, after = make_wide_dfs()
        report = AttributionReport.from_before_after(before, after)
        r = repr(report)
        assert "AttributionReport" in r

    def test_zero_adjustment(self):
        """Before and after identical — all adjustments should be zero."""
        before, _ = make_wide_dfs()
        report = AttributionReport.from_before_after(before, before)
        for adj in report.adjustments:
            assert adj.mean_adjustment == pytest.approx(0.0, abs=1e-10)

    def test_to_dataframe_sorted_by_abs(self):
        before, after = make_wide_dfs(n_series=5)
        # Make series_0 have a large adjustment
        after_big = after.copy()
        after_big["series_0"] = before["series_0"] * 2.0
        report = AttributionReport.from_before_after(before, after_big)
        df = report.to_dataframe()
        # First row should have series_0 (largest adjustment)
        assert df.iloc[0]["series_name"] == "series_0"
