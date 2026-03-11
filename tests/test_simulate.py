"""Tests for synthetic data generation."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.simulate import (
    simulate_uk_home,
    simulate_incoherent_uk_home,
    make_hierarchy_dataframe,
)


class TestSimulateUkHome:
    def test_basic_shape(self):
        lc_df, ep_df = simulate_uk_home(n_periods=12)
        assert len(lc_df) == 12
        assert len(ep_df) == 12

    def test_columns_present(self):
        lc_df, ep_df = simulate_uk_home()
        expected_cols = {"Fire", "EoW", "Subsidence", "Flood", "Storm",
                         "Theft", "AccidentalDamage", "Buildings", "Contents", "Home"}
        assert expected_cols.issubset(set(lc_df.columns))

    def test_loss_costs_positive(self):
        lc_df, ep_df = simulate_uk_home()
        assert (lc_df >= 0).all().all()

    def test_ep_positive(self):
        lc_df, ep_df = simulate_uk_home()
        assert (ep_df > 0).all().all()

    def test_ep_adds_up(self):
        """Buildings EP = sum of Buildings peril EPs."""
        lc_df, ep_df = simulate_uk_home()
        bld_perils = ["Fire", "EoW", "Subsidence", "Flood", "Storm"]
        expected_bld_ep = ep_df[bld_perils].sum(axis=1)
        np.testing.assert_allclose(
            ep_df["Buildings"].values,
            expected_bld_ep.values,
            rtol=1e-6,
        )

    def test_portfolio_ep_correct(self):
        lc_df, ep_df = simulate_uk_home()
        all_perils = ["Fire", "EoW", "Subsidence", "Flood", "Storm", "Theft", "AccidentalDamage"]
        expected_portfolio_ep = ep_df[all_perils].sum(axis=1)
        np.testing.assert_allclose(
            ep_df["Home"].values,
            expected_portfolio_ep.values,
            rtol=1e-6,
        )

    def test_coherence_by_construction(self):
        """Buildings LC = EP-weighted average of its perils."""
        lc_df, ep_df = simulate_uk_home()
        bld_perils = ["Fire", "EoW", "Subsidence", "Flood", "Storm"]
        ep_bld = ep_df[bld_perils]
        lc_bld = lc_df[bld_perils]
        expected = (lc_bld * ep_bld.values).sum(axis=1) / ep_bld.sum(axis=1)
        np.testing.assert_allclose(
            lc_df["Buildings"].values,
            expected.values,
            rtol=1e-6,
        )

    def test_seed_reproducible(self):
        lc1, ep1 = simulate_uk_home(seed=99)
        lc2, ep2 = simulate_uk_home(seed=99)
        pd.testing.assert_frame_equal(lc1, lc2)

    def test_different_seeds_differ(self):
        lc1, _ = simulate_uk_home(seed=1)
        lc2, _ = simulate_uk_home(seed=2)
        assert not lc1.equals(lc2)

    def test_n_periods_param(self):
        lc_df, ep_df = simulate_uk_home(n_periods=36)
        assert len(lc_df) == 36

    def test_total_ep_param(self):
        lc_df, ep_df = simulate_uk_home(total_ep=100_000_000.0)
        # Portfolio EP should be ~100M (with small seasonality variation)
        assert ep_df["Home"].mean() == pytest.approx(100_000_000.0, rel=0.05)

    def test_index_is_datetime(self):
        lc_df, ep_df = simulate_uk_home()
        assert pd.api.types.is_datetime64_any_dtype(lc_df.index)


class TestSimulateIncoherent:
    def test_returns_two_dfs(self):
        lc_df, ep_df = simulate_incoherent_uk_home()
        assert isinstance(lc_df, pd.DataFrame)
        assert isinstance(ep_df, pd.DataFrame)

    def test_incoherence_present(self):
        """Incoherent data should differ from coherent data at aggregate level."""
        lc_coherent, _ = simulate_uk_home()
        lc_incoherent, _ = simulate_incoherent_uk_home()
        # Aggregate columns should differ
        assert not lc_coherent["Buildings"].equals(lc_incoherent["Buildings"])

    def test_bottom_level_unchanged(self):
        """Bottom-level perils should be the same in both versions."""
        lc_coherent, _ = simulate_uk_home()
        lc_incoherent, _ = simulate_incoherent_uk_home()
        pd.testing.assert_series_equal(
            lc_coherent["Fire"], lc_incoherent["Fire"]
        )


class TestMakeHierarchyDataframe:
    def test_long_format(self):
        lc_df, ep_df = simulate_uk_home(n_periods=6)
        df = make_hierarchy_dataframe(lc_df, ep_df)
        assert "unique_id" in df.columns
        assert "ds" in df.columns
        assert "loss_cost" in df.columns
        assert "earned_premium" in df.columns

    def test_all_series_present(self):
        lc_df, ep_df = simulate_uk_home(n_periods=6)
        df = make_hierarchy_dataframe(lc_df, ep_df)
        assert set(lc_df.columns) == set(df["unique_id"].unique())

    def test_row_count(self):
        lc_df, ep_df = simulate_uk_home(n_periods=6)
        df = make_hierarchy_dataframe(lc_df, ep_df)
        assert len(df) == 6 * len(lc_df.columns)

    def test_model_column(self):
        lc_df, ep_df = simulate_uk_home(n_periods=6)
        df = make_hierarchy_dataframe(lc_df, ep_df, model_name="TestGLM")
        assert "model" in df.columns
        assert (df["model"] == "TestGLM").all()
