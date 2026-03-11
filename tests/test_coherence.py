"""Tests for coherence diagnostics."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.hierarchy.builder import build_S_df
from insurance_reconcile.hierarchy.spec import PerilTree
from insurance_reconcile.diagnostics.coherence import (
    check_coherence,
    CoherenceReport,
    CoherenceViolation,
)
from insurance_reconcile.simulate import simulate_uk_home, simulate_incoherent_uk_home


def make_simple_hierarchy():
    """3-peril, 2-cover hierarchy for testing."""
    tree = PerilTree(
        portfolio="Home",
        covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
    )
    return tree


def make_coherent_data(n_periods=6):
    """Build a coherent wide DataFrame for testing."""
    tree = make_simple_hierarchy()
    s_df, tags = build_S_df(tree, ["Fire", "EoW", "Theft"])

    rng = np.random.default_rng(42)
    # Define peril EPs
    ep = {"Fire": 5e6, "EoW": 4e6, "Theft": 2e6}
    # Compute cover EPs
    ep["Buildings"] = ep["Fire"] + ep["EoW"]
    ep["Home"] = sum(ep[k] for k in ["Fire", "EoW", "Theft"])
    ep["Contents"] = ep["Theft"]

    # Generate peril loss costs
    lc_fire = 0.025 + rng.normal(0, 0.001, n_periods)
    lc_eow = 0.032 + rng.normal(0, 0.001, n_periods)
    lc_theft = 0.018 + rng.normal(0, 0.001, n_periods)

    # Compute coherent aggregates
    lc_bld = (lc_fire * ep["Fire"] + lc_eow * ep["EoW"]) / ep["Buildings"]
    lc_cont = lc_theft  # Only one peril
    lc_home = (lc_bld * ep["Buildings"] + lc_cont * ep["Contents"]) / ep["Home"]

    idx = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    lc_df = pd.DataFrame({
        "Fire": lc_fire, "EoW": lc_eow, "Theft": lc_theft,
        "Buildings": lc_bld, "Contents": lc_cont, "Home": lc_home,
    }, index=idx)

    ep_df = pd.DataFrame(
        {col: np.full(n_periods, val) for col, val in ep.items()},
        index=idx,
    )
    return lc_df, ep_df, s_df


class TestCoherenceViolation:
    def test_str_representation(self):
        v = CoherenceViolation(
            series_name="Buildings",
            ds=pd.Timestamp("2022-01-01"),
            actual_value=0.028,
            expected_value=0.025,
            discrepancy_abs=0.003,
            discrepancy_pct=12.0,
            children=["Fire", "EoW"],
        )
        s = str(v)
        assert "Buildings" in s
        assert "12.00" in s


class TestCheckCoherence:
    def test_coherent_data_passes(self):
        lc_df, ep_df, s_df = make_coherent_data()
        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.1,
        )
        assert report.is_coherent
        assert report.max_discrepancy_pct < 0.1

    def test_incoherent_data_fails(self):
        lc_df, ep_df, s_df = make_coherent_data()
        # Break coherence by perturbing aggregate
        lc_bad = lc_df.copy()
        lc_bad["Buildings"] = lc_bad["Buildings"] * 1.10  # 10% off
        report = check_coherence(
            values_df=lc_bad,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert not report.is_coherent
        assert report.max_discrepancy_pct > 5.0

    def test_violations_list_populated(self):
        lc_df, ep_df, s_df = make_coherent_data()
        lc_bad = lc_df.copy()
        lc_bad["Home"] = lc_bad["Home"] * 1.05
        report = check_coherence(
            values_df=lc_bad,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert len(report.violations) > 0

    def test_to_dataframe(self):
        lc_df, ep_df, s_df = make_coherent_data()
        lc_bad = lc_df.copy()
        lc_bad["Buildings"] = lc_bad["Buildings"] * 1.1
        report = check_coherence(
            values_df=lc_bad,
            s_df=s_df,
            earned_premium_df=ep_df,
        )
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_to_string(self):
        lc_df, ep_df, s_df = make_coherent_data()
        report = check_coherence(
            values_df=lc_df, s_df=s_df, earned_premium_df=ep_df
        )
        s = report.to_string()
        assert "Coherence Report" in s
        assert "COHERENT" in s

    def test_incoherent_to_string(self):
        lc_df, ep_df, s_df = make_coherent_data()
        lc_bad = lc_df.copy()
        lc_bad["Buildings"] *= 1.1
        report = check_coherence(
            values_df=lc_bad, s_df=s_df, earned_premium_df=ep_df
        )
        s = report.to_string()
        assert "INCOHERENT" in s

    def test_additive_coherence(self):
        """Test additive (claims amounts) coherence check."""
        tree = make_simple_hierarchy()
        s_df, _ = build_S_df(tree, ["Fire", "EoW", "Theft"])
        rng = np.random.default_rng(0)
        n = 4
        idx = pd.date_range("2022-01-01", periods=n, freq="MS")
        fire = rng.uniform(100, 200, n)
        eow = rng.uniform(80, 180, n)
        theft = rng.uniform(50, 100, n)
        bld = fire + eow
        cont = theft
        home = fire + eow + theft
        df = pd.DataFrame(
            {"Fire": fire, "EoW": eow, "Theft": theft,
             "Buildings": bld, "Contents": cont, "Home": home},
            index=idx,
        )
        report = check_coherence(
            values_df=df,
            s_df=s_df,
            earned_premium_df=None,
            is_loss_cost=False,
            tolerance_pct=0.01,
        )
        assert report.is_coherent

    def test_requires_ep_for_loss_cost(self):
        lc_df, ep_df, s_df = make_coherent_data()
        with pytest.raises(ValueError, match="earned_premium_df is required"):
            check_coherence(
                values_df=lc_df,
                s_df=s_df,
                earned_premium_df=None,
                is_loss_cost=True,
            )

    def test_worst_series(self):
        lc_df, ep_df, s_df = make_coherent_data()
        lc_bad = lc_df.copy()
        lc_bad["Buildings"] *= 1.15
        lc_bad["Home"] *= 1.05
        report = check_coherence(
            values_df=lc_bad, s_df=s_df, earned_premium_df=ep_df
        )
        worst = report.worst_series(n=2)
        assert len(worst) <= 2

    def test_repr(self):
        lc_df, ep_df, s_df = make_coherent_data()
        report = check_coherence(
            values_df=lc_df, s_df=s_df, earned_premium_df=ep_df
        )
        r = repr(report)
        assert "CoherenceReport" in r

    def test_n_series_checked(self):
        lc_df, ep_df, s_df = make_coherent_data()
        report = check_coherence(
            values_df=lc_df, s_df=s_df, earned_premium_df=ep_df
        )
        # Aggregate series: Home, Buildings, Contents = 3
        assert report.n_series_checked == 3

    def test_simulate_coherent(self):
        """Simulated coherent data should pass coherence check."""
        from insurance_reconcile.hierarchy.builder import build_S_df
        from insurance_reconcile.hierarchy.spec import PerilTree

        lc_df, ep_df = simulate_uk_home(n_periods=6)
        tree = PerilTree.uk_home()
        s_df, _ = build_S_df(tree, tree.all_perils)
        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert report.is_coherent

    def test_simulate_incoherent(self):
        """Simulated incoherent data should fail coherence check."""
        from insurance_reconcile.hierarchy.builder import build_S_df
        from insurance_reconcile.hierarchy.spec import PerilTree

        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=6)
        tree = PerilTree.uk_home()
        s_df, _ = build_S_df(tree, tree.all_perils)
        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert not report.is_coherent
