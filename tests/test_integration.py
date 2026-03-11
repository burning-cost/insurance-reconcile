"""
End-to-end integration tests.

These tests exercise the full workflow from hierarchy spec -> S_df ->
coherence check -> reconciliation -> diagnostics.

Tests that require hierarchicalforecast are conditionally skipped.
"""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile import (
    InsuranceHierarchy,
    InsuranceReconciler,
    HierarchyBuilder,
    LossCostFrame,
    check_coherence,
    HAS_HIERARCHICALFORECAST,
)
from insurance_reconcile.simulate import simulate_uk_home, simulate_incoherent_uk_home, make_hierarchy_dataframe
from insurance_reconcile.hierarchy.builder import build_S_df


class TestFullWorkflowCoherenceOnly:
    """Tests that work without hierarchicalforecast."""

    def test_simulate_and_check(self):
        lc_df, ep_df = simulate_uk_home(n_periods=12)
        tree = InsuranceHierarchy.uk_home().peril_tree
        s_df, tags = build_S_df(tree, tree.all_perils)

        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert report.is_coherent
        assert report.max_discrepancy_pct < 0.01

    def test_incoherent_detected(self):
        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=12)
        tree = InsuranceHierarchy.uk_home().peril_tree
        s_df, tags = build_S_df(tree, tree.all_perils)

        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
            tolerance_pct=0.01,
        )
        assert not report.is_coherent
        assert len(report.violations) > 0
        assert "Buildings" in report.worst_series() or "Contents" in report.worst_series() or "Home" in report.worst_series()

    def test_losscost_frame_integration(self):
        lc_df, ep_df = simulate_uk_home(n_periods=6)
        long_df = make_hierarchy_dataframe(lc_df, ep_df)
        frame = LossCostFrame(long_df)
        errors = frame.validate(raise_on_error=False)
        assert errors == []

    def test_hierarchy_builder_integration(self):
        h = InsuranceHierarchy.uk_home()
        builder = HierarchyBuilder(h)
        perils = h.peril_tree.all_perils
        S_df, tags = builder.build_S_df(perils)
        assert S_df.shape[0] == 10  # 7 perils + 2 covers + 1 portfolio
        assert S_df.shape[1] == 7
        assert "Portfolio" in tags
        assert "Cover" in tags
        assert "Peril" in tags

    def test_coherence_report_to_string_verbose(self):
        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=6)
        tree = InsuranceHierarchy.uk_home().peril_tree
        s_df, _ = build_S_df(tree, tree.all_perils)
        report = check_coherence(
            values_df=lc_df,
            s_df=s_df,
            earned_premium_df=ep_df,
        )
        text = report.to_string()
        assert len(text) > 50  # Non-trivial output
        df = report.to_dataframe()
        assert len(df) > 0

    def test_all_public_imports(self):
        from insurance_reconcile import (
            InsuranceHierarchy, PerilTree, GeographicHierarchy,
            HierarchyBuilder, build_S_df,
            InsuranceReconciler, ReconciliationResult,
            LossRatioReconciler, FreqSevReconciler,
            PremiumWeightedMinTrace, premium_wls_reconcile,
            LossCostFrame,
            CoherenceReport, check_coherence,
            AttributionReport,
            HAS_HIERARCHICALFORECAST,
        )
        assert True  # Import succeeded


@pytest.mark.skipif(
    not HAS_HIERARCHICALFORECAST,
    reason="Full reconciliation requires hierarchicalforecast",
)
class TestFullWorkflowWithReconciliation:
    def test_reconcile_improves_coherence(self):
        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=24)
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())

        before_report = rec.check_coherence(lc_df, ep_df)
        result = rec.reconcile(lc_df, ep_df)
        after_report = result.coherence_after

        assert not before_report.is_coherent
        # After reconciliation, max discrepancy should be much reduced
        assert after_report.max_discrepancy_pct < before_report.max_discrepancy_pct

    def test_reconcile_values_reasonable(self):
        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=12)
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df)
        # Loss costs should be in sensible range (0 to 1 for insurance)
        assert (result.reconciled_df >= 0).all().all()
        assert (result.reconciled_df < 1.0).all().all()

    def test_attribution_captures_adjustments(self):
        lc_df, ep_df = simulate_incoherent_uk_home(n_periods=12)
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df)
        attr = result.attribution
        assert attr is not None
        assert attr.n_series > 0
        # At least some series should have been adjusted
        total_adj = sum(abs(a.mean_adjustment) for a in attr.adjustments)
        assert total_adj > 0

    def test_freq_sev_workflow(self):
        """Test FreqSevReconciler end-to-end."""
        from insurance_reconcile.reconcile.freqsev import FreqSevReconciler
        tree = InsuranceHierarchy.uk_home().peril_tree
        s_df, _ = build_S_df(tree, tree.all_perils)
        S = s_df.values
        names = list(s_df.index)

        rng = np.random.default_rng(0)
        n_series = len(names)
        freq = rng.uniform(0.02, 0.1, n_series)
        sev = rng.uniform(200, 1000, n_series)

        rec = FreqSevReconciler()
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert (freq_r >= 0).all()
        assert (sev_r >= 0).all()
        assert freq_r.shape == (n_series,)
