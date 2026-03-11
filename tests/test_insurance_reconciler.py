"""Tests for InsuranceReconciler (main API)."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile._compat import HAS_HIERARCHICALFORECAST
from insurance_reconcile.reconcile.wrapper import InsuranceReconciler, ReconciliationResult
from insurance_reconcile.hierarchy.spec import InsuranceHierarchy
from insurance_reconcile.simulate import simulate_uk_home, simulate_incoherent_uk_home


def make_test_data(n=12):
    return simulate_uk_home(n_periods=n)


def make_incoherent_test_data(n=12):
    return simulate_incoherent_uk_home(n_periods=n)


class TestInsuranceReconcilerCoherenceCheck:
    """coherence check works without hierarchicalforecast."""

    def test_check_coherence_coherent(self):
        lc_df, ep_df = make_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())
        report = rec.check_coherence(lc_df, ep_df)
        assert report.is_coherent

    def test_check_coherence_incoherent(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())
        report = rec.check_coherence(lc_df, ep_df)
        assert not report.is_coherent

    def test_check_coherence_returns_report(self):
        lc_df, ep_df = make_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        report = rec.check_coherence(lc_df, ep_df)
        from insurance_reconcile.diagnostics.coherence import CoherenceReport
        assert isinstance(report, CoherenceReport)

    def test_check_coherence_no_ep(self):
        """Without EP, should fall back to additive check."""
        lc_df, ep_df = make_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        # Without ep_df, is_loss_cost=False — additive check
        report = rec.check_coherence(lc_df, earned_premium_df=None)
        # Coherent data is not additively coherent (it's EP-weighted)
        # so we just check the call doesn't raise
        assert report is not None

    def test_bottom_series_inference(self):
        lc_df, ep_df = make_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        # Should infer bottom series from hierarchy
        report = rec.check_coherence(lc_df, ep_df)
        assert report.n_series_checked > 0


@pytest.mark.skipif(
    not HAS_HIERARCHICALFORECAST,
    reason="InsuranceReconciler.reconcile() requires hierarchicalforecast",
)
class TestInsuranceReconcilerReconcile:
    def test_reconcile_returns_result(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())
        result = rec.reconcile(lc_df, ep_df)
        assert isinstance(result, ReconciliationResult)

    def test_reconciled_df_shape(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())
        result = rec.reconcile(lc_df, ep_df)
        assert result.reconciled_df.shape == lc_df.shape

    def test_reconcile_improves_coherence(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, earned_premium=ep_df.mean())
        result = rec.reconcile(lc_df, ep_df)
        # After should have lower or equal discrepancy than before
        if result.coherence_before and result.coherence_after:
            assert (
                result.coherence_after.max_discrepancy_pct
                <= result.coherence_before.max_discrepancy_pct + 0.1
            )

    def test_reconcile_nonnegative(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h, nonnegative=True)
        result = rec.reconcile(lc_df, ep_df)
        assert (result.reconciled_df >= 0).all().all()

    def test_reconcile_with_diagnostics(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df, run_diagnostics=True)
        assert result.coherence_before is not None
        assert result.coherence_after is not None
        assert result.attribution is not None

    def test_reconcile_without_diagnostics(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df, run_diagnostics=False)
        assert result.coherence_before is None
        assert result.coherence_after is None
        assert result.attribution is None

    def test_reconcile_no_ep(self):
        """Reconciliation without EP uses direct MinTrace."""
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, earned_premium_df=None)
        assert result.reconciled_df is not None

    def test_reconcile_result_repr(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df)
        r = repr(result)
        assert "ReconciliationResult" in r

    def test_reconcile_method_name(self):
        lc_df, ep_df = make_incoherent_test_data()
        h = InsuranceHierarchy.uk_home()
        rec = InsuranceReconciler(h)
        result = rec.reconcile(lc_df, ep_df)
        assert result.method in ("PremiumWeightedMinTrace", "LossRatioReconciler")


class TestReconciliationResult:
    def test_repr_minimal(self):
        lc_df, _ = make_test_data(n=6)
        result = ReconciliationResult(reconciled_df=lc_df, method="test")
        r = repr(result)
        assert "ReconciliationResult" in r
