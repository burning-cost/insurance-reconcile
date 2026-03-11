"""Tests for FreqSevReconciler."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile._compat import HAS_HIERARCHICALFORECAST
from insurance_reconcile.reconcile.freqsev import FreqSevReconciler
from insurance_reconcile.hierarchy.builder import build_S_df
from insurance_reconcile.hierarchy.spec import PerilTree


pytestmark = pytest.mark.skipif(
    not HAS_HIERARCHICALFORECAST,
    reason="FreqSevReconciler default reconciler requires hierarchicalforecast",
)


def simple_S():
    tree = PerilTree(
        portfolio="Home",
        covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
    )
    S_df, tags = build_S_df(tree, ["Fire", "EoW", "Theft"])
    return S_df.values, list(S_df.index)


class TestFreqSevReconciler:
    def test_instantiation(self):
        rec = FreqSevReconciler()
        assert rec is not None

    def test_reconcile_1d(self):
        S, names = simple_S()
        rec = FreqSevReconciler()
        freq = np.array([0.05, 0.08, 0.03, 0.065, 0.03, 0.058])
        sev = np.array([500, 400, 600, 450, 600, 462])
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert freq_r.shape == (6,)
        assert sev_r.shape == (6,)

    def test_reconcile_2d(self):
        S, names = simple_S()
        rec = FreqSevReconciler()
        rng = np.random.default_rng(0)
        freq = rng.uniform(0.01, 0.1, (6, 12))
        sev = rng.uniform(200, 800, (6, 12))
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert freq_r.shape == (6, 12)
        assert sev_r.shape == (6, 12)

    def test_output_nonnegative(self):
        S, names = simple_S()
        rec = FreqSevReconciler(nonnegative=True)
        freq = np.array([0.05, 0.08, 0.03, 0.065, 0.03, 0.058])
        sev = np.array([500, 400, 600, 450, 600, 462])
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert (freq_r >= 0).all()
        assert (sev_r >= 0).all()

    def test_reconcile_loss_cost(self):
        S, names = simple_S()
        rec = FreqSevReconciler()
        freq = np.array([0.05, 0.08, 0.03, 0.065, 0.03, 0.058])
        sev = np.array([500, 400, 600, 450, 600, 462])
        lc = rec.reconcile_loss_cost(freq, sev, S, series_names=names)
        assert lc.shape == (6,)
        # Loss cost should be freq * sev
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        np.testing.assert_allclose(lc, freq_r * sev_r, rtol=1e-10)

    def test_log_floor_prevents_log_zero(self):
        S, names = simple_S()
        rec = FreqSevReconciler(log_floor=1e-8)
        freq = np.array([0.0, 0.08, 0.03, 0.065, 0.03, 0.058])
        sev = np.array([500, 400, 600, 450, 600, 462])
        # Should not raise, freq[0]=0 is handled by log_floor
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert np.isfinite(freq_r).all()
        assert np.isfinite(sev_r).all()

    def test_with_ep_weighting(self):
        S, names = simple_S()
        ep = {n: 1e6 for n in names}
        rec = FreqSevReconciler(earned_premium=ep)
        freq = np.array([0.05, 0.08, 0.03, 0.065, 0.03, 0.058])
        sev = np.array([500, 400, 600, 450, 600, 462])
        freq_r, sev_r = rec.reconcile(freq, sev, S, series_names=names)
        assert freq_r.shape == (6,)
