"""
Tests for PremiumWeightedMinTrace.

These tests do NOT require hierarchicalforecast to be installed —
PremiumWeightedMinTrace._compat check only fires when the class is instantiated
if hierarchicalforecast is not available. But the mathematical reconciliation
logic (numpy only) can be tested directly.

We test the reconcile() method which only uses numpy linear algebra.
"""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile._compat import HAS_HIERARCHICALFORECAST


pytestmark = pytest.mark.skipif(
    not HAS_HIERARCHICALFORECAST,
    reason="PremiumWeightedMinTrace requires hierarchicalforecast",
)


from insurance_reconcile.reconcile.premium_wls import (
    PremiumWeightedMinTrace,
    premium_wls_reconcile,
)
from insurance_reconcile.hierarchy.builder import build_S_df
from insurance_reconcile.hierarchy.spec import PerilTree


def simple_tree():
    return PerilTree(
        portfolio="Home",
        covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
    )


def simple_S():
    tree = simple_tree()
    S_df, tags = build_S_df(tree, ["Fire", "EoW", "Theft"])
    return S_df.values, list(S_df.index)


class TestPremiumWeightedMinTrace:
    def test_instantiation_with_dict(self):
        ep = {"Fire": 5e6, "EoW": 3e6, "Theft": 2e6}
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        assert rec is not None

    def test_instantiation_with_array(self):
        ep = np.array([5e6, 3e6, 2e6, 8e6, 2e6, 10e6])
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        assert rec is not None

    def test_reconcile_1d(self):
        S, names = simple_S()
        ep = {"Fire": 5e6, "EoW": 3e6, "Theft": 2e6,
              "Buildings": 8e6, "Contents": 2e6, "Home": 10e6}
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        y_hat = np.array([0.028, 0.020, 0.032, 0.018, 0.025, 0.024])
        result = rec.reconcile(y_hat, S, series_names=names)
        assert result.shape == (6,)

    def test_reconcile_2d(self):
        S, names = simple_S()
        ep = {"Fire": 5e6, "EoW": 3e6, "Theft": 2e6,
              "Buildings": 8e6, "Contents": 2e6, "Home": 10e6}
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        y_hat = np.random.default_rng(0).uniform(0.01, 0.05, (6, 12))
        result = rec.reconcile(y_hat, S, series_names=names)
        assert result.shape == (6, 12)

    def test_nonnegative_enforced(self):
        S, names = simple_S()
        ep = {n: 1e6 for n in names}
        rec = PremiumWeightedMinTrace(earned_premium=ep, nonnegative=True)
        # Pass negative base forecasts
        y_hat = np.array([-0.01, -0.02, 0.03, -0.005, 0.02, -0.01])
        result = rec.reconcile(y_hat, S, series_names=names)
        assert (result >= 0).all()

    def test_result_is_coherent(self):
        """
        Reconciled forecasts should satisfy S @ G @ y_hat = result,
        meaning bottom-level sums satisfy hierarchy.
        We check: result[portfolio] approx sum(result[bottom via S]).
        """
        S, names = simple_S()
        ep_vals = np.array([5e6, 3e6, 2e6, 8e6, 2e6, 10e6])
        ep = dict(zip(names, ep_vals))
        rec = PremiumWeightedMinTrace(earned_premium=ep, nonnegative=False)
        y_hat = np.array([0.028, 0.020, 0.032, 0.018, 0.025, 0.024])
        result = rec.reconcile(y_hat, S, series_names=names)
        # Recover bottom series reconciled values
        idx = {n: i for i, n in enumerate(names)}
        fire_r = result[idx["Fire"]]
        eow_r = result[idx["EoW"]]
        theft_r = result[idx["Theft"]]
        bld_r = result[idx["Buildings"]]
        home_r = result[idx["Home"]]
        # Additive coherence: Buildings = Fire + EoW (in terms of row sums from S)
        np.testing.assert_allclose(bld_r, fire_r + eow_r, rtol=1e-4)
        np.testing.assert_allclose(home_r, fire_r + eow_r + theft_r, rtol=1e-4)

    def test_missing_series_uses_fallback_weight(self):
        S, names = simple_S()
        ep = {"Fire": 5e6}  # Only Fire — others get fallback 1.0
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        y_hat = np.array([0.028, 0.020, 0.032, 0.018, 0.025, 0.024])
        result = rec.reconcile(y_hat, S, series_names=names)
        assert result.shape == (6,)

    def test_array_ep_wrong_length_raises(self):
        S, names = simple_S()
        ep = np.array([1.0, 2.0])  # Wrong length
        rec = PremiumWeightedMinTrace(earned_premium=ep)
        y_hat = np.array([0.028, 0.020, 0.032, 0.018, 0.025, 0.024])
        with pytest.raises(ValueError, match="does not match n_series"):
            rec.reconcile(y_hat, S, series_names=names)

    def test_convenience_function(self):
        S, names = simple_S()
        ep = {n: 1e6 for n in names}
        y_hat = np.array([0.028, 0.020, 0.032, 0.018, 0.025, 0.024])
        result = premium_wls_reconcile(y_hat, S, earned_premium=ep, series_names=names)
        assert result.shape == (6,)
