"""Tests for LossRatioReconciler."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile._compat import HAS_HIERARCHICALFORECAST
from insurance_reconcile.reconcile.loss_ratio import LossRatioReconciler
from insurance_reconcile.hierarchy.builder import build_S_df
from insurance_reconcile.hierarchy.spec import PerilTree


pytestmark = pytest.mark.skipif(
    not HAS_HIERARCHICALFORECAST,
    reason="LossRatioReconciler with default reconciler requires hierarchicalforecast",
)


def simple_setup():
    tree = PerilTree(
        portfolio="Home",
        covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
    )
    S_df, tags = build_S_df(tree, ["Fire", "EoW", "Theft"])
    S = S_df.values
    names = list(S_df.index)
    ep = {"Fire": 5e6, "EoW": 4e6, "Theft": 2e6,
          "Buildings": 9e6, "Contents": 2e6, "Home": 11e6}
    return S, names, ep, S_df


class TestLossRatioReconciler:
    def test_instantiation(self):
        ep = {"Fire": 5e6, "EoW": 4e6}
        rec = LossRatioReconciler(earned_premium=ep)
        assert rec is not None

    def test_reconcile_1d(self):
        S, names, ep, S_df = simple_setup()
        rec = LossRatioReconciler(earned_premium=ep)
        lc_hat = np.array([0.028, 0.032, 0.018, 0.029, 0.018, 0.026])
        result = rec.reconcile(lc_hat, S, series_names=names)
        assert result.shape == (6,)

    def test_reconcile_2d(self):
        S, names, ep, S_df = simple_setup()
        rec = LossRatioReconciler(earned_premium=ep)
        rng = np.random.default_rng(1)
        lc_hat = 0.025 + rng.normal(0, 0.002, (6, 12))
        lc_hat = np.abs(lc_hat)
        result = rec.reconcile(lc_hat, S, series_names=names)
        assert result.shape == (6, 12)

    def test_nonnegative_output(self):
        S, names, ep, S_df = simple_setup()
        rec = LossRatioReconciler(earned_premium=ep, nonnegative=True)
        lc_hat = np.array([0.028, 0.032, 0.018, 0.029, 0.018, 0.026])
        result = rec.reconcile(lc_hat, S, series_names=names)
        assert (result >= 0).all()

    def test_coherence_after_reconcile(self):
        """Reconciled loss costs should satisfy EP-weighted constraint."""
        S, names, ep, S_df = simple_setup()
        rec = LossRatioReconciler(earned_premium=ep, nonnegative=False)
        lc_hat = np.array([0.028, 0.032, 0.018, 0.029, 0.018, 0.026])
        result = rec.reconcile(lc_hat, S, series_names=names)

        idx = {n: i for i, n in enumerate(names)}
        ep_fire = ep["Fire"]
        ep_eow = ep["EoW"]
        ep_bld = ep["Buildings"]
        lc_fire_r = result[idx["Fire"]]
        lc_eow_r = result[idx["EoW"]]
        lc_bld_r = result[idx["Buildings"]]

        expected_bld = (lc_fire_r * ep_fire + lc_eow_r * ep_eow) / ep_bld
        np.testing.assert_allclose(lc_bld_r, expected_bld, rtol=1e-4)

    def test_ep_as_series(self):
        S, names, ep, S_df = simple_setup()
        ep_series = pd.Series(ep)
        rec = LossRatioReconciler(earned_premium=ep_series)
        lc_hat = np.array([0.028, 0.032, 0.018, 0.029, 0.018, 0.026])
        result = rec.reconcile(lc_hat, S, series_names=names)
        assert result.shape == (6,)

    def test_ep_as_array(self):
        S, names, ep, S_df = simple_setup()
        ep_arr = np.array([ep[n] for n in names])
        rec = LossRatioReconciler(earned_premium=ep_arr)
        lc_hat = np.array([0.028, 0.032, 0.018, 0.029, 0.018, 0.026])
        result = rec.reconcile(lc_hat, S, series_names=names)
        assert result.shape == (6,)

    def test_reconcile_dataframe(self):
        S, names, ep, S_df = simple_setup()
        rec = LossRatioReconciler(earned_premium=ep)
        n = 6
        idx = pd.date_range("2022-01-01", periods=n, freq="MS")
        lc_df = pd.DataFrame(
            np.tile([0.028, 0.032, 0.018, 0.029, 0.018, 0.026], (n, 1)),
            index=idx,
            columns=names,
        )
        result_df = rec.reconcile_dataframe(lc_df, S_df)
        assert result_df.shape == lc_df.shape
        assert list(result_df.columns) == names
