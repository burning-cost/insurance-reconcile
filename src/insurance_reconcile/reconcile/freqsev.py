"""
FreqSevReconciler: reconcile frequency and severity via log-space.

Loss cost = frequency * severity (multiplicative constraint). This is non-linear
and standard MinTrace cannot directly reconcile (freq, sev) pairs because the
constraints involve products, not sums.

The practical approach: reconcile log(LC) where
    log(LC) = log(freq) + log(sev)
This is an additive constraint in log space. After reconciliation, back-transform
to get coherent (freq, sev) pairs.

This is an approximation. The true constraint is:
    LC_total = sum(LC_i * EP_i) / sum(EP_i)
              = sum(freq_i * sev_i * EP_i) / sum(EP_i)

which is non-linear in (freq_i, sev_i). The log-space approach is widely used
in actuarial practice because it:
1. Respects non-negativity automatically (exp is always positive)
2. Handles the multiplicative structure correctly in isolation
3. Works well when freq and sev are relatively stable across hierarchy levels

Limitations: the log-space approach does not guarantee that the back-transformed
(freq, sev) pairs satisfy the exact premium-weighted constraint at the portfolio
level. For exact nonlinear reconciliation, see arxiv:2510.21249 (no Python
implementation exists as of 2026-03).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["FreqSevReconciler"]


class FreqSevReconciler:
    """
    Reconcile frequency/severity pairs via log-space additive reconciliation.

    The reconciliation operates on log(loss_cost) = log(freq) + log(sev),
    which makes the multiplicative constraint additive. This allows standard
    MinTrace to be applied.

    Parameters
    ----------
    earned_premium:
        Earned premium for weighting. Same formats as LossRatioReconciler.
    freq_reconciler:
        Reconciler for log-frequency. If None, uses OLS MinTrace (flat weights
        in log space).
    sev_reconciler:
        Reconciler for log-severity. Same.
    nonnegative:
        Enforce non-negativity on back-transformed values (default True).
    log_floor:
        Minimum value before log transform to avoid log(0). Default 1e-8.

    Example
    -------
    >>> reconciler = FreqSevReconciler(earned_premium=ep_dict)
    >>> freq_rec, sev_rec = reconciler.reconcile(freq_hat, sev_hat, S)
    """

    def __init__(
        self,
        earned_premium: Any = None,
        freq_reconciler: Any = None,
        sev_reconciler: Any = None,
        nonnegative: bool = True,
        log_floor: float = 1e-8,
    ) -> None:
        self._earned_premium = earned_premium
        self._freq_reconciler = freq_reconciler
        self._sev_reconciler = sev_reconciler
        self._nonnegative = nonnegative
        self._log_floor = log_floor

    def _build_default_reconciler(
        self,
        series_names: list[str],
        ep_array: np.ndarray | None = None,
    ):
        """Build a simple OLS reconciler for log-space (no premium weighting needed)."""
        from .premium_wls import PremiumWeightedMinTrace

        if ep_array is not None:
            ep_dict = dict(zip(series_names, ep_array))
        else:
            ep_dict = {s: 1.0 for s in series_names}

        return PremiumWeightedMinTrace(earned_premium=ep_dict, nonnegative=False)

    def _get_ep_array(self, series_names: list[str]) -> np.ndarray | None:
        """Resolve earned premium to array, or None if not provided."""
        if self._earned_premium is None:
            return None

        ep = self._earned_premium
        if isinstance(ep, dict):
            return np.array([float(ep.get(s, 1.0)) for s in series_names])
        if isinstance(ep, pd.Series):
            return np.array([float(ep.get(s, 1.0)) for s in series_names])
        if isinstance(ep, np.ndarray):
            return ep.astype(float)
        return None

    def reconcile(
        self,
        freq_hat: np.ndarray,
        sev_hat: np.ndarray,
        S: np.ndarray,
        series_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconcile frequency and severity arrays.

        Both arrays are log-transformed, reconciled separately using additive
        MinTrace, then back-transformed via exp().

        Parameters
        ----------
        freq_hat:
            Frequency forecasts, shape (n_series,) or (n_series, n_periods).
            Must be positive.
        sev_hat:
            Severity forecasts, same shape. Must be positive.
        S:
            Summing matrix, shape (n_series, n_bottom).
        series_names:
            Optional series names for EP lookup.

        Returns
        -------
        freq_reconciled, sev_reconciled
            Reconciled frequencies and severities, same shape as inputs.
        """
        freq_1d = freq_hat.ndim == 1
        if freq_1d:
            freq_hat = freq_hat.reshape(-1, 1)
            sev_hat = sev_hat.reshape(-1, 1)

        n_series = freq_hat.shape[0]
        if series_names is None:
            series_names = [str(i) for i in range(n_series)]

        ep_array = self._get_ep_array(series_names)

        # Log transform (floor to avoid log(0))
        log_freq = np.log(np.maximum(freq_hat, self._log_floor))
        log_sev = np.log(np.maximum(sev_hat, self._log_floor))

        # Build reconcilers if not provided
        freq_rec = self._freq_reconciler or self._build_default_reconciler(
            series_names, ep_array
        )
        sev_rec = self._sev_reconciler or self._build_default_reconciler(
            series_names, ep_array
        )

        # Reconcile in log space
        log_freq_rec = freq_rec.reconcile(log_freq, S, series_names=series_names)
        log_sev_rec = sev_rec.reconcile(log_sev, S, series_names=series_names)

        if log_freq_rec.ndim == 1:
            log_freq_rec = log_freq_rec.reshape(-1, 1)
        if log_sev_rec.ndim == 1:
            log_sev_rec = log_sev_rec.reshape(-1, 1)

        # Back-transform
        freq_reconciled = np.exp(log_freq_rec)
        sev_reconciled = np.exp(log_sev_rec)

        if self._nonnegative:
            freq_reconciled = np.maximum(freq_reconciled, 0.0)
            sev_reconciled = np.maximum(sev_reconciled, 0.0)

        if freq_1d:
            return freq_reconciled[:, 0], sev_reconciled[:, 0]
        return freq_reconciled, sev_reconciled

    def reconcile_loss_cost(
        self,
        freq_hat: np.ndarray,
        sev_hat: np.ndarray,
        S: np.ndarray,
        series_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Reconcile and return loss cost = freq * sev.

        Parameters
        ----------
        freq_hat, sev_hat, S, series_names:
            Same as reconcile().

        Returns
        -------
        Loss cost array = reconciled_freq * reconciled_sev.
        """
        freq_rec, sev_rec = self.reconcile(freq_hat, sev_hat, S, series_names)
        return freq_rec * sev_rec
