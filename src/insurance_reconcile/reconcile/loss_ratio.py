"""
LossRatioReconciler: reconcile loss ratios via claims amounts.

Loss ratios (or loss costs) are rates. Standard MinTrace reconciles absolute
quantities assuming additive constraints. Loss costs violate this:

    LC_total != LC_fire + LC_EoW + ...

The correct constraint is:
    LC_total = sum(LC_i * EP_i) / sum(EP_i)    (premium-weighted average)

The solution is to transform before reconciling:
    1. Convert: claims_i = LC_i * EP_i
    2. Reconcile: claims_reconciled = MinTrace(claims)
       [claims satisfy additive constraints: claims_total = sum(claims_i)]
    3. Invert: LC_reconciled_i = claims_reconciled_i / EP_i

This module implements that transform. It is independent of the specific
reconciler used — you can pass any reconciler that operates on amounts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["LossRatioReconciler"]


class LossRatioReconciler:
    """
    Reconcile loss ratios (or loss costs) by transforming to claims amounts.

    Loss costs are rates, not amounts. MinTrace reconciles additive quantities.
    This class handles the conversion so the caller can think in loss costs
    throughout.

    Parameters
    ----------
    earned_premium:
        Earned premium, as:
        - pd.Series with index = series names (for long/wide DF workflows)
        - dict {series_name: float} for a single period
        - np.ndarray aligned to S_df row order (for multi-period, shape n_series)
        - pd.DataFrame (periods x series) for period-varying EP
    reconciler:
        A reconciler that operates on amounts. Must have a reconcile() method
        accepting (y_hat, S_df, ...) and returning reconciled values.
        If None, uses PremiumWeightedMinTrace.

    Example
    -------
    >>> ep = {'Fire': 5e6, 'EoW': 3e6, 'Buildings': 8e6}
    >>> reconciler = LossRatioReconciler(earned_premium=ep)
    >>> lc_hat = np.array([0.08, 0.12, 0.095])   # loss costs
    >>> lc_reconciled = reconciler.reconcile(lc_hat, S, series_names=names)
    """

    def __init__(
        self,
        earned_premium: pd.Series | dict | np.ndarray | pd.DataFrame,
        reconciler: Any = None,
        nonnegative: bool = True,
    ) -> None:
        self._earned_premium = earned_premium
        self._reconciler = reconciler
        self._nonnegative = nonnegative

    def _get_ep_array(
        self,
        series_names: list[str],
        period: Any = None,
    ) -> np.ndarray:
        """Resolve earned premium to a 1D array aligned to series_names."""
        ep = self._earned_premium

        if isinstance(ep, np.ndarray):
            if ep.ndim == 1:
                if len(ep) != len(series_names):
                    raise ValueError(
                        f"earned_premium array length {len(ep)} != {len(series_names)}"
                    )
                return ep.astype(float)
            # 2D: (periods, series) — select row by period
            raise NotImplementedError(
                "Pass earned_premium as pd.DataFrame for period-varying EP"
            )

        if isinstance(ep, dict):
            return np.array([float(ep.get(s, 1.0)) for s in series_names])

        if isinstance(ep, pd.Series):
            return np.array([float(ep.get(s, 1.0)) for s in series_names])

        if isinstance(ep, pd.DataFrame):
            if period is None:
                # Use mean across periods
                return np.array(
                    [float(ep[s].mean()) if s in ep.columns else 1.0 for s in series_names]
                )
            row = ep.loc[period] if period in ep.index else ep.mean()
            return np.array([float(row.get(s, 1.0)) for s in series_names])

        raise TypeError(f"Unexpected earned_premium type: {type(ep)}")

    def _get_default_reconciler(self, ep_array: np.ndarray, series_names: list[str]):
        """Build a PremiumWeightedMinTrace reconciler as default."""
        from .premium_wls import PremiumWeightedMinTrace

        ep_dict = dict(zip(series_names, ep_array))
        return PremiumWeightedMinTrace(
            earned_premium=ep_dict,
            nonnegative=self._nonnegative,
        )

    def reconcile(
        self,
        lc_hat: np.ndarray,
        S: np.ndarray,
        series_names: list[str] | None = None,
        period: Any = None,
    ) -> np.ndarray:
        """
        Reconcile loss cost / loss ratio forecasts.

        Steps:
            1. Convert LC -> claims (multiply by EP)
            2. Reconcile claims with additive MinTrace
            3. Convert claims back to LC (divide by EP)

        Parameters
        ----------
        lc_hat:
            Loss cost (or loss ratio) forecasts, shape (n_series,) or
            (n_series, n_periods).
        S:
            Summing matrix, shape (n_series, n_bottom).
        series_names:
            Series names for EP lookup.
        period:
            Period label for period-varying EP lookup.

        Returns
        -------
        Reconciled loss costs, same shape as lc_hat.
        """
        lc_1d = lc_hat.ndim == 1
        if lc_1d:
            lc_hat = lc_hat.reshape(-1, 1)

        n_series = lc_hat.shape[0]
        if series_names is None:
            series_names = [str(i) for i in range(n_series)]

        ep_array = self._get_ep_array(series_names, period=period)
        # Shape: (n_series, 1) for broadcasting
        ep_col = ep_array.reshape(-1, 1)

        # Step 1: LC -> claims
        claims_hat = lc_hat * ep_col

        # Step 2: reconcile claims
        reconciler = self._reconciler
        if reconciler is None:
            reconciler = self._get_default_reconciler(ep_array, series_names)

        claims_reconciled = reconciler.reconcile(
            claims_hat, S, series_names=series_names
        )

        if claims_reconciled.ndim == 1:
            claims_reconciled = claims_reconciled.reshape(-1, 1)

        # Step 3: claims -> LC
        lc_reconciled = np.where(
            ep_col > 0,
            claims_reconciled / ep_col,
            0.0,
        )

        if self._nonnegative:
            lc_reconciled = np.maximum(lc_reconciled, 0.0)

        return lc_reconciled[:, 0] if lc_1d else lc_reconciled

    def reconcile_dataframe(
        self,
        lc_df: pd.DataFrame,
        S_df: pd.DataFrame,
        ep_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Reconcile a wide DataFrame of loss costs.

        Parameters
        ----------
        lc_df:
            Wide DataFrame (periods x series) of loss cost forecasts.
        S_df:
            Summing matrix DataFrame.
        ep_df:
            Optional wide DataFrame of earned premiums. If None, uses
            earned_premium from __init__.

        Returns
        -------
        Reconciled loss cost DataFrame, same shape as lc_df.
        """
        series_names = list(lc_df.columns)
        lc_array = lc_df.values.T  # (n_series, n_periods)
        S_array = S_df.loc[series_names].values if set(series_names).issubset(S_df.index) else S_df.values

        if ep_df is not None:
            self._earned_premium = ep_df

        reconciled = self.reconcile(lc_array, S_array, series_names=series_names)
        return pd.DataFrame(reconciled.T, index=lc_df.index, columns=series_names)
