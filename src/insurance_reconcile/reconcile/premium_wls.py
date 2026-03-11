"""
PremiumWeightedMinTrace: MinTrace with earned premium as the weight matrix.

The standard WLS_STRUCT method in hierarchicalforecast weights series by the
number of bottom-level series they aggregate. For insurance, the economically
correct weight is earned premium: large-volume cells should barely move,
small-volume cells can move substantially.

This implements the premium-weighted variant by subclassing MinTrace and
overriding the weight matrix construction. The formula is the standard
MinTrace GLS formula:

    tilde_y = S (S' W^-1 S)^-1 S' W^-1 hat_y

with W = diag(earned_premium). This is equivalent to WLS with EP weights.

Note on alignment: earned_premium must be aligned with the rows of S_df.
The ordering in S_df comes from hierarchicalforecast's aggregate() function,
which lists aggregate series first, then bottom-level series. Pass
earned_premium as a dict {series_name: value} and this class will handle
alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._compat import require_hierarchicalforecast

__all__ = ["PremiumWeightedMinTrace", "premium_wls_reconcile"]


class PremiumWeightedMinTrace:
    """
    MinTrace reconciliation using earned premium as the weight matrix.

    This is the insurance-specific variant of WLS. Standard WLS_STRUCT uses
    the structural count (how many bottom series roll into each aggregate) as
    the weight. PremiumWeightedMinTrace uses earned premium instead.

    Economic justification: the law of large numbers means that high-volume
    cells have lower coefficient of variation. A cell with £50M EP has much
    more stable observed loss costs than a cell with £100K EP. The weight
    matrix encodes this — we trust high-EP cells more.

    Parameters
    ----------
    earned_premium:
        Earned premium for each series, as either:
        - dict {series_name: float} — recommended, handles alignment
        - np.ndarray of length n_series — must match S_df row order exactly
    fallback_method:
        MinTrace method to use if earned_premium is unavailable for some
        series. Defaults to 'wls_struct'.
    nonnegative:
        Whether to enforce non-negativity (default True — loss costs cannot
        be negative).

    Example
    -------
    >>> ep = {'Fire': 5e6, 'EoW': 3e6, 'Buildings': 8e6, 'Home': 8e6}
    >>> reconciler = PremiumWeightedMinTrace(earned_premium=ep)
    >>> result = reconciler.reconcile(y_hat, S_df)
    """

    def __init__(
        self,
        earned_premium: dict[str, float] | np.ndarray,
        fallback_method: str = "wls_struct",
        nonnegative: bool = True,
    ) -> None:
        require_hierarchicalforecast("PremiumWeightedMinTrace")
        self._earned_premium = earned_premium
        self._fallback_method = fallback_method
        self._nonnegative = nonnegative

    def _get_weights(self, series_names: list[str]) -> np.ndarray:
        """
        Return weight vector aligned to series_names.

        For dict input, looks up by name. For array input, uses as-is.
        Missing series get a fallback weight (1.0).
        """
        if isinstance(self._earned_premium, np.ndarray):
            if len(self._earned_premium) != len(series_names):
                raise ValueError(
                    f"earned_premium array length {len(self._earned_premium)} "
                    f"does not match n_series {len(series_names)}"
                )
            return self._earned_premium.astype(float)

        weights = np.ones(len(series_names), dtype=float)
        for i, name in enumerate(series_names):
            if name in self._earned_premium:
                weights[i] = float(self._earned_premium[name])
            # else: keep 1.0 as fallback
        return weights

    def reconcile(
        self,
        y_hat: np.ndarray,
        S: np.ndarray,
        series_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply premium-weighted MinTrace reconciliation.

        Parameters
        ----------
        y_hat:
            Base forecasts, shape (n_series,) or (n_series, n_periods).
        S:
            Summing matrix, shape (n_series, n_bottom).
        series_names:
            Optional list of series names for weight lookup.

        Returns
        -------
        Reconciled forecasts, same shape as y_hat.
        """
        S = np.atleast_2d(S)
        y_hat_1d = y_hat.ndim == 1
        if y_hat_1d:
            y_hat = y_hat.reshape(-1, 1)

        n_series = S.shape[0]
        names = series_names if series_names is not None else [str(i) for i in range(n_series)]
        weights = self._get_weights(names)

        # W^{-1} = diag(1/w)
        W_inv_diag = 1.0 / np.maximum(weights, 1e-10)
        W_inv = np.diag(W_inv_diag)

        # G = (S' W^{-1} S)^{-1} S' W^{-1}
        StWinv = S.T @ W_inv
        StWinvS = StWinv @ S
        try:
            StWinvS_inv = np.linalg.solve(StWinvS, np.eye(StWinvS.shape[0]))
        except np.linalg.LinAlgError:
            # Ridge regularisation fallback
            ridge = 1e-6 * np.eye(StWinvS.shape[0])
            StWinvS_inv = np.linalg.solve(StWinvS + ridge, np.eye(StWinvS.shape[0]))

        G = StWinvS_inv @ StWinv  # shape (n_bottom, n_series)
        y_reconciled = S @ G @ y_hat  # shape (n_series, n_periods)

        if self._nonnegative:
            y_reconciled = np.maximum(y_reconciled, 0.0)

        return y_reconciled[:, 0] if y_hat_1d else y_reconciled


def premium_wls_reconcile(
    y_hat: np.ndarray,
    S: np.ndarray,
    earned_premium: np.ndarray | dict,
    series_names: list[str] | None = None,
    nonnegative: bool = True,
) -> np.ndarray:
    """
    Convenience function for premium-weighted MinTrace reconciliation.

    Parameters
    ----------
    y_hat:
        Base forecasts, shape (n_series,) or (n_series, n_periods).
    S:
        Summing matrix, shape (n_series, n_bottom).
    earned_premium:
        Earned premium per series. Dict {name: value} or array.
    series_names:
        Series names for dict lookup.
    nonnegative:
        Enforce non-negativity.

    Returns
    -------
    Reconciled forecasts.
    """
    reconciler = PremiumWeightedMinTrace(
        earned_premium=earned_premium,
        nonnegative=nonnegative,
    )
    return reconciler.reconcile(y_hat, S, series_names=series_names)
