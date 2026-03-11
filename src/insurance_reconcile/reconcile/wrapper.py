"""
InsuranceReconciler: the main entry point.

Wraps the various reconcilers with a consistent API and handles the
bookkeeping of building S_df, assembling diagnostics, and returning a
clean result object.

Design choice: InsuranceReconciler does NOT require hierarchicalforecast to
be installed for coherence checks. The check_coherence() method runs purely
on numpy/pandas. Only the reconcile() method requires hierarchicalforecast
(via the reconcilers it calls).

The result object (ReconciliationResult) carries both the reconciled values
and the diagnostic reports, so callers can decide how much to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..hierarchy.spec import InsuranceHierarchy
from ..hierarchy.builder import build_S_df
from ..diagnostics.coherence import check_coherence, CoherenceReport
from ..diagnostics.attribution import AttributionReport

__all__ = ["ReconciliationResult", "InsuranceReconciler"]


@dataclass
class ReconciliationResult:
    """
    Output from InsuranceReconciler.reconcile().

    Attributes
    ----------
    reconciled_df:
        Wide DataFrame (periods x series) of reconciled loss costs.
    coherence_before:
        Coherence report on the base forecasts (pre-reconciliation).
    coherence_after:
        Coherence report on the reconciled forecasts.
    attribution:
        Attribution report showing adjustments made.
    method:
        Name of the reconciliation method used.
    metadata:
        Dict with any additional info from the reconciler.
    """

    reconciled_df: pd.DataFrame
    coherence_before: CoherenceReport | None = None
    coherence_after: CoherenceReport | None = None
    attribution: AttributionReport | None = None
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_series = len(self.reconciled_df.columns)
        n_periods = len(self.reconciled_df)
        after_str = (
            f", after={self.coherence_after.max_discrepancy_pct:.4f}%"
            if self.coherence_after is not None
            else ""
        )
        return (
            f"ReconciliationResult({n_series} series, {n_periods} periods, "
            f"method={self.method!r}{after_str})"
        )


class InsuranceReconciler:
    """
    High-level reconciler for insurance loss cost hierarchies.

    Combines hierarchy specification, reconciliation method, and diagnostics
    into a single object. The typical workflow:

        1. Specify the hierarchy (PerilTree + optional geography)
        2. Call check_coherence() to audit base forecasts
        3. Call reconcile() to produce coherent forecasts
        4. Inspect ReconciliationResult for adjustments

    Parameters
    ----------
    hierarchy:
        The hierarchy specification (InsuranceHierarchy).
    reconciler:
        A reconciler with a reconcile(y_hat, S, series_names) method.
        If None, uses PremiumWeightedMinTrace (requires hierarchicalforecast).
    earned_premium:
        Dict {series_name: float} or pd.Series. Used for:
        - PremiumWeightedMinTrace weights (if reconciler is None)
        - Coherence check weighting
        - Attribution EP-impact calculation
    nonnegative:
        Enforce non-negativity in reconciliation. Default True.

    Example
    -------
    >>> from insurance_reconcile import InsuranceReconciler
    >>> from insurance_reconcile.hierarchy import InsuranceHierarchy
    >>>
    >>> hierarchy = InsuranceHierarchy.uk_home()
    >>> ep = {'Fire': 5e6, 'EoW': 3e6, 'Subsidence': 1e6,
    ...       'Flood': 0.5e6, 'Storm': 1.5e6,
    ...       'Buildings': 11e6,
    ...       'Home': 11e6}
    >>> reconciler = InsuranceReconciler(hierarchy, earned_premium=ep)
    >>> report = reconciler.check_coherence(forecasts_df, ep_df)
    >>> result = reconciler.reconcile(forecasts_df, ep_df)
    """

    def __init__(
        self,
        hierarchy: InsuranceHierarchy,
        reconciler: Any = None,
        earned_premium: dict | pd.Series | None = None,
        nonnegative: bool = True,
    ) -> None:
        self.hierarchy = hierarchy
        self._reconciler = reconciler
        self._earned_premium = earned_premium
        self._nonnegative = nonnegative

    def _get_reconciler(self, series_names: list[str], ep_array: np.ndarray):
        """Build default PremiumWeightedMinTrace if no reconciler supplied."""
        if self._reconciler is not None:
            return self._reconciler
        from .premium_wls import PremiumWeightedMinTrace

        ep_dict = dict(zip(series_names, ep_array))
        return PremiumWeightedMinTrace(
            earned_premium=ep_dict,
            nonnegative=self._nonnegative,
        )

    def _resolve_ep(
        self, series_names: list[str], ep_df: pd.DataFrame | None
    ) -> np.ndarray:
        """Resolve earned premium to a 1D array aligned to series_names."""
        if ep_df is not None:
            return np.array(
                [float(ep_df[s].mean()) if s in ep_df.columns else 1.0 for s in series_names]
            )
        if self._earned_premium is not None:
            ep = self._earned_premium
            if isinstance(ep, dict):
                return np.array([float(ep.get(s, 1.0)) for s in series_names])
            if isinstance(ep, pd.Series):
                return np.array([float(ep.get(s, 1.0)) for s in series_names])
        return np.ones(len(series_names))

    def check_coherence(
        self,
        forecasts_df: pd.DataFrame,
        earned_premium_df: pd.DataFrame | None = None,
        bottom_series: list[str] | None = None,
        tolerance_pct: float = 0.01,
    ) -> CoherenceReport:
        """
        Check whether base forecasts satisfy hierarchy constraints.

        This method works without hierarchicalforecast installed.

        Parameters
        ----------
        forecasts_df:
            Wide DataFrame (periods x series) containing all hierarchy levels.
        earned_premium_df:
            Wide DataFrame of earned premiums. Required for loss cost coherence.
        bottom_series:
            Ordered list of bottom-level series names. If None, uses
            hierarchy.peril_tree.all_perils.
        tolerance_pct:
            Percentage tolerance for coherence violations.

        Returns
        -------
        CoherenceReport
        """
        if bottom_series is None:
            bottom_series = self.hierarchy.peril_tree.all_perils

        # Intersect with columns actually present in forecasts_df
        available = set(forecasts_df.columns)
        bottom_series = [s for s in bottom_series if s in available]

        S_df, tags = build_S_df(self.hierarchy.peril_tree, bottom_series)

        # Filter to series we have forecasts for
        present_series = [s for s in S_df.index if s in forecasts_df.columns]
        filtered_forecasts = forecasts_df[present_series]
        filtered_S = S_df.loc[present_series]

        ep_df_filtered = None
        if earned_premium_df is not None:
            ep_cols = [s for s in present_series if s in earned_premium_df.columns]
            if ep_cols:
                ep_df_filtered = earned_premium_df[ep_cols]

        return check_coherence(
            values_df=filtered_forecasts,
            s_df=filtered_S,
            earned_premium_df=ep_df_filtered,
            tolerance_pct=tolerance_pct,
            is_loss_cost=(earned_premium_df is not None),
        )

    def reconcile(
        self,
        forecasts_df: pd.DataFrame,
        earned_premium_df: pd.DataFrame | None = None,
        bottom_series: list[str] | None = None,
        run_diagnostics: bool = True,
    ) -> ReconciliationResult:
        """
        Reconcile loss cost forecasts to satisfy hierarchy constraints.

        Parameters
        ----------
        forecasts_df:
            Wide DataFrame (periods x series). Must include all aggregate
            and bottom-level series.
        earned_premium_df:
            Wide DataFrame of earned premiums. Strongly recommended.
        bottom_series:
            Ordered list of bottom-level series. If None, infers from hierarchy.
        run_diagnostics:
            Whether to compute coherence and attribution diagnostics.

        Returns
        -------
        ReconciliationResult
        """
        if bottom_series is None:
            bottom_series = self.hierarchy.peril_tree.all_perils

        # Filter to available series
        available = set(forecasts_df.columns)
        bottom_series = [s for s in bottom_series if s in available]

        S_df, tags = build_S_df(self.hierarchy.peril_tree, bottom_series)
        series_names = list(S_df.index)
        present = [s for s in series_names if s in forecasts_df.columns]
        fc = forecasts_df[present]

        ep_array = self._resolve_ep(present, earned_premium_df)
        reconciler = self._get_reconciler(present, ep_array)

        # Coherence check before
        coherence_before = None
        if run_diagnostics:
            coherence_before = self.check_coherence(
                fc, earned_premium_df, bottom_series=bottom_series
            )

        # Reconcile using loss ratio transform if EP is available
        if earned_premium_df is not None:
            from .loss_ratio import LossRatioReconciler

            lr_rec = LossRatioReconciler(
                earned_premium=earned_premium_df,
                reconciler=reconciler,
                nonnegative=self._nonnegative,
            )
            reconciled_array = lr_rec.reconcile(
                lc_hat=fc.values.T,
                S=S_df.loc[present].values,
                series_names=present,
            )
        else:
            reconciled_array = reconciler.reconcile(
                fc.values.T,
                S_df.loc[present].values,
                series_names=present,
            )

        if reconciled_array.ndim == 1:
            reconciled_array = reconciled_array.reshape(-1, 1)

        reconciled_df = pd.DataFrame(
            reconciled_array.T,
            index=fc.index,
            columns=present,
        )

        # Post-reconciliation diagnostics
        coherence_after = None
        attribution = None
        if run_diagnostics:
            coherence_after = self.check_coherence(
                reconciled_df, earned_premium_df, bottom_series=bottom_series
            )
            ep_df_for_attr = earned_premium_df
            attribution = AttributionReport.from_before_after(
                before_df=fc,
                after_df=reconciled_df,
                earned_premium_df=ep_df_for_attr,
            )

        method_name = type(reconciler).__name__
        return ReconciliationResult(
            reconciled_df=reconciled_df,
            coherence_before=coherence_before,
            coherence_after=coherence_after,
            attribution=attribution,
            method=method_name,
        )
