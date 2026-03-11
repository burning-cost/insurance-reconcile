"""
Attribution: how much was each series adjusted and why.

After reconciliation, actuaries want to understand what changed and whether
the changes are defensible. A loss cost that moved from £145 to £138 on a
peril with £50M earned premium is a material change that needs sign-off.

AttributionReport captures the before/after for every series, the absolute
adjustment, the percentage adjustment, and the contribution of that adjustment
to the overall portfolio loss ratio movement.

Design choice: attribution is purely computational — it does not know why
an adjustment happened (that requires understanding the reconciler's W matrix).
It reports what changed; interpretation is left to the actuary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["SeriesAdjustment", "AttributionReport"]


@dataclass
class SeriesAdjustment:
    """Adjustment record for a single series."""

    series_name: str
    mean_before: float
    mean_after: float
    mean_adjustment: float
    mean_adjustment_pct: float
    max_abs_adjustment: float
    total_ep: float  # total earned premium for this series (if available)
    ep_weighted_lc_impact: float  # impact on portfolio LR (if EP available)

    def __str__(self) -> str:
        return (
            f"{self.series_name}: "
            f"{self.mean_before:.4f} -> {self.mean_after:.4f} "
            f"({self.mean_adjustment_pct:+.2f}%)"
        )


@dataclass
class AttributionReport:
    """
    Attribution report showing adjustments made during reconciliation.

    Attributes
    ----------
    adjustments:
        List of SeriesAdjustment records, one per series.
    portfolio_lc_before:
        EP-weighted portfolio loss cost before reconciliation.
    portfolio_lc_after:
        EP-weighted portfolio loss cost after reconciliation.
    n_series:
        Number of series in the reconciliation.
    """

    adjustments: list[SeriesAdjustment]
    portfolio_lc_before: float | None
    portfolio_lc_after: float | None
    n_series: int

    @classmethod
    def from_before_after(
        cls,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        earned_premium_df: pd.DataFrame | None = None,
    ) -> "AttributionReport":
        """
        Compute attribution from before/after wide DataFrames.

        Parameters
        ----------
        before_df:
            Wide DataFrame (periods x series) with pre-reconciliation values.
        after_df:
            Wide DataFrame (periods x series) with post-reconciliation values.
        earned_premium_df:
            Optional wide DataFrame with earned premiums (same shape as before_df).
        """
        series = list(before_df.columns)
        adjustments: list[SeriesAdjustment] = []

        for s in series:
            b = before_df[s].dropna()
            a = after_df[s].dropna() if s in after_df.columns else b

            mean_b = float(b.mean())
            mean_a = float(a.mean())
            adj = mean_a - mean_b
            denom = abs(mean_b) if abs(mean_b) > 1e-10 else 1e-10
            adj_pct = 100.0 * adj / denom
            max_abs = float((a - b).abs().max()) if len(a) == len(b) else np.nan

            total_ep = 0.0
            ep_impact = 0.0
            if earned_premium_df is not None and s in earned_premium_df.columns:
                ep = earned_premium_df[s].dropna()
                total_ep = float(ep.sum())
                ep_impact = float((ep * (a - b)).sum()) if len(ep) == len(b) else 0.0

            adjustments.append(
                SeriesAdjustment(
                    series_name=s,
                    mean_before=mean_b,
                    mean_after=mean_a,
                    mean_adjustment=adj,
                    mean_adjustment_pct=adj_pct,
                    max_abs_adjustment=max_abs,
                    total_ep=total_ep,
                    ep_weighted_lc_impact=ep_impact,
                )
            )

        portfolio_before = None
        portfolio_after = None
        if earned_premium_df is not None:
            try:
                ep_totals = earned_premium_df[series].sum(axis=1)
                total_ep = ep_totals.sum()
                if total_ep > 0:
                    portfolio_before = float(
                        (before_df[series] * earned_premium_df[series].values).sum().sum()
                        / total_ep
                    )
                    portfolio_after = float(
                        (after_df[series] * earned_premium_df[series].values).sum().sum()
                        / total_ep
                    )
            except Exception:
                pass

        return cls(
            adjustments=adjustments,
            portfolio_lc_before=portfolio_before,
            portfolio_lc_after=portfolio_after,
            n_series=len(series),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return adjustments as a DataFrame, sorted by absolute adjustment."""
        rows = [
            {
                "series_name": a.series_name,
                "mean_before": a.mean_before,
                "mean_after": a.mean_after,
                "mean_adjustment": a.mean_adjustment,
                "mean_adjustment_pct": a.mean_adjustment_pct,
                "max_abs_adjustment": a.max_abs_adjustment,
                "total_ep": a.total_ep,
                "ep_weighted_lc_impact": a.ep_weighted_lc_impact,
            }
            for a in self.adjustments
        ]
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("mean_adjustment_pct", key=abs, ascending=False)
        return df

    def to_string(self) -> str:
        """Return a human-readable attribution summary."""
        lines = [
            "=== Attribution Report ===",
            f"Series reconciled: {self.n_series}",
        ]
        if self.portfolio_lc_before is not None:
            lines.append(
                f"Portfolio loss cost: {self.portfolio_lc_before:.4f} -> "
                f"{self.portfolio_lc_after:.4f}"
            )
        lines.append("")
        lines.append("Adjustments by series (largest first):")
        top = sorted(
            self.adjustments, key=lambda a: abs(a.mean_adjustment_pct), reverse=True
        )
        for adj in top[:20]:
            lines.append(f"  {adj}")
        return "\n".join(lines)

    def largest_adjustments(self, n: int = 10) -> list[SeriesAdjustment]:
        """Return the n series with the largest percentage adjustment."""
        return sorted(
            self.adjustments, key=lambda a: abs(a.mean_adjustment_pct), reverse=True
        )[:n]

    def __repr__(self) -> str:
        max_adj = max(
            (abs(a.mean_adjustment_pct) for a in self.adjustments), default=0.0
        )
        return f"AttributionReport({self.n_series} series, max_adj={max_adj:.2f}%)"
