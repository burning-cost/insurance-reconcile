"""
Coherence diagnostics for insurance loss cost hierarchies.

'Coherent' means the hierarchy constraints are satisfied: every aggregate series
equals the correct weighted combination of its children. For loss costs that
means premium-weighted averages, not simple sums.

The CoherenceReport is designed to be readable by actuaries and pricing
managers, not just data scientists. The output names series by their business
labels (e.g. 'Buildings cover' rather than 'node_0042') and states violations
in percentage terms.

This module works without hierarchicalforecast — it only needs numpy and pandas.
The check_coherence() function is the main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["CoherenceViolation", "CoherenceReport", "check_coherence"]


@dataclass
class CoherenceViolation:
    """A single coherence violation at one node/period."""

    series_name: str
    ds: Any  # date or period label
    actual_value: float
    expected_value: float
    discrepancy_abs: float
    discrepancy_pct: float
    children: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        child_str = " + ".join(self.children) if self.children else "children"
        return (
            f"{self.series_name} (period {self.ds}): "
            f"actual={self.actual_value:.4f}, "
            f"expected from {child_str}={self.expected_value:.4f}, "
            f"discrepancy={self.discrepancy_pct:+.2f}%"
        )


@dataclass
class CoherenceReport:
    """
    Report on whether a set of forecasts satisfies hierarchy constraints.

    Attributes
    ----------
    is_coherent:
        True if all violations are within tolerance.
    violations:
        List of all CoherenceViolation instances found.
    max_discrepancy_pct:
        Largest absolute percentage discrepancy found.
    n_series_checked:
        Total number of aggregate series checked.
    n_periods_checked:
        Total number of periods checked per series.
    tolerance_pct:
        The tolerance used (percentage).
    """

    is_coherent: bool
    violations: list[CoherenceViolation]
    max_discrepancy_pct: float
    n_series_checked: int
    n_periods_checked: int
    tolerance_pct: float

    def to_dataframe(self) -> pd.DataFrame:
        """Return violations as a DataFrame."""
        if not self.violations:
            return pd.DataFrame(
                columns=[
                    "series_name",
                    "ds",
                    "actual_value",
                    "expected_value",
                    "discrepancy_abs",
                    "discrepancy_pct",
                ]
            )
        rows = [
            {
                "series_name": v.series_name,
                "ds": v.ds,
                "actual_value": v.actual_value,
                "expected_value": v.expected_value,
                "discrepancy_abs": v.discrepancy_abs,
                "discrepancy_pct": v.discrepancy_pct,
                "children": ", ".join(v.children),
            }
            for v in self.violations
        ]
        return pd.DataFrame(rows)

    def to_string(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=== Coherence Report ===",
            f"Status: {'COHERENT' if self.is_coherent else 'INCOHERENT'}",
            f"Series checked: {self.n_series_checked}",
            f"Periods checked: {self.n_periods_checked}",
            f"Tolerance: {self.tolerance_pct:.3f}%",
            f"Max discrepancy: {self.max_discrepancy_pct:.3f}%",
            f"Violations found: {len(self.violations)}",
        ]
        if self.violations:
            lines.append("")
            lines.append("Top violations:")
            top = sorted(
                self.violations, key=lambda v: abs(v.discrepancy_pct), reverse=True
            )[:10]
            for v in top:
                lines.append(f"  {v}")
        return "\n".join(lines)

    def worst_series(self, n: int = 5) -> list[str]:
        """Return the n series with the largest maximum discrepancy."""
        by_series: dict[str, float] = {}
        for v in self.violations:
            by_series[v.series_name] = max(
                by_series.get(v.series_name, 0.0), abs(v.discrepancy_pct)
            )
        return sorted(by_series, key=lambda s: -by_series[s])[:n]

    def __repr__(self) -> str:
        status = "COHERENT" if self.is_coherent else f"{len(self.violations)} violations"
        return f"CoherenceReport({status}, max={self.max_discrepancy_pct:.3f}%)"


def check_coherence(
    values_df: pd.DataFrame,
    s_df: pd.DataFrame,
    earned_premium_df: pd.DataFrame | None = None,
    tolerance_pct: float = 0.01,
    is_loss_cost: bool = True,
) -> CoherenceReport:
    """
    Check whether a set of forecasts satisfies the hierarchy constraints.

    For loss costs (rates), the constraint is premium-weighted:
        aggregate_lc = sum(child_lc * child_ep) / sum(child_ep)

    For amounts (claims), the constraint is additive:
        aggregate_claims = sum(child_claims)

    Parameters
    ----------
    values_df:
        Wide DataFrame with columns = series names, index = periods.
        Columns must include all series in s_df.index.
    s_df:
        Summing matrix with index = all series, columns = bottom-level series.
        Shape (n_all, n_bottom).
    earned_premium_df:
        Wide DataFrame matching values_df shape, with earned premiums.
        Required when is_loss_cost=True.
    tolerance_pct:
        Percentage tolerance for coherence (default 0.01%).
    is_loss_cost:
        If True, expect loss cost (rate) aggregation. If False, additive.

    Returns
    -------
    CoherenceReport
    """
    if is_loss_cost and earned_premium_df is None:
        raise ValueError(
            "earned_premium_df is required for loss cost coherence checks. "
            "Loss costs aggregate as EP-weighted averages, not simple sums. "
            "Pass is_loss_cost=False to check additive aggregation."
        )

    # Identify aggregate rows (those that are not purely bottom-level)
    bottom_series = list(s_df.columns)
    all_series = list(s_df.index)
    aggregate_series = [s for s in all_series if s not in bottom_series]

    violations: list[CoherenceViolation] = []
    n_periods = len(values_df)

    for agg_series in aggregate_series:
        row = s_df.loc[agg_series]
        child_bottom = [c for c in bottom_series if row[c] > 0]

        if not child_bottom:
            continue

        # Find which aggregate children this series has (one level down)
        # For coherence checks we compare direct parent vs all bottom children
        actual_vals = values_df[agg_series] if agg_series in values_df.columns else None
        if actual_vals is None:
            continue

        if is_loss_cost:
            ep_agg = earned_premium_df[agg_series] if agg_series in earned_premium_df.columns else None
            ep_children = earned_premium_df[child_bottom] if all(
                c in earned_premium_df.columns for c in child_bottom
            ) else None

            if ep_agg is None or ep_children is None:
                continue

            lc_children = values_df[child_bottom]
            # EP-weighted average
            total_ep_children = ep_children.sum(axis=1)
            expected_vals = (lc_children * ep_children.values).sum(axis=1) / total_ep_children.replace(0, np.nan)
        else:
            # Additive: sum of bottom-level children
            expected_vals = values_df[child_bottom].sum(axis=1)

        for period in values_df.index:
            actual = actual_vals.loc[period]
            expected = expected_vals.loc[period]

            if np.isnan(expected) or np.isnan(actual):
                continue

            denom = abs(expected) if abs(expected) > 1e-10 else 1e-10
            disc_pct = 100.0 * (actual - expected) / denom
            disc_abs = abs(actual - expected)

            if abs(disc_pct) > tolerance_pct:
                violations.append(
                    CoherenceViolation(
                        series_name=agg_series,
                        ds=period,
                        actual_value=float(actual),
                        expected_value=float(expected),
                        discrepancy_abs=float(disc_abs),
                        discrepancy_pct=float(disc_pct),
                        children=child_bottom,
                    )
                )

    max_disc = max((abs(v.discrepancy_pct) for v in violations), default=0.0)
    return CoherenceReport(
        is_coherent=len(violations) == 0,
        violations=violations,
        max_discrepancy_pct=max_disc,
        n_series_checked=len(aggregate_series),
        n_periods_checked=n_periods,
        tolerance_pct=tolerance_pct,
    )
