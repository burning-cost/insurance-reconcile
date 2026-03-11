"""
LossCostFrame: validated container for loss cost data.

Loss cost = claims incurred / earned premium. This is a rate, not an amount.
This matters enormously for reconciliation: loss costs at different hierarchy
levels do NOT aggregate by simple sum. They aggregate as exposure-weighted
averages:

    LC_aggregate = sum(LC_i * EP_i) / sum(EP_i)

This module enforces that data satisfies this relationship and provides
utilities for converting between loss costs, claim amounts, and frequencies/
severities. The validation is the guard rail — callers often load data from
spreadsheets where someone has already summed rather than weighted the loss
costs, which produces silent errors downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["LossCostFrame"]


class LossCostFrame:
    """
    A validated container for insurance loss cost data at multiple hierarchy levels.

    Enforces the basic relationship: loss_cost = claims_incurred / earned_premium.
    Also validates that aggregate loss costs are earned-premium-weighted averages
    of their constituent bottom-level loss costs.

    Parameters
    ----------
    df:
        DataFrame with required columns: 'unique_id', 'ds', 'loss_cost',
        'earned_premium'. Optional columns: 'claims', 'frequency', 'severity'.
    hierarchy_s_df:
        Optional S_df (from HierarchyBuilder) for cross-level coherence checks.
    tolerance:
        Fractional tolerance for loss cost validation (default 1e-6).

    Example
    -------
    >>> frame = LossCostFrame(df)
    >>> frame.validate()
    >>> frame.to_claims_df()  # returns df with loss_cost replaced by claims
    """

    REQUIRED_COLS = {"unique_id", "ds", "loss_cost", "earned_premium"}

    def __init__(
        self,
        df: pd.DataFrame,
        hierarchy_s_df: pd.DataFrame | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"LossCostFrame missing required columns: {missing}")
        self._df = df.copy()
        self._s_df = hierarchy_s_df
        self.tolerance = tolerance

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def validate_non_negative(self) -> list[str]:
        """
        Return a list of validation errors for negative values.

        Loss costs, earned premium, and claims must be non-negative.
        """
        errors: list[str] = []

        neg_lc = self._df[self._df["loss_cost"] < 0]
        if len(neg_lc) > 0:
            errors.append(
                f"{len(neg_lc)} rows have negative loss_cost: "
                f"{neg_lc['unique_id'].unique().tolist()[:5]}"
            )

        neg_ep = self._df[self._df["earned_premium"] < 0]
        if len(neg_ep) > 0:
            errors.append(
                f"{len(neg_ep)} rows have negative earned_premium: "
                f"{neg_ep['unique_id'].unique().tolist()[:5]}"
            )

        if "claims" in self._df.columns:
            neg_claims = self._df[self._df["claims"] < 0]
            if len(neg_claims) > 0:
                errors.append(
                    f"{len(neg_claims)} rows have negative claims"
                )

        return errors

    def validate_loss_cost_relationship(self) -> list[str]:
        """
        Check that loss_cost = claims / earned_premium where claims is present.
        """
        if "claims" not in self._df.columns:
            return []

        errors: list[str] = []
        nonzero_ep = self._df[self._df["earned_premium"] > 0].copy()
        computed = nonzero_ep["claims"] / nonzero_ep["earned_premium"]
        diff = (computed - nonzero_ep["loss_cost"]).abs()
        bad = nonzero_ep[diff > self.tolerance]
        if len(bad) > 0:
            max_diff = diff.max()
            errors.append(
                f"{len(bad)} rows where loss_cost != claims/earned_premium "
                f"(max discrepancy: {max_diff:.6f})"
            )
        return errors

    def validate_freq_sev_relationship(self) -> list[str]:
        """
        Check that loss_cost = frequency * severity where both are present.
        """
        if "frequency" not in self._df.columns or "severity" not in self._df.columns:
            return []

        errors: list[str] = []
        computed = self._df["frequency"] * self._df["severity"]
        diff = (computed - self._df["loss_cost"]).abs() / (
            self._df["loss_cost"].abs() + 1e-10
        )
        bad = self._df[diff > self.tolerance]
        if len(bad) > 0:
            errors.append(
                f"{len(bad)} rows where loss_cost != frequency * severity"
            )
        return errors

    def validate(self, raise_on_error: bool = True) -> list[str]:
        """
        Run all validations and return a list of error messages.

        Parameters
        ----------
        raise_on_error:
            If True, raise ValueError on first non-empty error list.
        """
        errors: list[str] = []
        errors.extend(self.validate_non_negative())
        errors.extend(self.validate_loss_cost_relationship())
        errors.extend(self.validate_freq_sev_relationship())

        if errors and raise_on_error:
            raise ValueError("LossCostFrame validation failed:\n" + "\n".join(errors))
        return errors

    def to_claims_df(self) -> pd.DataFrame:
        """
        Return a copy of the DataFrame with 'claims' column added (or verified).

        claims = loss_cost * earned_premium.
        """
        df = self._df.copy()
        df["claims"] = df["loss_cost"] * df["earned_premium"]
        return df

    def from_claims_df(self, claims_df: pd.DataFrame) -> "LossCostFrame":
        """
        Create a new LossCostFrame from a claims DataFrame (after reconciliation).

        Converts claims back to loss costs using earned_premium.

        Parameters
        ----------
        claims_df:
            DataFrame with columns ['unique_id', 'ds', 'claims', 'earned_premium'].
        """
        df = claims_df.copy()
        nonzero = df["earned_premium"] > 0
        df["loss_cost"] = np.where(
            nonzero,
            df["claims"] / df["earned_premium"],
            0.0,
        )
        return LossCostFrame(df, hierarchy_s_df=self._s_df, tolerance=self.tolerance)

    def pivot_wide(
        self, value_col: str = "loss_cost", index_col: str = "ds"
    ) -> pd.DataFrame:
        """
        Pivot to a wide DataFrame with series as columns.

        Useful for building Y_hat_df inputs for hierarchicalforecast.
        """
        return self._df.pivot(index=index_col, columns="unique_id", values=value_col)

    def describe_coverage(self) -> pd.DataFrame:
        """
        Summarise earned premium and loss cost by series.

        Returns a DataFrame with one row per unique_id showing total EP,
        mean loss cost (EP-weighted), and series count.
        """
        grouped = (
            self._df.groupby("unique_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "n_periods": len(g),
                        "total_ep": g["earned_premium"].sum(),
                        "ep_weighted_lc": (
                            (g["loss_cost"] * g["earned_premium"]).sum()
                            / g["earned_premium"].sum()
                            if g["earned_premium"].sum() > 0
                            else np.nan
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        return grouped

    def __repr__(self) -> str:
        n_series = self._df["unique_id"].nunique()
        n_periods = self._df["ds"].nunique()
        return (
            f"LossCostFrame({n_series} series, {n_periods} periods, "
            f"{len(self._df)} rows)"
        )
