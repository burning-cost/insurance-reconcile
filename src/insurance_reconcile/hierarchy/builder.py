"""
Builds S_df (summing matrix) and tags for hierarchicalforecast.

The hierarchicalforecast library expects:
  - S_df: pd.DataFrame with index = all series names, columns = bottom-level
    series names, values = 0/1 indicating which bottom series contribute to
    each aggregate. Bottom-level series have exactly one 1 (diagonal block).
  - tags: dict mapping level name to np.ndarray of series names at that level.

This module derives S_df and tags from an InsuranceHierarchy + a list of
bottom-level series names, without requiring the user to understand the
internal format.

We also wrap hierarchicalforecast's aggregate() for cases where the user has
a flat DataFrame with hierarchy columns and wants S_df built from it directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .spec import InsuranceHierarchy, PerilTree

__all__ = ["HierarchyBuilder", "build_S_df"]


def build_S_df(
    peril_tree: PerilTree,
    bottom_series: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Build the summing matrix S_df and tags dict from a PerilTree.

    Parameters
    ----------
    peril_tree:
        The hierarchy specification.
    bottom_series:
        Ordered list of bottom-level series names (one per peril). Must be
        in the same order as peril_tree.all_perils.

    Returns
    -------
    S_df:
        DataFrame with shape (n_all_series, n_bottom). Index = all series
        names; columns = bottom series names.
    tags:
        Dict mapping level name to array of series names at that level.

    Notes
    -----
    The S matrix is the standard summing matrix from Hyndman & Athanasopoulos
    (2021). Row i contains ones in the columns corresponding to bottom-level
    series that aggregate into series i. For the bottom level, S is identity.

    For a hierarchy:
        Portfolio -> {Cover_A: [Peril_1, Peril_2], Cover_B: [Peril_3]}

    S has 6 rows (1 portfolio + 2 covers + 3 perils) and 3 columns (perils).
    """
    perils = peril_tree.all_perils
    if len(bottom_series) != len(perils):
        raise ValueError(
            f"bottom_series has {len(bottom_series)} entries but peril_tree "
            f"has {len(perils)} perils"
        )

    n = len(perils)
    peril_to_idx = {p: i for i, p in enumerate(perils)}
    series_to_idx = {s: i for i, s in enumerate(bottom_series)}

    # --- Build rows ---
    # Portfolio row: all ones
    portfolio_row = np.ones((1, n), dtype=float)
    portfolio_name = peril_tree.portfolio

    # Cover rows: 1 for each peril in that cover
    cover_rows = []
    cover_names = []
    for cover, cover_perils in peril_tree.covers.items():
        row = np.zeros(n, dtype=float)
        for p in cover_perils:
            row[peril_to_idx[p]] = 1.0
        cover_rows.append(row)
        cover_names.append(cover)

    # Bottom-level rows: identity
    bottom_rows = np.eye(n, dtype=float)

    S = np.vstack([portfolio_row, np.array(cover_rows), bottom_rows])
    all_series = [portfolio_name] + cover_names + bottom_series

    S_df = pd.DataFrame(S, index=all_series, columns=bottom_series)

    # --- Build tags ---
    tags: dict[str, np.ndarray] = {
        "Portfolio": np.array([portfolio_name]),
        "Cover": np.array(cover_names),
        "Peril": np.array(bottom_series),
    }

    return S_df, tags


class HierarchyBuilder:
    """
    Builds reconciliation inputs from an InsuranceHierarchy.

    Parameters
    ----------
    hierarchy:
        The hierarchy specification.
    """

    def __init__(self, hierarchy: InsuranceHierarchy) -> None:
        self.hierarchy = hierarchy

    def build_S_df(
        self, bottom_series: list[str]
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Build S_df and tags for use with hierarchicalforecast.

        Parameters
        ----------
        bottom_series:
            Ordered list of bottom-level unique_id strings. For a peril-only
            hierarchy these correspond 1:1 with perils in definition order.

        Returns
        -------
        S_df, tags
        """
        return build_S_df(self.hierarchy.peril_tree, bottom_series)

    def build_Y_hat_df(
        self,
        forecasts: dict[str, pd.DataFrame],
        model_name: str = "model",
    ) -> pd.DataFrame:
        """
        Assemble a Y_hat_df in the format expected by HierarchicalReconciliation.

        Parameters
        ----------
        forecasts:
            Dict mapping series unique_id to DataFrame with columns ['ds', model_name].
        model_name:
            Name of the forecast column.

        Returns
        -------
        Y_hat_df with columns ['unique_id', 'ds', model_name].
        """
        parts = []
        for uid, df in forecasts.items():
            part = df.copy()
            part["unique_id"] = uid
            parts.append(part)
        result = pd.concat(parts, ignore_index=True)
        return result[["unique_id", "ds", model_name]]

    def from_wide_df(
        self,
        df: pd.DataFrame,
        ds_col: str = "ds",
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Build S_df and a Y_hat_df from a wide DataFrame.

        Parameters
        ----------
        df:
            Wide DataFrame with one column per series, plus a date column.
        ds_col:
            Name of the date column.

        Returns
        -------
        S_df, tags
        """
        series_cols = [c for c in df.columns if c != ds_col]
        bottom_series = [
            c for c in series_cols if c in self.hierarchy.peril_tree.all_perils
        ]
        return build_S_df(self.hierarchy.peril_tree, bottom_series)
