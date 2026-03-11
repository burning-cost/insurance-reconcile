"""
Synthetic data generation for testing and examples.

Generates realistic hierarchical insurance loss cost data:
  - Top-level portfolio loss cost
  - Cover-level loss costs (Buildings, Contents)
  - Peril-level loss costs (Fire, EoW, Subsidence, Flood, Storm, Theft, AD)
  - Earned premium at each level
  - Intentional incoherence for testing diagnostics

The data is loosely calibrated to UK home insurance market rates as of 2024.
These are invented numbers but in realistic ballparks for demonstration purposes.

Loss cost rates (£ per £1 of earned premium, or equivalently the pure premium
rate as a decimal):
  Fire: 0.020–0.035
  Escape of Water: 0.025–0.045
  Subsidence: 0.008–0.015
  Flood: 0.005–0.025
  Storm: 0.008–0.018
  Theft: 0.012–0.020
  Accidental Damage: 0.018–0.030
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "simulate_uk_home",
    "simulate_incoherent_uk_home",
    "make_hierarchy_dataframe",
]

# UK home peril rates (mean loss cost as fraction of earned premium)
_PERIL_RATES = {
    "Fire": 0.028,
    "EoW": 0.035,
    "Subsidence": 0.011,
    "Flood": 0.014,
    "Storm": 0.013,
    "Theft": 0.016,
    "AccidentalDamage": 0.024,
}

# Earned premium allocation (fraction of total)
_PERIL_EP_WEIGHTS = {
    "Fire": 0.18,
    "EoW": 0.22,
    "Subsidence": 0.08,
    "Flood": 0.10,
    "Storm": 0.12,
    "Theft": 0.14,
    "AccidentalDamage": 0.16,
}

# Cover membership
_COVER_PERILS = {
    "Buildings": ["Fire", "EoW", "Subsidence", "Flood", "Storm"],
    "Contents": ["Theft", "AccidentalDamage"],
}


def _compute_cover_and_portfolio(
    peril_lc: dict[str, np.ndarray],
    peril_ep: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute cover and portfolio loss costs as premium-weighted averages.
    Also computes aggregate EPs.
    """
    cover_lc: dict[str, np.ndarray] = {}
    cover_ep: dict[str, np.ndarray] = {}

    for cover, perils in _COVER_PERILS.items():
        ep_total = sum(peril_ep[p] for p in perils)
        lc_weighted = sum(peril_lc[p] * peril_ep[p] for p in perils)
        cover_ep[cover] = ep_total
        cover_lc[cover] = lc_weighted / ep_total

    portfolio_ep = sum(cover_ep.values())
    portfolio_lc = sum(cover_lc[c] * cover_ep[c] for c in cover_ep) / portfolio_ep

    return {**peril_lc, **cover_lc, "Home": portfolio_lc}, {
        **peril_ep, **cover_ep, "Home": portfolio_ep
    }


def simulate_uk_home(
    n_periods: int = 24,
    total_ep: float = 50_000_000.0,
    seed: int = 42,
    noise_scale: float = 0.05,
    trend: float = 0.003,  # ~3.6% annual loss cost trend, monthly
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a coherent synthetic UK home insurance loss cost dataset.

    The data is coherent by construction — aggregate loss costs are correctly
    computed as earned premium-weighted averages of their children.

    Parameters
    ----------
    n_periods:
        Number of monthly periods.
    total_ep:
        Total portfolio earned premium per period.
    seed:
        Random seed for reproducibility.
    noise_scale:
        Multiplicative noise on peril loss costs (std as fraction of mean).
    trend:
        Monthly loss cost trend (additive to loss cost rate).

    Returns
    -------
    lc_df:
        Wide DataFrame (n_periods x n_series) of loss costs.
        Columns: all peril, cover, and portfolio series.
    ep_df:
        Wide DataFrame (n_periods x n_series) of earned premiums.
    """
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    t = np.arange(n_periods)

    # Simulate peril-level loss costs with trend + noise
    peril_lc: dict[str, np.ndarray] = {}
    peril_ep: dict[str, np.ndarray] = {}

    for peril, base_rate in _PERIL_RATES.items():
        ep_weight = _PERIL_EP_WEIGHTS[peril]
        ep_series = np.full(n_periods, total_ep * ep_weight)
        # Add small EP seasonality
        ep_series *= 1.0 + 0.02 * np.sin(2 * np.pi * t / 12)

        noise = rng.normal(0, noise_scale * base_rate, n_periods)
        lc_series = base_rate + trend * t + noise
        lc_series = np.maximum(lc_series, 0.0)

        peril_lc[peril] = lc_series
        peril_ep[peril] = ep_series

    all_lc, all_ep = _compute_cover_and_portfolio(peril_lc, peril_ep)

    lc_df = pd.DataFrame(all_lc, index=periods)
    ep_df = pd.DataFrame(all_ep, index=periods)

    return lc_df, ep_df


def simulate_incoherent_uk_home(
    n_periods: int = 24,
    total_ep: float = 50_000_000.0,
    seed: int = 42,
    incoherence_scale: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a dataset with intentional incoherence for testing diagnostics.

    Takes the coherent dataset and adds random perturbations to aggregate
    series to simulate the kind of discrepancies that arise when portfolio
    models and cell models are run independently.

    Parameters
    ----------
    incoherence_scale:
        Scale of perturbation to add to aggregate series (fraction of mean).

    Returns
    -------
    lc_df:
        Incoherent loss cost DataFrame.
    ep_df:
        Earned premium DataFrame (coherent — EP doesn't have this problem).
    """
    lc_df, ep_df = simulate_uk_home(n_periods=n_periods, total_ep=total_ep, seed=seed)
    rng = np.random.default_rng(seed + 1000)

    lc_incoherent = lc_df.copy()
    # Perturb aggregate series only (not bottom-level perils)
    for col in ["Buildings", "Contents", "Home"]:
        if col in lc_incoherent.columns:
            mean_lc = lc_incoherent[col].mean()
            noise = rng.normal(0, incoherence_scale * mean_lc, n_periods)
            lc_incoherent[col] = lc_incoherent[col] + noise
            lc_incoherent[col] = lc_incoherent[col].clip(lower=0.001)

    return lc_incoherent, ep_df


def make_hierarchy_dataframe(
    lc_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    model_name: str = "GLM",
) -> pd.DataFrame:
    """
    Combine loss costs and earned premiums into a long-format DataFrame
    suitable for LossCostFrame.

    Parameters
    ----------
    lc_df:
        Wide loss cost DataFrame (periods x series).
    ep_df:
        Wide earned premium DataFrame (periods x series).
    model_name:
        Optional column added to identify the forecast model.

    Returns
    -------
    Long DataFrame with columns: unique_id, ds, loss_cost, earned_premium.
    """
    parts = []
    for col in lc_df.columns:
        df = pd.DataFrame(
            {
                "unique_id": col,
                "ds": lc_df.index,
                "loss_cost": lc_df[col].values,
                "earned_premium": ep_df[col].values if col in ep_df.columns else np.nan,
            }
        )
        parts.append(df)

    result = pd.concat(parts, ignore_index=True)
    if model_name:
        result["model"] = model_name
    return result
