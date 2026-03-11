"""
Graceful degradation when hierarchicalforecast is not installed.

The library has two tiers of functionality:
  - Tier 1 (no optional deps): hierarchy spec, S_df construction, coherence
    checks, loss cost frame validation. All pure numpy/pandas.
  - Tier 2 (requires hierarchicalforecast): MinTrace reconciliation,
    PremiumWeightedMinTrace, probabilistic reconciliation.

Import hierarchicalforecast components through this module so callers get
a clear error rather than a cryptic ImportError deep in a call stack.
"""

from __future__ import annotations

__all__ = [
    "HAS_HIERARCHICALFORECAST",
    "require_hierarchicalforecast",
    "get_HierarchicalReconciliation",
    "get_MinTrace",
    "get_aggregate",
]

try:
    import hierarchicalforecast  # noqa: F401

    HAS_HIERARCHICALFORECAST = True
except ImportError:
    HAS_HIERARCHICALFORECAST = False


def require_hierarchicalforecast(feature: str = "MinTrace reconciliation") -> None:
    """Raise a helpful ImportError if hierarchicalforecast is not available."""
    if not HAS_HIERARCHICALFORECAST:
        raise ImportError(
            f"{feature} requires hierarchicalforecast. "
            "Install it with: pip install 'insurance-reconcile[mintrace]' "
            "or: pip install hierarchicalforecast>=1.5.0"
        )


def get_HierarchicalReconciliation():  # type: ignore[return]
    """Return HierarchicalReconciliation class, raising if not available."""
    require_hierarchicalforecast("HierarchicalReconciliation")
    from hierarchicalforecast.core import HierarchicalReconciliation  # type: ignore

    return HierarchicalReconciliation


def get_MinTrace():  # type: ignore[return]
    """Return MinTrace class, raising if not available."""
    require_hierarchicalforecast("MinTrace")
    from hierarchicalforecast.methods import MinTrace  # type: ignore

    return MinTrace


def get_aggregate():  # type: ignore[return]
    """Return aggregate function, raising if not available."""
    require_hierarchicalforecast("aggregate()")
    from hierarchicalforecast.utils import aggregate  # type: ignore

    return aggregate
