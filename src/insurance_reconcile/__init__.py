"""
insurance-reconcile: hierarchical forecast reconciliation with insurance semantics.

The core problem: insurance pricing teams run separate models at multiple
levels (portfolio, cover, peril, geography). These models give different
answers. MinTrace reconciliation finds the optimal linear combination that
satisfies the hierarchy constraints with minimum variance adjustment.

This library wraps Nixtla's hierarchicalforecast (optional) and adds:
  - Earned premium weighting (economically correct for insurance)
  - Loss ratio / loss cost transform (rates, not amounts)
  - Frequency x severity decomposition (log-space)
  - Peril/cover/geography hierarchy DSL
  - Coherence diagnostics with business-readable output

Quick start
-----------
>>> from insurance_reconcile import InsuranceReconciler
>>> from insurance_reconcile.hierarchy import InsuranceHierarchy
>>> from insurance_reconcile.simulate import simulate_uk_home
>>>
>>> lc_df, ep_df = simulate_uk_home()
>>> hierarchy = InsuranceHierarchy.uk_home()
>>> reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())
>>> report = reconciler.check_coherence(lc_df, ep_df)
>>> print(report.to_string())
"""

from .hierarchy.spec import InsuranceHierarchy, PerilTree, GeographicHierarchy
from .hierarchy.builder import HierarchyBuilder, build_S_df
from .reconcile.wrapper import InsuranceReconciler, ReconciliationResult
from .reconcile.loss_ratio import LossRatioReconciler
from .reconcile.freqsev import FreqSevReconciler
from .reconcile.premium_wls import PremiumWeightedMinTrace, premium_wls_reconcile
from .data.losscost import LossCostFrame
from .diagnostics.coherence import CoherenceReport, check_coherence
from .diagnostics.attribution import AttributionReport
from ._compat import HAS_HIERARCHICALFORECAST

__version__ = "0.1.0"

__all__ = [
    # Hierarchy
    "InsuranceHierarchy",
    "PerilTree",
    "GeographicHierarchy",
    "HierarchyBuilder",
    "build_S_df",
    # Reconcilers
    "InsuranceReconciler",
    "ReconciliationResult",
    "LossRatioReconciler",
    "FreqSevReconciler",
    "PremiumWeightedMinTrace",
    "premium_wls_reconcile",
    # Data
    "LossCostFrame",
    # Diagnostics
    "CoherenceReport",
    "check_coherence",
    "AttributionReport",
    # Compat
    "HAS_HIERARCHICALFORECAST",
]
