# insurance-reconcile

Hierarchical forecast reconciliation for insurance pricing teams.

## The problem

Pricing teams run separate models at multiple levels of the same hierarchy:
- A portfolio GLM says loss cost trend is +8%
- The cell-level GLMs aggregate to +6%
- Neither is obviously wrong

Standard practice is to pick one and override the other, or do a manual blend in a spreadsheet. Both approaches throw away information and introduce inconsistencies that pile up over reserve reviews and FCA audits.

MinTrace reconciliation (Wickramasuriya et al., 2019) finds the optimal linear combination that satisfies the hierarchy constraints with minimum variance adjustment. It is not a new idea — but applying it correctly to insurance data requires handling things that actuarial-generic implementations miss.

## Why a new library

Nixtla's [hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast) implements MinTrace well. The gaps are insurance-specific:

1. **Premium-weighted WLS**: `wls_struct` weights by number of bottom series in each aggregate. Insurance requires weight = earned premium. A cell with £50M EP should barely move; a cell with £100K EP can move substantially.

2. **Loss ratio / loss cost reconciliation**: Loss costs are rates, not amounts. `LC_total ≠ Σ LC_i`. Correct aggregation is `LC_total = Σ(LC_i × EP_i) / Σ(EP_i)`. Reconciling loss costs directly with standard MinTrace produces incoherent results even when the constraints appear satisfied.

3. **Frequency × severity decomposition**: `LC = freq × sev` is a multiplicative constraint. Standard MinTrace cannot reconcile `(freq, sev)` pairs directly. Log-space reconciliation makes it tractable.

4. **Peril tree DSL**: Building `S_df` by hand for a UK home peril tree (7 perils, 2 covers, 1 portfolio) is error-prone and undocumented. This library provides a declarative spec.

5. **Actuary-readable diagnostics**: hierarchicalforecast diagnostics give numerical metrics. Insurance needs: "Buildings cover (£29/RY) does not aggregate: Fire (£11) + EoW (£9) + Subsidence (£6) + Flood (£3) = £29. Discrepancy: 0.0%. Wait, that is coherent." It needs to find the ones that are not, by percentage.

## Installation

```bash
pip install insurance-reconcile
# With MinTrace reconciliation:
pip install 'insurance-reconcile[mintrace]'
```

The base package (numpy, pandas, scipy) handles coherence checking and hierarchy spec without hierarchicalforecast. The `[mintrace]` extra adds `hierarchicalforecast>=1.5.0` for full reconciliation.

## Quick start

```python
from insurance_reconcile import InsuranceReconciler
from insurance_reconcile.hierarchy import InsuranceHierarchy
from insurance_reconcile.simulate import simulate_uk_home, simulate_incoherent_uk_home

# Generate synthetic data (coherent, then perturbed)
lc_coherent, ep_df = simulate_uk_home(n_periods=24)
lc_incoherent, _ = simulate_incoherent_uk_home(n_periods=24)

# Check coherence of base forecasts — works without hierarchicalforecast
hierarchy = InsuranceHierarchy.uk_home()
reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())

report = reconciler.check_coherence(lc_incoherent, ep_df)
print(report.to_string())
# === Coherence Report ===
# Status: INCOHERENT
# Max discrepancy: 5.23%
# Violations found: 36
# ...

# Reconcile (requires hierarchicalforecast)
result = reconciler.reconcile(lc_incoherent, ep_df)
print(result.attribution.to_string())
```

## Hierarchy specification

Define your peril tree declaratively:

```python
from insurance_reconcile.hierarchy import PerilTree, InsuranceHierarchy

tree = PerilTree(
    portfolio='Home',
    covers={
        'Buildings': ['Fire', 'EoW', 'Subsidence', 'Flood', 'Storm'],
        'Contents':  ['Theft', 'AccidentalDamage'],
    },
)

# Or use the pre-built UK home structure:
hierarchy = InsuranceHierarchy.uk_home()
```

Build `S_df` for direct use with hierarchicalforecast:

```python
from insurance_reconcile.hierarchy import HierarchyBuilder

builder = HierarchyBuilder(hierarchy)
S_df, tags = builder.build_S_df(bottom_series=tree.all_perils)
```

## Premium-weighted reconciliation

```python
from insurance_reconcile.reconcile import PremiumWeightedMinTrace

ep = {
    'Fire': 5_000_000, 'EoW': 4_000_000, 'Subsidence': 1_000_000,
    'Flood': 500_000,  'Storm': 1_500_000,
    'Theft': 2_000_000, 'AccidentalDamage': 1_000_000,
    'Buildings': 12_000_000, 'Contents': 3_000_000,
    'Home': 15_000_000,
}

rec = PremiumWeightedMinTrace(earned_premium=ep, nonnegative=True)
y_reconciled = rec.reconcile(y_hat, S_df.values, series_names=list(S_df.index))
```

## Loss ratio reconciliation

```python
from insurance_reconcile.reconcile import LossRatioReconciler

lr_rec = LossRatioReconciler(earned_premium=ep)
# Automatically converts LC -> claims -> reconciles -> converts back
lc_reconciled = lr_rec.reconcile(lc_hat, S_df.values, series_names=names)
```

## Frequency × severity reconciliation

```python
from insurance_reconcile.reconcile import FreqSevReconciler

rec = FreqSevReconciler(earned_premium=ep)
freq_reconciled, sev_reconciled = rec.reconcile(
    freq_hat, sev_hat, S_df.values, series_names=names
)
```

The reconciler works in log space — log(LC) = log(freq) + log(sev) is an additive constraint — then back-transforms. This handles non-negativity automatically.

## Coherence diagnostics

```python
from insurance_reconcile.diagnostics import check_coherence

report = check_coherence(
    values_df=forecasts_wide_df,   # periods × series
    s_df=S_df,
    earned_premium_df=ep_wide_df,  # same shape
    tolerance_pct=0.01,
    is_loss_cost=True,
)

print(report.is_coherent)
print(report.max_discrepancy_pct)
df = report.to_dataframe()  # one row per violation
```

## Data validation

```python
from insurance_reconcile.data import LossCostFrame

frame = LossCostFrame(df)  # expects: unique_id, ds, loss_cost, earned_premium
errors = frame.validate(raise_on_error=False)

# Convert to claims for reconciliation:
claims_df = frame.to_claims_df()  # adds claims = loss_cost * earned_premium
```

## UK use cases

This library was built for UK general insurance pricing. Specific applications:

- **FCA geographic coherence**: Post PS21/5, geographic pricing differentials across postcode/area/region must be defensible. MinTrace reconciliation gives a minimum-adjustment, statistically optimal resolution.
- **Multi-peril home coherence**: Separate peril GLMs (Fire, EoW, Subsidence, ...) must aggregate to match combined model. Typical spreadsheet discrepancy: 3–8%.
- **Rate indication aggregation**: Portfolio trend must be consistent with weighted average of class trends.
- **Budget decomposition**: Board-approved portfolio LR target decomposed to cells via bottom-up models.

## Architecture

```
src/insurance_reconcile/
  hierarchy/
    spec.py       — InsuranceHierarchy, PerilTree, GeographicHierarchy (DSL)
    builder.py    — S_df and tags construction
  reconcile/
    premium_wls.py  — PremiumWeightedMinTrace
    loss_ratio.py   — LossRatioReconciler (LC <-> claims transform)
    freqsev.py      — FreqSevReconciler (log-space)
    wrapper.py      — InsuranceReconciler (main API)
  data/
    losscost.py   — LossCostFrame (validation + transforms)
  diagnostics/
    coherence.py  — CoherenceReport
    attribution.py — AttributionReport
  _compat.py     — graceful degradation without hierarchicalforecast
  simulate.py    — synthetic UK home data for testing
```

hierarchicalforecast is an optional dependency. The coherence checking and hierarchy specification tiers work without it. Only MinTrace reconciliation requires the optional install.

## Relationship to hierarchicalforecast

This library wraps hierarchicalforecast for the MinTrace computation. The Nixtla implementation handles numerical edge cases (ridge protection for `mint_shrink`, non-negative QP via OSQP) that we do not replicate. Our contribution is the insurance semantics layer: premium weighting, rate-to-amount transforms, peril tree DSL, and actuary-readable diagnostics.

## References

- Wickramasuriya, S.L., Athanasopoulos, G. & Hyndman, R.J. (2019). Optimal forecast reconciliation using unbiased estimating equations. *JASA* 114(526):804–819.
- Panagiotelis, A. et al. (2021). Forecast reconciliation: A geometric view. *IJF* 37(1):343–359.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
