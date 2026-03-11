# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-reconcile: UK Home Loss Cost Hierarchy Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Define a UK home peril hierarchy
# MAGIC 2. Generate synthetic loss cost data (coherent + incoherent)
# MAGIC 3. Run coherence diagnostics on base forecasts
# MAGIC 4. Reconcile using PremiumWeightedMinTrace
# MAGIC 5. Inspect attribution report

# COMMAND ----------

# MAGIC %pip install insurance-reconcile hierarchicalforecast>=1.5.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Hierarchy Specification

# COMMAND ----------

from insurance_reconcile.hierarchy.spec import PerilTree, InsuranceHierarchy
from insurance_reconcile.hierarchy.builder import HierarchyBuilder, build_S_df

# UK home peril tree — standard structure used by most UK property insurers
tree = PerilTree(
    portfolio="Home",
    covers={
        "Buildings": ["Fire", "EoW", "Subsidence", "Flood", "Storm"],
        "Contents":  ["Theft", "AccidentalDamage"],
    },
)

print(f"Portfolio: {tree.portfolio}")
print(f"Covers: {tree.all_covers}")
print(f"Perils: {tree.all_perils}")
print(f"Total nodes: {len(tree.all_nodes())}")

# COMMAND ----------

# Build the summing matrix S
S_df, tags = build_S_df(tree, tree.all_perils)

print("S_df shape:", S_df.shape)
print("\nS_df (summing matrix):")
display(S_df)

# COMMAND ----------

print("Tags:")
for level, series in tags.items():
    print(f"  {level}: {list(series)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Data Generation

# COMMAND ----------

from insurance_reconcile.simulate import simulate_uk_home, simulate_incoherent_uk_home

# Generate 24 months of coherent data
lc_coherent, ep_df = simulate_uk_home(
    n_periods=24,
    total_ep=50_000_000.0,
    seed=42,
    noise_scale=0.04,
)

print("Loss cost DataFrame (coherent):")
print(f"  Shape: {lc_coherent.shape}")
print(f"  Columns: {list(lc_coherent.columns)}")
print(f"\nMean loss costs by series:")
for col in lc_coherent.columns:
    ep_mean = ep_df[col].mean() / 1e6
    lc_mean = lc_coherent[col].mean()
    print(f"  {col:<25}: LC={lc_mean:.4f}  EP=£{ep_mean:.1f}M")

# COMMAND ----------

display(lc_coherent)

# COMMAND ----------

# Now generate incoherent data (simulates what happens when portfolio model
# and cell-level models are run independently without reconciliation)
lc_incoherent, _ = simulate_incoherent_uk_home(
    n_periods=24,
    total_ep=50_000_000.0,
    incoherence_scale=0.06,
)

print("Difference between coherent and incoherent (aggregate series only):")
for col in ["Buildings", "Contents", "Home"]:
    diff = (lc_incoherent[col] - lc_coherent[col]).abs().mean()
    pct = 100 * diff / lc_coherent[col].mean()
    print(f"  {col}: mean abs diff = {diff:.5f} ({pct:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Coherence Diagnostics

# COMMAND ----------

from insurance_reconcile.diagnostics.coherence import check_coherence

# Check the COHERENT data — should pass
report_coherent = check_coherence(
    values_df=lc_coherent,
    s_df=S_df,
    earned_premium_df=ep_df,
    tolerance_pct=0.01,
    is_loss_cost=True,
)
print(report_coherent.to_string())

# COMMAND ----------

# Check the INCOHERENT data — should find violations
report_incoherent = check_coherence(
    values_df=lc_incoherent,
    s_df=S_df,
    earned_premium_df=ep_df,
    tolerance_pct=0.01,
    is_loss_cost=True,
)
print(report_incoherent.to_string())

# COMMAND ----------

print("Worst offending series:")
for s in report_incoherent.worst_series(n=5):
    print(f"  {s}")

violations_df = report_incoherent.to_dataframe()
print(f"\nViolations sample (first 5):")
display(violations_df.head(5))

# COMMAND ----------

# Use the high-level API
from insurance_reconcile import InsuranceReconciler

hierarchy = InsuranceHierarchy.uk_home()
reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())

report = reconciler.check_coherence(lc_incoherent, ep_df)
print(f"Is coherent: {report.is_coherent}")
print(f"Max discrepancy: {report.max_discrepancy_pct:.3f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Reconciliation

# COMMAND ----------

# Full reconciliation: converts LC -> claims, runs MinTrace, converts back
result = reconciler.reconcile(lc_incoherent, ep_df, run_diagnostics=True)

print(repr(result))
print(f"\nMethod used: {result.method}")
print(f"\nBefore reconciliation — max discrepancy: {result.coherence_before.max_discrepancy_pct:.3f}%")
print(f"After reconciliation  — max discrepancy: {result.coherence_after.max_discrepancy_pct:.3f}%")

# COMMAND ----------

print("\nReconciled loss costs (sample):")
display(result.reconciled_df.head(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Attribution Report

# COMMAND ----------

attr = result.attribution
print(attr.to_string())

# COMMAND ----------

attr_df = attr.to_dataframe()
display(attr_df)

# COMMAND ----------

print("Largest adjustments:")
for adj in attr.largest_adjustments(n=5):
    print(f"  {adj}")

if attr.portfolio_lc_before is not None:
    print(f"\nPortfolio LC: {attr.portfolio_lc_before:.5f} -> {attr.portfolio_lc_after:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Loss Ratio Reconciler (direct use)

# COMMAND ----------

from insurance_reconcile.reconcile.loss_ratio import LossRatioReconciler
import numpy as np

# Build mean EP dict for weighting
ep_mean = ep_df.mean()
ep_dict = ep_mean.to_dict()

lr_rec = LossRatioReconciler(earned_premium=ep_dict)

# Reconcile a single snapshot (1D)
series_names = list(S_df.index)
lc_snapshot = np.array([lc_incoherent[s].mean() for s in series_names])
lc_reconciled = lr_rec.reconcile(lc_snapshot, S_df.values, series_names=series_names)

print("Before / after reconciliation (single snapshot):")
for i, s in enumerate(series_names):
    diff_pct = 100 * (lc_reconciled[i] - lc_snapshot[i]) / (lc_snapshot[i] + 1e-10)
    print(f"  {s:<25}: {lc_snapshot[i]:.5f} -> {lc_reconciled[i]:.5f}  ({diff_pct:+.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Frequency × Severity Reconciliation

# COMMAND ----------

from insurance_reconcile.reconcile.freqsev import FreqSevReconciler

# Simulate freq/sev split
rng = np.random.default_rng(99)
base_lc = np.array([lc_incoherent[s].mean() for s in series_names])
freq_hat = base_lc * rng.uniform(0.8, 1.2, len(series_names)) * 15  # ~0.3-0.5 freq
sev_hat = base_lc / freq_hat

freq_rec = FreqSevReconciler(earned_premium=ep_dict)
freq_r, sev_r = freq_rec.reconcile(freq_hat, sev_hat, S_df.values, series_names=series_names)

print("Frequency reconciliation:")
for i, s in enumerate(series_names[:5]):
    print(f"  {s:<25}: freq {freq_hat[i]:.4f} -> {freq_r[i]:.4f}, "
          f"sev {sev_hat[i]:.2f} -> {sev_r[i]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. LossCostFrame Validation

# COMMAND ----------

from insurance_reconcile.simulate import make_hierarchy_dataframe
from insurance_reconcile.data import LossCostFrame

long_df = make_hierarchy_dataframe(lc_coherent, ep_df)
frame = LossCostFrame(long_df)

errors = frame.validate(raise_on_error=False)
print(f"Validation errors: {errors if errors else 'None — data is valid'}")
print(repr(frame))

coverage = frame.describe_coverage()
print("\nCoverage summary:")
display(coverage.sort_values("total_ep", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC - **PerilTree DSL**: declarative hierarchy spec that generates S_df automatically
# MAGIC - **CoherenceReport**: identifies which series violate hierarchy constraints, by how much
# MAGIC - **LossRatioReconciler**: handles the LC-as-rate problem via claims transform
# MAGIC - **PremiumWeightedMinTrace**: economically correct WLS with EP as weight
# MAGIC - **FreqSevReconciler**: multiplicative constraint via log-space
# MAGIC - **AttributionReport**: before/after adjustment tracking for sign-off
# MAGIC
# MAGIC The typical discrepancy before reconciliation in UK home pricing is 3–8% at the
# MAGIC cover level. MinTrace reduces this to machine precision while making the minimum
# MAGIC variance adjustment to all series.
