"""Tests for LossCostFrame validation."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.data.losscost import LossCostFrame


def make_valid_df(n=10):
    """Build a minimal valid LossCostFrame DataFrame."""
    ep = np.full(n, 5_000_000.0)
    lc = np.full(n, 0.025)
    claims = lc * ep
    return pd.DataFrame({
        "unique_id": ["Fire"] * n,
        "ds": pd.date_range("2022-01-01", periods=n, freq="MS"),
        "loss_cost": lc,
        "earned_premium": ep,
        "claims": claims,
    })


class TestLossCostFrameConstruction:
    def test_basic_creation(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        assert frame is not None

    def test_missing_column_raises(self):
        df = make_valid_df().drop(columns=["loss_cost"])
        with pytest.raises(ValueError, match="missing required columns"):
            LossCostFrame(df)

    def test_missing_earned_premium_raises(self):
        df = make_valid_df().drop(columns=["earned_premium"])
        with pytest.raises(ValueError, match="missing required columns"):
            LossCostFrame(df)

    def test_repr(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        r = repr(frame)
        assert "LossCostFrame" in r

    def test_df_property_is_copy(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        # Mutating df should not affect frame
        df["loss_cost"] = 999
        assert (frame.df["loss_cost"] != 999).all()


class TestValidation:
    def test_valid_passes(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        errors = frame.validate(raise_on_error=False)
        assert errors == []

    def test_negative_loss_cost_flagged(self):
        df = make_valid_df()
        df.loc[0, "loss_cost"] = -0.01
        frame = LossCostFrame(df)
        errors = frame.validate_non_negative()
        assert len(errors) > 0
        assert "negative loss_cost" in errors[0]

    def test_negative_ep_flagged(self):
        df = make_valid_df()
        df.loc[0, "earned_premium"] = -100
        frame = LossCostFrame(df)
        errors = frame.validate_non_negative()
        assert any("earned_premium" in e for e in errors)

    def test_lc_relationship_correct(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        errors = frame.validate_loss_cost_relationship()
        assert errors == []

    def test_lc_relationship_wrong(self):
        df = make_valid_df()
        df.loc[0, "claims"] = 999999  # Breaks claims = lc * ep
        frame = LossCostFrame(df)
        errors = frame.validate_loss_cost_relationship()
        assert len(errors) > 0

    def test_freq_sev_relationship_correct(self):
        df = make_valid_df()
        df["frequency"] = 0.05
        df["severity"] = 0.5
        df["loss_cost"] = df["frequency"] * df["severity"]
        frame = LossCostFrame(df)
        errors = frame.validate_freq_sev_relationship()
        assert errors == []

    def test_freq_sev_relationship_wrong(self):
        df = make_valid_df()
        # loss_cost = 0.025 (from make_valid_df), freq * sev = 0.05 * 0.6 = 0.030 != 0.025
        df["frequency"] = 0.05
        df["severity"] = 0.6
        frame = LossCostFrame(df)
        errors = frame.validate_freq_sev_relationship()
        assert len(errors) > 0

    def test_validate_raises_on_error(self):
        df = make_valid_df()
        df.loc[0, "loss_cost"] = -0.1
        frame = LossCostFrame(df)
        with pytest.raises(ValueError, match="validation failed"):
            frame.validate(raise_on_error=True)

    def test_validate_no_claims_col(self):
        df = make_valid_df().drop(columns=["claims"])
        frame = LossCostFrame(df)
        errors = frame.validate_loss_cost_relationship()
        assert errors == []  # No claims column = no relationship to check

    def test_tolerance_param(self):
        df = make_valid_df()
        df.loc[0, "claims"] = df.loc[0, "loss_cost"] * df.loc[0, "earned_premium"] * 1.0001
        frame_tight = LossCostFrame(df, tolerance=1e-10)
        errors_tight = frame_tight.validate_loss_cost_relationship()
        frame_loose = LossCostFrame(df, tolerance=1e-2)
        errors_loose = frame_loose.validate_loss_cost_relationship()
        assert len(errors_tight) > 0
        assert len(errors_loose) == 0


class TestTransforms:
    def test_to_claims_df(self):
        df = make_valid_df().drop(columns=["claims"])
        frame = LossCostFrame(df)
        claims_df = frame.to_claims_df()
        assert "claims" in claims_df.columns
        expected = df["loss_cost"] * df["earned_premium"]
        np.testing.assert_allclose(claims_df["claims"].values, expected.values)

    def test_from_claims_df(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        claims_df = frame.to_claims_df()
        # Modify claims slightly, recover loss costs
        claims_df["claims"] = claims_df["claims"] * 1.1
        new_frame = frame.from_claims_df(claims_df)
        # New loss cost = 1.1x original
        np.testing.assert_allclose(
            new_frame.df["loss_cost"].values,
            df["loss_cost"].values * 1.1,
            rtol=1e-6,
        )

    def test_from_claims_df_zero_ep(self):
        df = make_valid_df()
        df.loc[0, "earned_premium"] = 0.0
        frame = LossCostFrame(df)
        claims_df = frame.to_claims_df()
        new_frame = frame.from_claims_df(claims_df)
        # Zero EP should give zero loss cost, not NaN
        assert new_frame.df.loc[0, "loss_cost"] == 0.0

    def test_pivot_wide(self):
        # Multi-series data
        rows = []
        for series in ["Fire", "EoW"]:
            for i in range(3):
                rows.append({
                    "unique_id": series,
                    "ds": pd.Timestamp(f"2022-0{i+1}-01"),
                    "loss_cost": 0.02 + i * 0.001,
                    "earned_premium": 1e6,
                })
        df = pd.DataFrame(rows)
        frame = LossCostFrame(df)
        wide = frame.pivot_wide()
        assert "Fire" in wide.columns
        assert "EoW" in wide.columns
        assert wide.shape == (3, 2)

    def test_describe_coverage(self):
        df = make_valid_df()
        frame = LossCostFrame(df)
        coverage = frame.describe_coverage()
        assert "unique_id" in coverage.columns
        assert "total_ep" in coverage.columns
        assert "ep_weighted_lc" in coverage.columns
