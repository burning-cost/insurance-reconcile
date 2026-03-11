"""Tests for S_df and tags construction."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.hierarchy.spec import PerilTree, InsuranceHierarchy
from insurance_reconcile.hierarchy.builder import build_S_df, HierarchyBuilder


class TestBuildSdf:
    def setup_method(self):
        self.tree = PerilTree(
            portfolio="Home",
            covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
        )
        self.bottom_series = ["Fire", "EoW", "Theft"]

    def test_s_df_shape(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        # 3 bottom + 2 covers + 1 portfolio = 6 rows, 3 columns
        assert S_df.shape == (6, 3)

    def test_s_df_index(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        assert "Home" in S_df.index
        assert "Buildings" in S_df.index
        assert "Contents" in S_df.index
        assert "Fire" in S_df.index

    def test_s_df_columns(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        assert list(S_df.columns) == ["Fire", "EoW", "Theft"]

    def test_portfolio_row_all_ones(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        assert (S_df.loc["Home"] == 1.0).all()

    def test_cover_row_correct(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        buildings_row = S_df.loc["Buildings"]
        assert buildings_row["Fire"] == 1.0
        assert buildings_row["EoW"] == 1.0
        assert buildings_row["Theft"] == 0.0

    def test_contents_row_correct(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        contents_row = S_df.loc["Contents"]
        assert contents_row["Fire"] == 0.0
        assert contents_row["EoW"] == 0.0
        assert contents_row["Theft"] == 1.0

    def test_bottom_rows_are_identity(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        for s in self.bottom_series:
            row = S_df.loc[s]
            assert row[s] == 1.0
            for other in self.bottom_series:
                if other != s:
                    assert row[other] == 0.0

    def test_tags_structure(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        assert "Portfolio" in tags
        assert "Cover" in tags
        assert "Peril" in tags
        assert "Home" in tags["Portfolio"]
        assert "Buildings" in tags["Cover"]
        assert "Fire" in tags["Peril"]

    def test_wrong_bottom_series_length(self):
        with pytest.raises(ValueError, match="peril_tree has 2 perils"):
            build_S_df(
                PerilTree(
                    portfolio="P", covers={"A": ["x", "y"]}
                ),
                ["x"],  # Missing y
            )

    def test_s_df_binary(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        # All values should be 0 or 1
        unique_vals = set(S_df.values.flatten())
        assert unique_vals.issubset({0.0, 1.0})

    def test_s_df_values_dtype(self):
        S_df, tags = build_S_df(self.tree, self.bottom_series)
        assert S_df.dtypes.unique()[0] == float

    def test_uk_home_full(self):
        tree = PerilTree.uk_home()
        perils = tree.all_perils
        S_df, tags = build_S_df(tree, perils)
        # 7 perils + 2 covers + 1 portfolio = 10 rows
        assert S_df.shape[0] == 10
        assert S_df.shape[1] == 7


class TestHierarchyBuilder:
    def setup_method(self):
        self.hierarchy = InsuranceHierarchy.uk_home()
        self.builder = HierarchyBuilder(self.hierarchy)

    def test_build_s_df(self):
        bottom = self.hierarchy.peril_tree.all_perils
        S_df, tags = self.builder.build_S_df(bottom)
        assert S_df.shape[1] == 7

    def test_build_y_hat_df(self):
        ds = pd.date_range("2022-01-01", periods=3, freq="MS")
        forecasts = {
            "Fire": pd.DataFrame({"ds": ds, "GLM": [0.02, 0.021, 0.022]}),
            "EoW": pd.DataFrame({"ds": ds, "GLM": [0.03, 0.031, 0.032]}),
        }
        y_hat_df = self.builder.build_Y_hat_df(forecasts, model_name="GLM")
        assert "unique_id" in y_hat_df.columns
        assert "ds" in y_hat_df.columns
        assert "GLM" in y_hat_df.columns
        assert len(y_hat_df) == 6  # 2 series x 3 periods
