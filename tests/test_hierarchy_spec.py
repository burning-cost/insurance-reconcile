"""Tests for hierarchy specification DSL."""

import pytest
import numpy as np
import pandas as pd

from insurance_reconcile.hierarchy.spec import (
    PerilTree,
    GeographicHierarchy,
    InsuranceHierarchy,
)


# ===== PerilTree =====

class TestPerilTree:
    def test_basic_construction(self):
        tree = PerilTree(
            portfolio="Home",
            covers={"Buildings": ["Fire", "EoW"], "Contents": ["Theft"]},
        )
        assert tree.portfolio == "Home"
        assert tree.all_covers == ["Buildings", "Contents"]

    def test_all_perils(self):
        tree = PerilTree.uk_home()
        perils = tree.all_perils
        assert "Fire" in perils
        assert "EoW" in perils
        assert "Subsidence" in perils
        assert len(perils) == 7

    def test_all_nodes_order(self):
        tree = PerilTree(
            portfolio="P",
            covers={"A": ["x", "y"], "B": ["z"]},
        )
        nodes = tree.all_nodes()
        # Portfolio first, then covers, then perils
        assert nodes[0] == "P"
        assert "A" in nodes
        assert "B" in nodes
        assert "x" in nodes
        assert len(nodes) == 6

    def test_cover_for_peril(self):
        tree = PerilTree.uk_home()
        assert tree.cover_for_peril("Fire") == "Buildings"
        assert tree.cover_for_peril("Theft") == "Contents"

    def test_cover_for_peril_missing(self):
        tree = PerilTree.uk_home()
        with pytest.raises(KeyError):
            tree.cover_for_peril("NonExistent")

    def test_empty_portfolio_raises(self):
        with pytest.raises(ValueError, match="portfolio name"):
            PerilTree(portfolio="", covers={"A": ["x"]})

    def test_empty_covers_raises(self):
        with pytest.raises(ValueError, match="covers"):
            PerilTree(portfolio="P", covers={})

    def test_empty_peril_list_raises(self):
        with pytest.raises(ValueError, match="no perils"):
            PerilTree(portfolio="P", covers={"A": []})

    def test_to_spec(self):
        tree = PerilTree.uk_home()
        spec = tree.to_spec()
        assert len(spec) == 3
        assert spec[0] == ["Home"]
        assert "cover" in spec[1]
        assert "peril" in spec[2]

    def test_to_dataframe(self):
        tree = PerilTree.uk_home()
        df = tree.to_dataframe()
        assert "cover" in df.columns
        assert "peril" in df.columns
        assert len(df) == len(tree.all_perils)
        # Each row should have the portfolio name in the portfolio column
        assert (df["Home"] == "Home").all()

    def test_from_dict(self):
        d = {"portfolio": "Motor", "covers": {"OD": ["Collision", "Fire"]}}
        tree = PerilTree.from_dict(d)
        assert tree.portfolio == "Motor"
        assert "Collision" in tree.all_perils

    def test_uk_home_factory(self):
        tree = PerilTree.uk_home()
        assert tree.portfolio == "Home"
        assert "Buildings" in tree.all_covers
        assert "Contents" in tree.all_covers

    def test_uk_motor_factory(self):
        tree = PerilTree.uk_motor()
        assert tree.portfolio == "Motor"
        perils = tree.all_perils
        assert "Collision" in perils

    def test_all_covers_property(self):
        tree = PerilTree(
            portfolio="P",
            covers={"A": ["x"], "B": ["y"], "C": ["z"]},
        )
        assert tree.all_covers == ["A", "B", "C"]

    def test_peril_count_uk_home(self):
        tree = PerilTree.uk_home()
        assert len(tree.all_perils) == 7

    def test_cover_count_uk_home(self):
        tree = PerilTree.uk_home()
        assert len(tree.all_covers) == 2


# ===== GeographicHierarchy =====

class TestGeographicHierarchy:
    def test_basic_construction(self):
        geo = GeographicHierarchy(levels=["postcode", "area", "region"])
        assert geo.bottom_level == "postcode"
        assert geo.top_level == "region"

    def test_single_level_raises(self):
        with pytest.raises(ValueError, match="2 levels"):
            GeographicHierarchy(levels=["national"])

    def test_uk_postcode_factory(self):
        geo = GeographicHierarchy.uk_postcode()
        assert "postcode_sector" in geo.levels
        assert "region" in geo.levels

    def test_to_spec(self):
        geo = GeographicHierarchy(levels=["sector", "area", "region"])
        spec = geo.to_spec()
        # spec[0] = just the coarsest level; spec[-1] = all levels (finest -> coarsest)
        assert spec[0] == ["region"]
        assert spec[-1] == ["sector", "area", "region"]

    def test_national_name_default(self):
        geo = GeographicHierarchy(levels=["a", "b"])
        assert geo.national_name == "National"


# ===== InsuranceHierarchy =====

class TestInsuranceHierarchy:
    def test_basic_construction(self):
        hierarchy = InsuranceHierarchy(peril_tree=PerilTree.uk_home())
        assert hierarchy.has_geography is False

    def test_with_geography(self):
        hierarchy = InsuranceHierarchy(
            peril_tree=PerilTree.uk_home(),
            geographic_hierarchy=GeographicHierarchy.uk_postcode(),
        )
        assert hierarchy.has_geography is True

    def test_bottom_level_perils(self):
        hierarchy = InsuranceHierarchy.uk_home()
        perils = hierarchy.bottom_level_perils
        assert "Fire" in perils
        assert len(perils) == 7

    def test_n_bottom_series_no_geo(self):
        hierarchy = InsuranceHierarchy.uk_home()
        assert hierarchy.n_bottom_series(n_geographies=1) == 7

    def test_n_bottom_series_with_geo(self):
        hierarchy = InsuranceHierarchy.uk_home_with_geography()
        assert hierarchy.n_bottom_series(n_geographies=5) == 35

    def test_uk_home_factory(self):
        h = InsuranceHierarchy.uk_home()
        assert h.name == "UK Home"
        assert h.peril_tree.portfolio == "Home"

    def test_uk_home_with_geography_factory(self):
        h = InsuranceHierarchy.uk_home_with_geography()
        assert h.has_geography

    def test_from_dict(self):
        d = {
            "peril_tree": {
                "portfolio": "Motor",
                "covers": {"OD": ["Fire", "Theft"]},
            },
            "name": "Motor Test",
        }
        h = InsuranceHierarchy.from_dict(d)
        assert h.name == "Motor Test"
        assert h.peril_tree.portfolio == "Motor"

    def test_to_aggregate_spec_no_geo(self):
        h = InsuranceHierarchy.uk_home()
        spec = h.to_aggregate_spec()
        assert len(spec) == 3
        assert spec[0] == ["Home"]

    def test_metadata_default(self):
        h = InsuranceHierarchy.uk_home()
        assert isinstance(h.metadata, dict)
