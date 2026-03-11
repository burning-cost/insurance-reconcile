"""
Insurance hierarchy specification DSL.

Defines the declarative API for specifying insurance pricing hierarchies.
The core problem is that insurers maintain coherent forecasts across multiple
aggregation levels simultaneously — portfolio, cover, peril, rating cell,
geography. Building the summing matrix S by hand is error-prone; this module
provides a type-safe DSL that produces S_df in the format expected by
hierarchicalforecast.

Design choice: separate PerilTree (cover -> peril) from GeographicHierarchy
(geography levels) because these are orthogonal concerns that are often
combined via a crossed hierarchy. InsuranceHierarchy composes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PerilTree",
    "GeographicHierarchy",
    "InsuranceHierarchy",
]


@dataclass
class PerilTree:
    """
    A cover -> peril tree for a single line of business.

    The portfolio level is implicitly added as the root. Covers aggregate
    perils, and the portfolio aggregates covers.

    Parameters
    ----------
    portfolio:
        Name for the top-level aggregate (e.g. 'Home', 'Motor').
    covers:
        Mapping from cover name to list of peril names.
        E.g. {'Buildings': ['Fire', 'EoW', 'Subsidence'],
               'Contents': ['Theft', 'AD']}.

    Example
    -------
    >>> tree = PerilTree(
    ...     portfolio='Home',
    ...     covers={
    ...         'Buildings': ['Fire', 'EoW', 'Subsidence', 'Flood', 'Storm'],
    ...         'Contents':  ['Theft', 'AD'],
    ...     },
    ... )
    >>> tree.all_nodes()
    ['Home', 'Buildings', 'Contents', 'Fire', 'EoW', 'Subsidence', 'Flood',
     'Storm', 'Theft', 'AD']
    """

    portfolio: str
    covers: dict[str, list[str]]

    def __post_init__(self) -> None:
        if not self.portfolio:
            raise ValueError("portfolio name must not be empty")
        if not self.covers:
            raise ValueError("covers must contain at least one entry")
        for cover, perils in self.covers.items():
            if not perils:
                raise ValueError(f"Cover '{cover}' has no perils")

    @property
    def all_perils(self) -> list[str]:
        """Return a flat list of all bottom-level peril names."""
        return [p for perils in self.covers.values() for p in perils]

    @property
    def all_covers(self) -> list[str]:
        """Return cover names in definition order."""
        return list(self.covers.keys())

    def all_nodes(self) -> list[str]:
        """Return all node names: portfolio, covers, perils (in order)."""
        nodes = [self.portfolio] + self.all_covers + self.all_perils
        return nodes

    def cover_for_peril(self, peril: str) -> str:
        """Return the cover that contains the given peril."""
        for cover, perils in self.covers.items():
            if peril in perils:
                return cover
        raise KeyError(f"Peril '{peril}' not found in tree")

    def to_spec(self) -> list[list[str]]:
        """
        Return the aggregate() spec format expected by hierarchicalforecast.

        Each entry is a list of column names representing one level of the
        hierarchy, from top to bottom. For a two-level cover->peril tree:
        [['portfolio'], ['portfolio', 'cover'], ['portfolio', 'cover', 'peril']]
        """
        return [
            [self.portfolio],
            [self.portfolio, "cover"],
            [self.portfolio, "cover", "peril"],
        ]

    def to_dataframe(self, perils: list[str] | None = None) -> pd.DataFrame:
        """
        Return a DataFrame with columns [portfolio, cover, peril] suitable
        for passing to hierarchicalforecast's aggregate() function.

        Parameters
        ----------
        perils:
            Optional list to specify ordering; must match self.all_perils.
            If None, uses definition order.
        """
        if perils is None:
            perils = self.all_perils

        rows = []
        for peril in perils:
            cover = self.cover_for_peril(peril)
            rows.append({self.portfolio: self.portfolio, "cover": cover, "peril": peril})
        return pd.DataFrame(rows)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerilTree":
        """Construct from a plain dictionary (e.g. loaded from YAML)."""
        return cls(portfolio=d["portfolio"], covers=d["covers"])

    @classmethod
    def uk_home(cls) -> "PerilTree":
        """Standard UK home insurance peril tree."""
        return cls(
            portfolio="Home",
            covers={
                "Buildings": ["Fire", "EoW", "Subsidence", "Flood", "Storm"],
                "Contents": ["Theft", "AccidentalDamage"],
            },
        )

    @classmethod
    def uk_motor(cls) -> "PerilTree":
        """Standard UK motor insurance peril tree."""
        return cls(
            portfolio="Motor",
            covers={
                "OwnDamage": ["Collision", "Fire", "Theft", "Weather"],
                "ThirdPartyInjury": ["Liability"],
                "Windscreen": ["ChipRepair", "FullReplacement"],
            },
        )


@dataclass
class GeographicHierarchy:
    """
    A geographic aggregation hierarchy.

    Models the UK postcode geography: postcode sector -> district -> area ->
    region -> national. Any subset of these levels can be used.

    Parameters
    ----------
    levels:
        Ordered list of geographic level names from finest to coarsest.
        E.g. ['postcode_sector', 'postcode_area', 'region', 'national'].
    national_name:
        Label for the top (national) level. Defaults to 'National'.

    Example
    -------
    >>> geo = GeographicHierarchy(
    ...     levels=['postcode_sector', 'postcode_area', 'region'],
    ...     national_name='National',
    ... )
    """

    levels: list[str]
    national_name: str = "National"

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError("Geographic hierarchy requires at least 2 levels")

    @property
    def bottom_level(self) -> str:
        return self.levels[0]

    @property
    def top_level(self) -> str:
        return self.levels[-1]

    def to_spec(self) -> list[list[str]]:
        """
        Return the aggregate() spec from finest to all-levels combined.
        Coarsest level (national) is listed first.
        """
        spec = []
        for i in range(len(self.levels)):
            spec.append(self.levels[i:])
        return list(reversed(spec))

    @classmethod
    def uk_postcode(cls) -> "GeographicHierarchy":
        """Standard UK postcode geography."""
        return cls(
            levels=["postcode_sector", "postcode_area", "region"],
            national_name="National",
        )


@dataclass
class InsuranceHierarchy:
    """
    A complete insurance pricing hierarchy combining peril and geographic
    dimensions.

    This is the main entry point for defining hierarchy structures. The
    peril tree and geographic hierarchy are orthogonal dimensions that
    create a crossed (grouped) hierarchy when combined.

    Parameters
    ----------
    peril_tree:
        The peril/cover tree for this line of business.
    geographic_hierarchy:
        Optional geographic dimension. If None, the hierarchy is purely
        peril-based.
    name:
        Optional descriptive name.

    Notes
    -----
    When a geographic hierarchy is included, the bottom-level nodes are
    (peril, geography) pairs. The summing matrix S encodes both peril
    aggregation and geographic aggregation.

    Example
    -------
    >>> hierarchy = InsuranceHierarchy(
    ...     peril_tree=PerilTree.uk_home(),
    ...     geographic_hierarchy=GeographicHierarchy.uk_postcode(),
    ...     name='UK Home',
    ... )
    """

    peril_tree: PerilTree
    geographic_hierarchy: GeographicHierarchy | None = None
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bottom_level_perils(self) -> list[str]:
        """Peril names at the bottom of the tree."""
        return self.peril_tree.all_perils

    @property
    def has_geography(self) -> bool:
        return self.geographic_hierarchy is not None

    def n_bottom_series(self, n_geographies: int = 1) -> int:
        """Number of bottom-level series given a geographic count."""
        if self.geographic_hierarchy is not None:
            return len(self.bottom_level_perils) * n_geographies
        return len(self.bottom_level_perils)

    def to_aggregate_spec(self) -> list[list[str]]:
        """
        Return the spec for hierarchicalforecast's aggregate() function.
        This describes the column groupings in the input DataFrame.
        """
        if self.geographic_hierarchy is None:
            return self.peril_tree.to_spec()

        # Crossed: peril x geography
        geo_levels = self.geographic_hierarchy.levels
        peril_cols = [self.peril_tree.portfolio, "cover", "peril"]

        spec = []
        # Pure peril aggregations (all geographies combined)
        spec.append([self.peril_tree.portfolio])
        spec.append([self.peril_tree.portfolio, "cover"])
        # Peril x geography combinations
        for i in range(len(geo_levels)):
            spec.append(peril_cols + geo_levels[i:])
        return spec

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InsuranceHierarchy":
        """Construct from a plain dictionary."""
        peril_tree = PerilTree.from_dict(d["peril_tree"])
        geo = None
        if "geographic_hierarchy" in d:
            geo = GeographicHierarchy(**d["geographic_hierarchy"])
        return cls(
            peril_tree=peril_tree,
            geographic_hierarchy=geo,
            name=d.get("name", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InsuranceHierarchy":
        """Load from a YAML file."""
        import yaml  # type: ignore

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def uk_home(cls) -> "InsuranceHierarchy":
        """Standard UK home insurance hierarchy (peril only)."""
        return cls(
            peril_tree=PerilTree.uk_home(),
            name="UK Home",
        )

    @classmethod
    def uk_home_with_geography(cls) -> "InsuranceHierarchy":
        """UK home insurance with postcode geographic dimension."""
        return cls(
            peril_tree=PerilTree.uk_home(),
            geographic_hierarchy=GeographicHierarchy.uk_postcode(),
            name="UK Home (Geographic)",
        )
