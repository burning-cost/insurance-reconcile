"""Tests for _compat module."""

import pytest
from insurance_reconcile._compat import (
    HAS_HIERARCHICALFORECAST,
    require_hierarchicalforecast,
    get_HierarchicalReconciliation,
    get_MinTrace,
    get_aggregate,
)


class TestCompat:
    def test_has_hierarchicalforecast_is_bool(self):
        assert isinstance(HAS_HIERARCHICALFORECAST, bool)

    def test_require_raises_when_not_installed(self):
        if HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast is installed")
        with pytest.raises(ImportError, match="hierarchicalforecast"):
            require_hierarchicalforecast("test feature")

    def test_require_passes_when_installed(self):
        if not HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast not installed")
        # Should not raise
        require_hierarchicalforecast("test feature")

    def test_get_mintrace_raises_when_not_installed(self):
        if HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast is installed")
        with pytest.raises(ImportError):
            get_MinTrace()

    def test_get_mintrace_returns_class_when_installed(self):
        if not HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast not installed")
        MinTrace = get_MinTrace()
        assert MinTrace is not None

    def test_get_aggregate_raises_when_not_installed(self):
        if HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast is installed")
        with pytest.raises(ImportError):
            get_aggregate()

    def test_get_hierarchical_reconciliation_raises_when_not_installed(self):
        if HAS_HIERARCHICALFORECAST:
            pytest.skip("hierarchicalforecast is installed")
        with pytest.raises(ImportError):
            get_HierarchicalReconciliation()
