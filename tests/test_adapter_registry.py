"""
Tests for core/adapter_registry.py — adapter registration, lookup, listing.

Phase A: Extract Domain-Specific Code into Adapter
"""

import pytest
from core.domain_adapter import DomainAdapter
from core.adapter_registry import (
    register_adapter,
    get_adapter,
    list_adapters,
    get_default_adapter,
    clear_registry,
)
from core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before each test, re-register built-in adapters after."""
    clear_registry()
    yield
    clear_registry()
    # Re-register built-in adapters for other tests
    from adapters.software import SOFTWARE_ADAPTER
    from adapters.process import PROCESS_ADAPTER
    from adapters.api import API_ADAPTER
    register_adapter(SOFTWARE_ADAPTER)
    register_adapter(PROCESS_ADAPTER)
    register_adapter(API_ADAPTER)


# =============================================================================
# register_adapter
# =============================================================================

class TestRegisterAdapter:
    def test_register_new(self):
        adapter = DomainAdapter(name="test", version="1.0")
        register_adapter(adapter)
        assert "test" in list_adapters()

    def test_idempotent_reregistration(self):
        adapter = DomainAdapter(name="test", version="1.0")
        register_adapter(adapter)
        register_adapter(adapter)  # Same version — no error
        assert list_adapters().count("test") == 1

    def test_conflict_different_version(self):
        a1 = DomainAdapter(name="test", version="1.0")
        a2 = DomainAdapter(name="test", version="2.0")
        register_adapter(a1)
        with pytest.raises(ConfigurationError, match="already registered"):
            register_adapter(a2)

    def test_register_multiple(self):
        register_adapter(DomainAdapter(name="alpha", version="1.0"))
        register_adapter(DomainAdapter(name="beta", version="1.0"))
        assert "alpha" in list_adapters()
        assert "beta" in list_adapters()


# =============================================================================
# get_adapter
# =============================================================================

class TestGetAdapter:
    def test_get_registered(self):
        adapter = DomainAdapter(name="mydom", version="2.0")
        register_adapter(adapter)
        result = get_adapter("mydom")
        assert result.name == "mydom"
        assert result.version == "2.0"

    def test_get_unknown_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown domain adapter"):
            get_adapter("nonexistent")

    def test_error_lists_available(self):
        register_adapter(DomainAdapter(name="aaa", version="1.0"))
        register_adapter(DomainAdapter(name="bbb", version="1.0"))
        with pytest.raises(ConfigurationError, match="aaa") as exc:
            get_adapter("ccc")
        assert "bbb" in str(exc.value)


# =============================================================================
# list_adapters
# =============================================================================

class TestListAdapters:
    def test_empty(self):
        assert list_adapters() == []

    def test_sorted(self):
        register_adapter(DomainAdapter(name="zebra", version="1.0"))
        register_adapter(DomainAdapter(name="alpha", version="1.0"))
        assert list_adapters() == ["alpha", "zebra"]


# =============================================================================
# get_default_adapter
# =============================================================================

class TestGetDefaultAdapter:
    def test_returns_software(self):
        from adapters.software import SOFTWARE_ADAPTER
        register_adapter(SOFTWARE_ADAPTER)
        result = get_default_adapter()
        assert result.name == "software"

    def test_raises_without_software(self):
        with pytest.raises(ConfigurationError):
            get_default_adapter()


# =============================================================================
# clear_registry
# =============================================================================

class TestClearRegistry:
    def test_clear(self):
        register_adapter(DomainAdapter(name="test", version="1.0"))
        assert len(list_adapters()) == 1
        clear_registry()
        assert len(list_adapters()) == 0
