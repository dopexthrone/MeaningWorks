"""
Motherlabs Adapter Registry — registration, lookup, listing of domain adapters.

Phase A: Extract Domain-Specific Code into Adapter

Provides a global registry for DomainAdapter instances.
Built-in adapters auto-register at import time.
"""

from typing import Dict, List, Optional

from core.domain_adapter import DomainAdapter
from core.exceptions import ConfigurationError


# Global registry
_REGISTRY: Dict[str, DomainAdapter] = {}


def register_adapter(adapter: DomainAdapter) -> None:
    """Register a domain adapter.

    Args:
        adapter: DomainAdapter instance to register

    Raises:
        ConfigurationError: If adapter with same name already registered
    """
    if adapter.name in _REGISTRY:
        existing = _REGISTRY[adapter.name]
        if existing.version == adapter.version:
            return  # Idempotent re-registration
        raise ConfigurationError(
            f"Adapter '{adapter.name}' already registered (v{existing.version}). "
            f"Cannot register v{adapter.version}.",
            config_key="adapter_name",
        )
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> DomainAdapter:
    """Get a registered domain adapter by name.

    Args:
        name: Adapter name (e.g., "software", "process")

    Returns:
        DomainAdapter instance

    Raises:
        ConfigurationError: If adapter not found
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise ConfigurationError(
            f"Unknown domain adapter '{name}'. Available: {available}",
            config_key="domain",
        )
    return _REGISTRY[name]


def list_adapters() -> List[str]:
    """List all registered adapter names.

    Returns:
        Sorted list of adapter names
    """
    return sorted(_REGISTRY.keys())


def get_default_adapter() -> DomainAdapter:
    """Get the default adapter (software).

    Returns:
        The software DomainAdapter

    Raises:
        ConfigurationError: If software adapter not registered
    """
    return get_adapter("software")


def clear_registry() -> None:
    """Clear the adapter registry. Used in tests."""
    _REGISTRY.clear()
