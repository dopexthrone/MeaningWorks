"""
Motherlabs Domain Adapters — pluggable domain configurations.

Importing this package auto-registers all built-in adapters.
"""

from adapters.software import SOFTWARE_ADAPTER  # noqa: F401
from adapters.process import PROCESS_ADAPTER  # noqa: F401
from adapters.api import API_ADAPTER  # noqa: F401
from adapters.agent_system import AGENT_SYSTEM_ADAPTER  # noqa: F401

# Auto-register built-in adapters
from core.adapter_registry import register_adapter as _register

_register(SOFTWARE_ADAPTER)
_register(PROCESS_ADAPTER)
_register(API_ADAPTER)
_register(AGENT_SYSTEM_ADAPTER)
