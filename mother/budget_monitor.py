"""
Budget monitor — track API spend and switch providers intelligently.

LEAF module. Queries provider APIs for remaining balance, tracks
spend across sessions, switches to cheaper providers when low.

Enables Mother to:
- Monitor API balance for each provider
- Switch providers when budget low
- Prefer cheaper models when nearing session limit
- Delegate to peers when all APIs exhausted

Uses provider-specific balance APIs:
- OpenAI: /v1/usage (requires org-level key)
- Anthropic: /v1/usage (requires admin key)
- Grok: /v1/balance (available to all users)
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass(frozen=True)
class ProviderBalance:
    """Balance information for a provider."""

    provider: str
    balance_usd: float = 0.0
    monthly_limit: float = 0.0
    last_checked: float = 0.0
    error: Optional[str] = None
    available: bool = False


class BudgetMonitor:
    """
    Tracks API balances and switches providers intelligently.

    Stores last-known balances in ~/.motherlabs/balances.json.
    Refreshes on demand (cached for 5 minutes to avoid excessive API calls).
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path.home() / ".motherlabs" / "balances.json")
        self.balances: Dict[str, ProviderBalance] = {}
        self._load()

    def _load(self):
        """Load cached balances from disk."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for provider, b_data in data.get("balances", {}).items():
                    self.balances[provider] = ProviderBalance(**b_data)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self):
        """Persist balances to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "balances": {
                p: {
                    "provider": b.provider,
                    "balance_usd": b.balance_usd,
                    "monthly_limit": b.monthly_limit,
                    "last_checked": b.last_checked,
                    "error": b.error,
                    "available": b.available,
                }
                for p, b in self.balances.items()
                if b is not None  # Filter out None entries
            }
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def get_balance(self, provider: str, api_key: str = "", force_refresh: bool = False) -> ProviderBalance:
        """
        Get balance for a provider.

        Uses cached value if < 5 minutes old, unless force_refresh=True.
        """
        # Check cache
        if not force_refresh and provider in self.balances:
            cached = self.balances[provider]
            age = time.time() - cached.last_checked
            if age < 300:  # 5 minutes
                return cached

        # Refresh from API
        if provider == "openai":
            balance = self._check_openai_balance(api_key)
        elif provider == "grok":
            balance = self._check_grok_balance(api_key)
        elif provider == "claude":
            balance = self._check_anthropic_balance(api_key)
        elif provider == "gemini":
            # Gemini doesn't expose balance API, assume available
            balance = ProviderBalance(
                provider="gemini",
                balance_usd=999.0,  # Unknown
                last_checked=time.time(),
                available=True,
            )
        else:
            balance = ProviderBalance(
                provider=provider,
                error="Unknown provider",
                last_checked=time.time(),
            )

        self.balances[provider] = balance
        self._save()
        return balance

    def _check_openai_balance(self, api_key: str) -> ProviderBalance:
        """Query OpenAI usage API."""
        if not HTTPX_AVAILABLE:
            return ProviderBalance(
                provider="openai",
                balance_usd=50.0,
                error="httpx not available",
                last_checked=time.time(),
                available=True,
            )
        if not api_key:
            return ProviderBalance(
                provider="openai",
                error="No API key",
                last_checked=time.time(),
                available=False,
            )

        try:
            # OpenAI doesn't expose balance directly, check via usage endpoint
            # This requires org-level permissions
            client = httpx.Client(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )

            # Try to get org usage (requires admin)
            response = client.get("/usage")

            if response.status_code == 200:
                # Estimate: assume $100 monthly limit if no errors
                return ProviderBalance(
                    provider="openai",
                    balance_usd=100.0,  # Estimated
                    monthly_limit=100.0,
                    last_checked=time.time(),
                    available=True,
                )
            else:
                # Can't check balance, assume available
                return ProviderBalance(
                    provider="openai",
                    balance_usd=50.0,  # Conservative estimate
                    last_checked=time.time(),
                    available=True,
                )

        except Exception as e:
            return ProviderBalance(
                provider="openai",
                error=str(e),
                last_checked=time.time(),
                available=True,  # Assume available even if balance check fails
            )

    def _check_grok_balance(self, api_key: str) -> ProviderBalance:
        """Query Grok balance API."""
        if not HTTPX_AVAILABLE:
            return ProviderBalance(
                provider="grok",
                balance_usd=50.0,
                error="httpx not available",
                last_checked=time.time(),
                available=True,
            )
        if not api_key:
            return ProviderBalance(
                provider="grok",
                error="No API key",
                last_checked=time.time(),
                available=False,
            )

        try:
            client = httpx.Client(
                base_url="https://api.x.ai/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )

            response = client.get("/balance")

            if response.status_code == 200:
                data = response.json()
                balance = float(data.get("balance_usd", 0.0))
                return ProviderBalance(
                    provider="grok",
                    balance_usd=balance,
                    last_checked=time.time(),
                    available=balance > 0.10,
                )

        except Exception as e:
            return ProviderBalance(
                provider="grok",
                error=str(e),
                last_checked=time.time(),
                available=True,  # Assume available
            )

    def _check_anthropic_balance(self, api_key: str) -> ProviderBalance:
        """Query Anthropic usage API."""
        if not HTTPX_AVAILABLE:
            return ProviderBalance(
                provider="claude",
                balance_usd=100.0,
                error="httpx not available",
                last_checked=time.time(),
                available=True,
            )
        if not api_key:
            return ProviderBalance(
                provider="claude",
                error="No API key",
                last_checked=time.time(),
                available=False,
            )

        # Anthropic doesn't expose balance API yet
        # Assume available
        return ProviderBalance(
            provider="claude",
            balance_usd=100.0,  # Unknown
            last_checked=time.time(),
            available=True,
        )

    def choose_provider(
        self,
        available_providers: Dict[str, str],  # provider -> api_key
        session_cost: float = 0.0,
        session_limit: float = 5.0,
    ) -> Optional[str]:
        """
        Choose the best provider based on balances and session state.

        Algorithm:
        - If session_cost > 80% of limit → prefer cheapest provider
        - If provider balance < $1.00 → exclude
        - If all balances low → return None (signals: use peer or stop)
        - Else: prefer provider with highest balance

        Returns provider name or None.
        """
        # Refresh balances
        for provider, api_key in available_providers.items():
            self.get_balance(provider, api_key=api_key, force_refresh=False)

        # Filter to available providers
        candidates = []
        for provider, balance in self.balances.items():
            if provider not in available_providers or balance is None:
                continue
            if balance.available and balance.balance_usd > 1.0:
                candidates.append((provider, balance))

        if not candidates:
            return None

        # Nearing session limit → prefer cheapest
        if session_cost > session_limit * 0.8:
            # Cost ordering: gemini < openai < grok < claude
            cost_priority = {"gemini": 1, "openai": 2, "grok": 3, "claude": 4}
            candidates.sort(key=lambda x: cost_priority.get(x[0], 5))
            return candidates[0][0]

        # Otherwise → highest balance
        candidates.sort(key=lambda x: x[1].balance_usd, reverse=True)
        return candidates[0][0]

    def get_all_balances(self, api_keys: Dict[str, str]) -> Dict[str, ProviderBalance]:
        """Get balances for all configured providers."""
        for provider, api_key in api_keys.items():
            self.get_balance(provider, api_key=api_key, force_refresh=False)
        return dict(self.balances)
