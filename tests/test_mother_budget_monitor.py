"""
Tests for mother/budget_monitor.py -- LEAF module.

Covers: ProviderBalance frozen dataclass, BudgetMonitor persistence,
balance checking (mocked), provider selection algorithm.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# httpx is optional
pytest.importorskip("httpx")

from mother.budget_monitor import ProviderBalance, BudgetMonitor
import time


class TestProviderBalance:
    def test_frozen(self):
        b = ProviderBalance(provider="openai", balance_usd=10.0)
        with pytest.raises(AttributeError):
            b.balance_usd = 20.0

    def test_defaults(self):
        b = ProviderBalance(provider="test")
        assert b.balance_usd == 0.0
        assert b.available is False


class TestBudgetMonitor:
    def test_persistence(self, tmp_path):
        storage = tmp_path / "balances.json"

        # Create and save
        monitor1 = BudgetMonitor(storage_path=storage)
        monitor1.balances["openai"] = ProviderBalance(
            provider="openai", balance_usd=50.0, available=True, last_checked=1234567.0
        )
        monitor1._save()

        # Load in new instance
        monitor2 = BudgetMonitor(storage_path=storage)
        balance = monitor2.balances.get("openai")

        assert balance is not None
        assert balance.balance_usd == 50.0
        assert balance.available is True

    @patch.object(BudgetMonitor, "get_balance")
    def test_choose_provider_highest_balance(self, mock_get_balance, tmp_path):
        """When no session pressure, choose provider with highest balance."""
        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        monitor.balances["openai"] = ProviderBalance(
            provider="openai", balance_usd=100.0, available=True, last_checked=time.time()
        )
        monitor.balances["grok"] = ProviderBalance(
            provider="grok", balance_usd=25.0, available=True, last_checked=time.time()
        )
        # Mock get_balance to not refresh
        mock_get_balance.return_value = None

        choice = monitor.choose_provider(
            available_providers={"openai": "key1", "grok": "key2"},
            session_cost=1.0,
            session_limit=5.0,
        )

        assert choice == "openai"

    @patch.object(BudgetMonitor, "get_balance")
    def test_choose_provider_nearing_limit(self, mock_get_balance, tmp_path):
        """When nearing session limit, choose cheapest provider."""
        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        monitor.balances["claude"] = ProviderBalance(
            provider="claude", balance_usd=100.0, available=True, last_checked=time.time()
        )
        monitor.balances["gemini"] = ProviderBalance(
            provider="gemini", balance_usd=50.0, available=True, last_checked=time.time()
        )
        mock_get_balance.return_value = None

        choice = monitor.choose_provider(
            available_providers={"claude": "key1", "gemini": "key2"},
            session_cost=4.5,  # 90% of 5.0 limit
            session_limit=5.0,
        )

        # Gemini is cheaper
        assert choice == "gemini"

    @patch.object(BudgetMonitor, "get_balance")
    def test_choose_provider_excludes_low_balance(self, mock_get_balance, tmp_path):
        """Providers with < $1 balance are excluded."""
        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        monitor.balances["openai"] = ProviderBalance(
            provider="openai", balance_usd=0.50, available=True, last_checked=time.time()
        )
        monitor.balances["grok"] = ProviderBalance(
            provider="grok", balance_usd=10.0, available=True, last_checked=time.time()
        )
        mock_get_balance.return_value = None

        choice = monitor.choose_provider(
            available_providers={"openai": "key1", "grok": "key2"},
            session_cost=1.0,
            session_limit=5.0,
        )

        assert choice == "grok"

    @patch.object(BudgetMonitor, "get_balance")
    def test_choose_provider_all_exhausted(self, mock_get_balance, tmp_path):
        """When all balances low, returns None."""
        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        monitor.balances["openai"] = ProviderBalance(
            provider="openai", balance_usd=0.20, available=True, last_checked=time.time()
        )
        mock_get_balance.return_value = None

        choice = monitor.choose_provider(
            available_providers={"openai": "key1"},
            session_cost=1.0,
            session_limit=5.0,
        )

        assert choice is None

    @patch("mother.budget_monitor.httpx.Client")
    def test_check_openai_balance_success(self, mock_client_cls, tmp_path):
        """OpenAI balance check with successful response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        balance = monitor._check_openai_balance("test_key")

        assert balance.provider == "openai"
        assert balance.available is True
        assert balance.balance_usd > 0

    @patch("mother.budget_monitor.httpx.Client")
    def test_check_openai_balance_failure(self, mock_client_cls, tmp_path):
        """OpenAI balance check with API error still returns available."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        monitor = BudgetMonitor(storage_path=tmp_path / "balances.json")
        balance = monitor._check_openai_balance("test_key")

        # Assume available even if balance check fails
        assert balance.available is True
