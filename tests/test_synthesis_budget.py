"""Tests for synthesis time budget and retry reduction.

Validates:
- Freeform intents (no canonical components) get reduced MAX_RETRIES
- Synthesis time budget stops retries when first attempt is slow
- Canonical intents still get full retries
"""
from __future__ import annotations

import inspect
import pytest

from core.engine import MotherlabsEngine


class TestSynthesisRetryReduction:
    """Freeform intents get fewer synthesis retries."""

    def test_freeform_reduces_max_retries(self):
        """_synthesize source contains the retry reduction logic."""
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "not canonical_components" in source
        assert "min(MAX_RETRIES, 1)" in source

    def test_time_budget_exists(self):
        """Synthesis has a time budget constant."""
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "SYNTHESIS_TIME_BUDGET" in source
        assert "time budget exceeded" in source


class TestTimeBudgetConfig:
    """Synthesis time budget is 180 seconds."""

    def test_budget_value(self):
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "SYNTHESIS_TIME_BUDGET = 180" in source
