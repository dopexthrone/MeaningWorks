"""
Phase 21: Cost API endpoint tests.

Tests for cost-related API additions in api/main.py and api/models.py.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, set_engine
from core.engine import MotherlabsEngine
from core.llm import MockClient


@pytest.fixture(autouse=True)
def _setup_engine():
    """Inject MockClient engine for all tests."""
    engine = MotherlabsEngine(llm_client=MockClient())
    set_engine(engine)
    yield
    set_engine(None)


client = TestClient(app)


class TestHealthIncludesCost:

    def test_health_includes_cost(self):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "recent_total_cost_usd" in data


class TestMetricsIncludesCost:

    def test_metrics_includes_cost_section(self):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "cost" in data
        cost = data["cost"]
        assert "total_input_tokens" in cost
        assert "total_output_tokens" in cost
        assert "session_cost_usd" in cost


class TestEstimateCostEndpoint:

    def test_estimate_cost_endpoint(self):
        resp = client.post("/v1/estimate-cost", json={
            "description": "Build a booking system for a tattoo studio with artists and clients",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "estimated_tokens" in data
        assert "estimated_cost_usd" in data
        assert "model" in data
        assert "breakdown" in data
        assert data["estimated_tokens"] > 0

    def test_estimate_cost_known_model(self):
        resp = client.post("/v1/estimate-cost", json={
            "description": "Build a simple to-do app",
            "provider": "gemini",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "gemini" in data["model"]
        assert data["estimated_cost_usd"] >= 0

    def test_estimate_cost_with_explicit_model(self):
        resp = client.post("/v1/estimate-cost", json={
            "description": "Build a CRM system",
            "model": "claude-opus-4-20250514",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "claude-opus-4-20250514"
        # Opus is expensive — cost should be > 0 for any reasonable description
        assert data["estimated_cost_usd"] > 0
