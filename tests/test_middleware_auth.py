"""
Tests for APIKeyMiddleware with auth integration.

~12 tests covering:
- No auth (default behavior preserved)
- 401 for missing/invalid/revoked keys
- 429 for rate limit exceeded
- 402 for budget exceeded
- Successful auth with rate-limit headers
- Health/domains endpoints bypass auth
"""

import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient
from fastapi import FastAPI, Request

from motherlabs_platform.middleware import APIKeyMiddleware
from motherlabs_platform.auth import KeyStore, RateLimiter


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_mw.db")


@pytest.fixture
def store(db_path):
    return KeyStore(db_path=db_path)


@pytest.fixture
def limiter(store):
    return RateLimiter(store, window_seconds=3600)


def _make_app(require_key=False, key_store=None, rate_limiter=None):
    """Create a minimal FastAPI app with APIKeyMiddleware."""
    app = FastAPI()

    @app.get("/v2/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v2/domains")
    async def domains():
        return {"domains": []}

    @app.get("/v2/domains/software")
    async def domain_info():
        return {"name": "software"}

    @app.post("/v2/compile")
    async def compile(request: Request):
        key_id = getattr(request.state, "api_key_id", None)
        key_name = getattr(request.state, "api_key_name", None)
        return {"key_id": key_id, "key_name": key_name}

    app.add_middleware(
        APIKeyMiddleware,
        require_key=require_key,
        key_store=key_store,
        rate_limiter=rate_limiter,
    )
    return app


# =============================================================================
# NO AUTH (default)
# =============================================================================

class TestNoAuth:
    def test_no_auth_required_passes(self):
        app = _make_app(require_key=False)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={})
        assert resp.status_code == 200

    def test_require_key_no_store_accepts_any(self):
        """When require_key=True but no store, only checks for non-empty key."""
        app = _make_app(require_key=True)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": "anything"})
        assert resp.status_code == 200


# =============================================================================
# AUTH BYPASS ENDPOINTS
# =============================================================================

class TestAuthBypass:
    def test_health_no_key_required(self, store, limiter):
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.get("/v2/health")
        assert resp.status_code == 200

    def test_domains_no_key_required(self, store, limiter):
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.get("/v2/domains")
        assert resp.status_code == 200

    def test_domain_info_no_key_required(self, store, limiter):
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.get("/v2/domains/software")
        assert resp.status_code == 200


# =============================================================================
# 401 — MISSING / INVALID / REVOKED
# =============================================================================

class TestAuth401:
    def test_missing_key_returns_401(self, store, limiter):
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={})
        assert resp.status_code == 401
        assert "E10001" in resp.json().get("error_code", "")

    def test_invalid_key_returns_401(self, store, limiter):
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

    def test_revoked_key_returns_401(self, store, limiter):
        key_id, raw_key = store.create_key("test")
        store.revoke_key(key_id)
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
        assert resp.status_code == 401


# =============================================================================
# 402 — BUDGET EXCEEDED
# =============================================================================

class TestAuth402:
    def test_budget_exceeded_returns_402(self, store, limiter):
        key_id, raw_key = store.create_key("budget-test", budget=10.0)
        store.record_spend(key_id, 10.0)
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
        assert resp.status_code == 402
        assert "E10003" in resp.json().get("error_code", "")


# =============================================================================
# 429 — RATE LIMIT
# =============================================================================

class TestAuth429:
    def test_rate_limit_exceeded_returns_429(self, store):
        key_id, raw_key = store.create_key("rate-test", rate_limit=3)
        limiter = RateLimiter(store, window_seconds=3600)
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        # Use up the limit
        for _ in range(3):
            resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
            assert resp.status_code == 200
        # This one should be rate limited
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
        assert resp.status_code == 429
        assert "E10002" in resp.json().get("error_code", "")


# =============================================================================
# SUCCESSFUL AUTH
# =============================================================================

class TestAuthSuccess:
    def test_valid_key_passes_with_state(self, store, limiter):
        key_id, raw_key = store.create_key("success-test")
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["key_id"] == key_id
        assert data["key_name"] == "success-test"

    def test_rate_limit_headers_present(self, store, limiter):
        _, raw_key = store.create_key("header-test", rate_limit=100)
        app = _make_app(require_key=True, key_store=store, rate_limiter=limiter)
        client = TestClient(app)
        resp = client.post("/v2/compile", json={}, headers={"X-API-Key": raw_key})
        assert "X-RateLimit-Limit" in resp.headers
        assert resp.headers["X-RateLimit-Limit"] == "100"
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers
