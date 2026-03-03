"""Tests for the /v2/recompile API endpoint and models."""

import pytest

from api.v2.models import (
    V2RecompileRequest,
    V2RecompileResponse,
    TrustResponse,
)


# =============================================================================
# REQUEST MODEL
# =============================================================================

class TestRecompileRequestModel:
    """Test V2RecompileRequest validation."""

    def test_valid_request(self):
        req = V2RecompileRequest(
            current_blueprint={"components": [], "domain": "test"},
            enhancement="Add web search capability",
        )
        assert req.domain == "agent_system"
        assert req.enhancement == "Add web search capability"

    def test_default_domain(self):
        req = V2RecompileRequest(
            current_blueprint={},
            enhancement="test",
        )
        assert req.domain == "agent_system"

    def test_custom_domain(self):
        req = V2RecompileRequest(
            current_blueprint={},
            enhancement="test",
            domain="software",
        )
        assert req.domain == "software"

    def test_empty_enhancement_rejected(self):
        with pytest.raises(Exception):
            V2RecompileRequest(
                current_blueprint={},
                enhancement="",
            )

    def test_provider_optional(self):
        req = V2RecompileRequest(
            current_blueprint={},
            enhancement="test",
            provider="claude",
        )
        assert req.provider == "claude"

    def test_provider_none_by_default(self):
        req = V2RecompileRequest(
            current_blueprint={},
            enhancement="test",
        )
        assert req.provider is None


# =============================================================================
# RESPONSE MODEL
# =============================================================================

class TestRecompileResponseModel:
    """Test V2RecompileResponse structure."""

    def test_success_response(self):
        resp = V2RecompileResponse(
            success=True,
            blueprint={"components": [{"name": "Agent"}]},
            enhancement_applied="Add search",
        )
        assert resp.success is True
        assert resp.enhancement_applied == "Add search"

    def test_failure_response(self):
        resp = V2RecompileResponse(
            success=False,
            error="Compilation failed",
        )
        assert resp.success is False
        assert resp.error == "Compilation failed"

    def test_default_domain(self):
        resp = V2RecompileResponse(success=True)
        assert resp.domain == "agent_system"

    def test_trust_default(self):
        resp = V2RecompileResponse(success=True)
        assert isinstance(resp.trust, TrustResponse)
        assert resp.trust.overall_score == 0.0

    def test_materialized_output_default_empty(self):
        resp = V2RecompileResponse(success=True)
        assert resp.materialized_output == {}


# =============================================================================
# ROUTE IMPORTS (no server needed, just verify wiring)
# =============================================================================

class TestRecompileRouteExists:
    """Verify the recompile route is properly registered."""

    def test_route_importable(self):
        """Verify the route module imports without error."""
        from api.v2.routes import router
        # Router prefix is /v2, so paths are /v2/recompile
        paths = [route.path for route in router.routes]
        assert "/v2/recompile" in paths

    def test_route_is_post(self):
        from api.v2.routes import router
        for route in router.routes:
            if getattr(route, 'path', None) == "/v2/recompile":
                assert "POST" in route.methods
                break
        else:
            pytest.fail("No /v2/recompile route found")
