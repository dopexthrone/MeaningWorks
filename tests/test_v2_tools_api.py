"""Tests for V2 API tool sharing and instance discovery endpoints."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from core.tool_package import (
    package_tool,
    serialize_tool_package,
    compute_package_id,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_package(name="TestTool", domain="software", trust=80.0, badge="verified", fp_suffix=None):
    fp = fp_suffix or name.lower()
    bp = {
        "core_need": name,
        "components": [
            {"name": f"{name}Core", "type": "service", "description": name},
        ],
        "relationships": [],
    }
    code = {f"{name}Core": f"class {name}Core:\n    pass\n"}
    return package_tool(
        compilation_id=f"comp_{name}",
        blueprint=bp,
        generated_code=code,
        trust_score=trust,
        verification_badge=badge,
        fidelity_scores={"completeness": 80, "consistency": 75, "coherence": 85, "traceability": 70},
        fingerprint_hash=f"fp_{fp}",
        instance_id="inst_test001",
        domain=domain,
        name=name,
    )


@pytest.fixture
def app():
    """Create a FastAPI test app with V2 router."""
    from fastapi import FastAPI
    from api.v2.routes import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def populated_registry(tmp_path):
    """Registry with some tools pre-loaded."""
    from motherlabs_platform.tool_registry import ToolRegistry
    db = str(tmp_path / "test_tools.db")
    registry = ToolRegistry(db_path=db)
    registry.register_tool(_make_package("Alpha", fp_suffix="alpha"))
    registry.register_tool(_make_package("Beta", domain="process", fp_suffix="beta"))
    return registry


# =============================================================================
# TOOLS ENDPOINTS
# =============================================================================

class TestListToolsEndpoint:
    def test_list_empty(self, client, tmp_path):
        db = str(tmp_path / "empty_tools.db")
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            from motherlabs_platform.tool_registry import ToolRegistry
            mock.return_value = ToolRegistry(db_path=db)
            resp = client.get("/v2/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["tools"] == []

    def test_list_with_tools(self, client, populated_registry):
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            mock.return_value = populated_registry
            resp = client.get("/v2/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_list_filter_domain(self, client, populated_registry):
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            mock.return_value = populated_registry
            resp = client.get("/v2/tools?domain=process")
        data = resp.json()
        assert data["total"] == 1
        assert data["tools"][0]["domain"] == "process"


class TestSearchToolsEndpoint:
    def test_search_by_name(self, client, populated_registry):
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            mock.return_value = populated_registry
            resp = client.get("/v2/tools/search?q=Alpha")
        data = resp.json()
        assert data["total"] == 1
        assert data["tools"][0]["name"] == "Alpha"


class TestGetToolEndpoint:
    def test_get_existing(self, client, populated_registry):
        tools = populated_registry.list_tools()
        pkg_id = tools[0].package_id
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            mock.return_value = populated_registry
            resp = client.get(f"/v2/tools/{pkg_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["package_id"] == pkg_id

    def test_get_nonexistent(self, client, tmp_path):
        db = str(tmp_path / "empty.db")
        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock:
            from motherlabs_platform.tool_registry import ToolRegistry
            mock.return_value = ToolRegistry(db_path=db)
            resp = client.get("/v2/tools/nonexistent_id")
        assert resp.status_code == 404


class TestImportToolEndpoint:
    def test_import_valid(self, client, tmp_path):
        db = str(tmp_path / "import.db")
        pkg = _make_package("ImportMe")
        pkg_data = serialize_tool_package(pkg)

        with patch("motherlabs_platform.tool_registry.get_tool_registry") as mock_reg, \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_id:
            from motherlabs_platform.tool_registry import ToolRegistry
            mock_reg.return_value = ToolRegistry(db_path=db)
            mock_id_instance = MagicMock()
            mock_id_instance.get_or_create_self.return_value = MagicMock(instance_id="inst_import")
            mock_id.return_value = mock_id_instance

            resp = client.post("/v2/tools/import", json={
                "package": pkg_data,
                "min_trust_score": 60.0,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["allowed"] is True

    def test_import_bad_package(self, client):
        resp = client.post("/v2/tools/import", json={
            "package": {"invalid": True},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


# =============================================================================
# INSTANCE ENDPOINTS
# =============================================================================

class TestInstanceDigestEndpoint:
    def test_digest(self, client, tmp_path):
        db = str(tmp_path / "digest.db")
        inst_db = str(tmp_path / "inst.db")
        from motherlabs_platform.tool_registry import ToolRegistry
        from motherlabs_platform.instance_identity import InstanceIdentityStore

        registry = ToolRegistry(db_path=db)
        store = InstanceIdentityStore(db_path=inst_db)
        identity = store.get_or_create_self("test-instance")

        with patch("motherlabs_platform.tool_registry.get_tool_registry", return_value=registry), \
             patch("motherlabs_platform.instance_identity.InstanceIdentityStore", return_value=store):
            resp = client.get("/v2/instance/digest")

        assert resp.status_code == 200
        data = resp.json()
        assert data["instance_id"] == identity.instance_id
        assert data["tool_count"] == 0


class TestPeerEndpoints:
    def test_register_and_list_peers(self, client, tmp_path):
        db = str(tmp_path / "peers.db")
        from motherlabs_platform.instance_identity import InstanceIdentityStore
        store = InstanceIdentityStore(db_path=db)

        with patch("motherlabs_platform.instance_identity.InstanceIdentityStore") as mock_cls:
            mock_cls.return_value = store

            # Register peer
            resp = client.post("/v2/instance/peers", json={
                "instance_id": "peer_abc12345",
                "name": "Alice",
                "api_endpoint": "http://alice:8000",
            })
            assert resp.status_code == 200
            assert resp.json()["success"] is True

            # List peers
            resp = client.get("/v2/instance/peers")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["peers"][0]["name"] == "Alice"
