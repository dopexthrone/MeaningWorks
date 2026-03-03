"""Tests for mother/appendage.py — AppendageSpec + AppendageStore."""

import json
import time
from pathlib import Path

import pytest

from mother.appendage import AppendageSpec, AppendageStore


@pytest.fixture
def store(tmp_path):
    """In-memory-equivalent store using tmp_path."""
    db = tmp_path / "test_appendages.db"
    s = AppendageStore(db)
    yield s
    s.close()


class TestAppendageSpec:
    def test_frozen(self):
        spec = AppendageSpec(name="test")
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_defaults(self):
        spec = AppendageSpec()
        assert spec.appendage_id == 0
        assert spec.name == ""
        assert spec.status == "spec"
        assert spec.entry_point == "main.py"
        assert spec.capabilities_json == "[]"
        assert spec.use_count == 0

    def test_construction(self):
        spec = AppendageSpec(
            name="screen-recorder",
            description="Records the screen",
            capability_gap="screen recording",
            status="built",
        )
        assert spec.name == "screen-recorder"
        assert spec.status == "built"


class TestAppendageStore:
    def test_register_and_get(self, store):
        aid = store.register(
            name="file-counter",
            description="Counts files in directories",
            capability_gap="file counting",
            build_prompt="Build an agent that counts files",
            project_dir="/tmp/appendages/file-counter",
            capabilities=["file", "count", "directory"],
        )
        assert aid > 0

        spec = store.get(aid)
        assert spec is not None
        assert spec.name == "file-counter"
        assert spec.description == "Counts files in directories"
        assert spec.capability_gap == "file counting"
        assert spec.status == "spec"
        assert spec.project_dir == "/tmp/appendages/file-counter"
        caps = json.loads(spec.capabilities_json)
        assert "file" in caps
        assert "count" in caps

    def test_get_nonexistent(self, store):
        assert store.get(9999) is None

    def test_get_by_name(self, store):
        store.register("my-agent", "desc", "gap", "prompt", "/tmp/a")
        spec = store.get_by_name("my-agent")
        assert spec is not None
        assert spec.name == "my-agent"

    def test_get_by_name_nonexistent(self, store):
        assert store.get_by_name("nope") is None

    def test_update_status(self, store):
        aid = store.register("test", "d", "g", "p", "/tmp/t")
        store.update_status(aid, "building")
        assert store.get(aid).status == "building"

        store.update_status(aid, "built")
        assert store.get(aid).status == "built"

        store.update_status(aid, "spawned", pid=12345, port=8080)
        spec = store.get(aid)
        assert spec.status == "spawned"
        assert spec.pid == 12345
        assert spec.port == 8080

    def test_update_status_invalid(self, store):
        aid = store.register("test", "d", "g", "p", "/tmp/t")
        with pytest.raises(ValueError, match="Invalid status"):
            store.update_status(aid, "bogus")

    def test_update_status_with_error(self, store):
        aid = store.register("test", "d", "g", "p", "/tmp/t")
        store.update_status(aid, "failed", error="Build crashed")
        spec = store.get(aid)
        assert spec.status == "failed"
        assert spec.error == "Build crashed"

    def test_active(self, store):
        a1 = store.register("a1", "d", "g", "p", "/tmp/a1")
        a2 = store.register("a2", "d", "g", "p", "/tmp/a2")
        a3 = store.register("a3", "d", "g", "p", "/tmp/a3")
        a4 = store.register("a4", "d", "g", "p", "/tmp/a4")

        store.update_status(a1, "spawned")
        store.update_status(a2, "active")
        store.update_status(a3, "solidified")
        # a4 stays as 'spec'

        active = store.active()
        names = [s.name for s in active]
        assert "a1" in names
        assert "a2" in names
        assert "a3" in names
        assert "a4" not in names

    def test_solidified(self, store):
        a1 = store.register("solid", "d", "g", "p", "/tmp/s")
        a2 = store.register("notsolid", "d", "g", "p", "/tmp/ns")
        store.update_status(a1, "solidified")
        store.update_status(a2, "active")

        solidified = store.solidified()
        assert len(solidified) == 1
        assert solidified[0].name == "solid"

    def test_find_for_capability_by_caps_json(self, store):
        aid = store.register(
            "screen-recorder", "Records screen", "screen recording",
            "prompt", "/tmp/sr",
            capabilities=["screen", "record", "capture"],
        )
        store.update_status(aid, "active")

        found = store.find_for_capability("screen")
        assert found is not None
        assert found.name == "screen-recorder"

    def test_find_for_capability_by_name(self, store):
        aid = store.register("file-counter", "Counts files", "gap", "p", "/tmp/fc")
        store.update_status(aid, "built")

        found = store.find_for_capability("file-counter")
        assert found is not None
        assert found.name == "file-counter"

    def test_find_for_capability_by_description(self, store):
        aid = store.register("agent-x", "Monitors network traffic", "gap", "p", "/tmp/ax")
        store.update_status(aid, "solidified")

        found = store.find_for_capability("network")
        assert found is not None
        assert found.name == "agent-x"

    def test_find_for_capability_not_found(self, store):
        store.register("unrelated", "does stuff", "gap", "p", "/tmp/u")
        assert store.find_for_capability("quantum") is None

    def test_find_for_capability_ignores_dissolved(self, store):
        aid = store.register(
            "old", "old agent", "gap", "p", "/tmp/o",
            capabilities=["search"],
        )
        store.update_status(aid, "dissolved")
        assert store.find_for_capability("search") is None

    def test_record_use(self, store):
        aid = store.register("test", "d", "g", "p", "/tmp/t")
        store.update_status(aid, "active")

        assert store.get(aid).use_count == 0

        store.record_use(aid)
        spec = store.get(aid)
        assert spec.use_count == 1
        assert spec.last_used > 0

        store.record_use(aid)
        assert store.get(aid).use_count == 2

    def test_update_cost(self, store):
        aid = store.register("test", "d", "g", "p", "/tmp/t")
        store.update_cost(aid, 1.50)
        assert store.get(aid).total_cost_usd == pytest.approx(1.50)
        store.update_cost(aid, 0.75)
        assert store.get(aid).total_cost_usd == pytest.approx(2.25)

    def test_candidates_for_dissolution(self, store):
        a1 = store.register("idle-low", "d", "g", "p", "/tmp/il")
        a2 = store.register("active-high", "d", "g", "p", "/tmp/ah")
        a3 = store.register("solidified", "d", "g", "p", "/tmp/s")

        store.update_status(a1, "active")
        store.update_status(a2, "active")
        store.update_status(a3, "solidified")

        # a2 gets used frequently
        for _ in range(5):
            store.record_use(a2)

        # a1 was never used, so last_used=0, use_count=0
        candidates = store.candidates_for_dissolution(idle_hours=48, min_uses=3)
        names = [c.name for c in candidates]
        assert "idle-low" in names
        assert "active-high" not in names  # use_count >= 3
        assert "solidified" not in names   # solidified excluded

    def test_all(self, store):
        store.register("a", "d", "g", "p", "/tmp/a")
        store.register("b", "d", "g", "p", "/tmp/b")
        assert len(store.all()) == 2

    def test_lifecycle_full(self, store):
        """Full lifecycle: spec → building → built → spawned → active → solidified."""
        aid = store.register("lifecycle", "test", "gap", "prompt", "/tmp/lc")
        assert store.get(aid).status == "spec"

        store.update_status(aid, "building")
        assert store.get(aid).status == "building"

        store.update_status(aid, "built")
        assert store.get(aid).status == "built"

        store.update_status(aid, "spawned", pid=100)
        spec = store.get(aid)
        assert spec.status == "spawned"
        assert spec.pid == 100

        store.update_status(aid, "active")
        assert store.get(aid).status == "active"

        store.update_status(aid, "solidified")
        assert store.get(aid).status == "solidified"
