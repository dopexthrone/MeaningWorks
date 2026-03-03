"""Tests for project writer runtime scaffold integration."""

import ast
import json
import os
import tempfile

import pytest

from core.domain_adapter import RuntimeCapabilities
from core.project_writer import (
    ProjectConfig,
    write_project,
    _generate_main_py,
    _infer_runtime_requirements,
)


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_BLUEPRINT = {
    "domain": "agent_system",
    "core_need": "A chat agent with search",
    "components": [
        {"name": "ChatAgent", "type": "agent", "description": "Main chat agent"},
        {"name": "SearchTool", "type": "tool", "description": "Web search tool"},
        {"name": "MemoryStore", "type": "state_store", "description": "Conversation memory"},
    ],
    "relationships": [
        {"from": "ChatAgent", "to": "SearchTool", "type": "uses_tool"},
        {"from": "ChatAgent", "to": "MemoryStore", "type": "reads_state"},
    ],
}

SAMPLE_CODE = {
    "ChatAgent": '''class ChatAgent:
    """Main chat agent."""
    def __init__(self):
        self.state = None
        self.llm = None
        self.tools = None

    async def handle(self, message):
        return {"response": "Hello!"}
''',
    "SearchTool": '''class SearchTool:
    """Web search tool."""
    async def handle(self, message):
        return {"results": []}
''',
    "MemoryStore": '''class MemoryStore:
    """Conversation memory."""
    async def handle(self, message):
        return {"stored": True}
''',
}

FULL_RUNTIME = RuntimeCapabilities(
    has_event_loop=True,
    has_llm_client=True,
    has_persistent_state=True,
    has_self_recompile=True,
    has_tool_execution=True,
    event_loop_type="asyncio",
    state_backend="sqlite",
    default_port=9999,
    tool_allowlist=("web_search", "file_read"),
)


# =============================================================================
# PROJECT CONFIG
# =============================================================================

class TestProjectConfigRuntime:
    """Test ProjectConfig with runtime_capabilities field."""

    def test_default_none(self):
        config = ProjectConfig()
        assert config.runtime_capabilities is None

    def test_with_runtime(self):
        config = ProjectConfig(runtime_capabilities=FULL_RUNTIME)
        assert config.runtime_capabilities is FULL_RUNTIME

    def test_frozen(self):
        config = ProjectConfig(runtime_capabilities=FULL_RUNTIME)
        with pytest.raises(AttributeError):
            config.runtime_capabilities = None


# =============================================================================
# RUNTIME REQUIREMENTS
# =============================================================================

class TestInferRuntimeRequirements:
    """Test runtime-specific dependency inference."""

    def test_llm_client_adds_httpx(self):
        cap = RuntimeCapabilities(has_llm_client=True)
        deps = _infer_runtime_requirements(cap)
        assert "httpx" in deps

    def test_websocket_adds_deps(self):
        cap = RuntimeCapabilities(has_event_loop=True, event_loop_type="websocket")
        deps = _infer_runtime_requirements(cap)
        assert "uvicorn" in deps
        assert "fastapi" in deps

    def test_default_caps_no_deps(self):
        cap = RuntimeCapabilities()
        deps = _infer_runtime_requirements(cap)
        assert deps == []

    def test_asyncio_no_extra_deps(self):
        cap = RuntimeCapabilities(has_event_loop=True, event_loop_type="asyncio")
        deps = _infer_runtime_requirements(cap)
        # asyncio is stdlib, no extra deps
        assert "uvicorn" not in deps


# =============================================================================
# ASYNC MAIN GENERATION
# =============================================================================

class TestAsyncMainGeneration:
    """Test async main.py generation for agent systems."""

    def test_generates_async_main(self):
        code = _generate_main_py(
            SAMPLE_BLUEPRINT,
            list(SAMPLE_CODE.keys()),
            "",
            grouped_files={"services.py": SAMPLE_CODE},
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "asyncio" in code
        assert "async def main" in code
        assert "asyncio.run(main())" in code

    def test_async_main_imports_runtime(self):
        code = _generate_main_py(
            SAMPLE_BLUEPRINT,
            list(SAMPLE_CODE.keys()),
            "",
            grouped_files={"services.py": SAMPLE_CODE},
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "from runtime import Runtime" in code
        assert "from state import StateStore" in code
        assert "from llm_client import LLMClient" in code
        assert "from tools import ToolExecutor" in code
        assert "from config import Config" in code

    def test_async_main_registers_components(self):
        code = _generate_main_py(
            SAMPLE_BLUEPRINT,
            list(SAMPLE_CODE.keys()),
            "",
            grouped_files={"services.py": SAMPLE_CODE},
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "runtime.register" in code

    def test_async_main_valid_python(self):
        code = _generate_main_py(
            SAMPLE_BLUEPRINT,
            list(SAMPLE_CODE.keys()),
            "",
            grouped_files={"services.py": SAMPLE_CODE},
            runtime_capabilities=FULL_RUNTIME,
        )
        ast.parse(code)

    def test_no_runtime_generates_sync_main(self):
        code = _generate_main_py(
            SAMPLE_BLUEPRINT,
            list(SAMPLE_CODE.keys()),
            "",
        )
        assert "async def main" not in code
        assert "def main" in code


# =============================================================================
# FULL PROJECT WRITE WITH RUNTIME
# =============================================================================

class TestWriteProjectWithRuntime:
    """Test write_project emits scaffold files."""

    def test_emits_scaffold_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            files = manifest.files_written
            assert "runtime.py" in files
            assert "state.py" in files
            assert "tools.py" in files
            assert "llm_client.py" in files
            assert "config.py" in files
            assert "recompile.py" in files

    def test_emits_blueprint_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            assert "blueprint.json" in manifest.files_written
            bp_path = os.path.join(manifest.project_dir, "blueprint.json")
            with open(bp_path) as f:
                bp = json.load(f)
            assert bp["domain"] == "agent_system"

    def test_scaffold_files_parse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            scaffold_names = ["runtime.py", "state.py", "tools.py",
                             "llm_client.py", "config.py", "recompile.py"]
            for fname in scaffold_names:
                filepath = os.path.join(manifest.project_dir, fname)
                assert os.path.exists(filepath), f"{fname} not written"
                with open(filepath) as f:
                    code = f.read()
                ast.parse(code, filename=fname)

    def test_main_is_async(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            main_path = os.path.join(manifest.project_dir, "main.py")
            with open(main_path) as f:
                code = f.read()
            assert "asyncio" in code
            assert "async def main" in code

    def test_requirements_include_httpx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            req_path = os.path.join(manifest.project_dir, "requirements.txt")
            with open(req_path) as f:
                reqs = f.read()
            assert "httpx" in reqs

    def test_no_runtime_no_scaffold(self):
        """Without runtime capabilities, no scaffold files should be emitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(project_name="test_plain")
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            assert "runtime.py" not in manifest.files_written
            assert "state.py" not in manifest.files_written

    def test_file_contents_dict_includes_scaffolds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(
                project_name="test_agent",
                runtime_capabilities=FULL_RUNTIME,
            )
            manifest = write_project(
                SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config,
            )
            assert "runtime.py" in manifest.file_contents
            assert "class Runtime:" in manifest.file_contents["runtime.py"]
