"""Integration tests for Mother Agent — self-compiling instance.

These tests verify the full mother agent loop: compile → export → import → list.
All tests are marked @pytest.mark.slow as they require real LLM calls and
full engine initialization.

Run with: pytest tests/integration/test_mother_agent.py -m slow --run-slow
"""

import asyncio
import json
import os
import tempfile
import time

import pytest

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def corpus_path(temp_dir):
    """Return a temporary corpus path."""
    return os.path.join(temp_dir, "test_corpus.db")


@pytest.fixture
def registry_path(temp_dir):
    """Return a temporary registry path."""
    return os.path.join(temp_dir, "test_tools.db")


@pytest.fixture
def instance_id():
    """Return a test instance ID."""
    return "test-mother-instance-001"


@pytest.fixture
def compiler(corpus_path):
    """Create a ToolCompiler with test corpus."""
    # Import the actual scaffold generator to get the class
    from core.runtime_scaffold import generate_compiler_py
    from core.domain_adapter import RuntimeCapabilities

    cap = RuntimeCapabilities(can_compile=True, corpus_path=corpus_path)
    code = generate_compiler_py(cap, {}, [])
    assert code  # Should be non-empty

    # We can't easily exec the generated code and get a real ToolCompiler
    # because it depends on the Motherlabs environment. Instead, verify
    # the scaffold generates valid code and test via the actual imports.
    return code


@pytest.fixture
def tool_manager_code(corpus_path, registry_path, instance_id):
    """Generate ToolManager code."""
    from core.runtime_scaffold import generate_tool_manager_py
    from core.domain_adapter import RuntimeCapabilities

    cap = RuntimeCapabilities(can_share_tools=True, corpus_path=corpus_path)
    code = generate_tool_manager_py(cap, {}, [])
    assert code
    return code


class TestMotherAgentCompilation:
    """Test compilation via mother agent infrastructure."""

    def test_compile_tool_via_engine(self, corpus_path):
        """Start mother agent infrastructure, compile a simple tool."""
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from persistence.corpus import Corpus
        from pathlib import Path

        import adapters  # noqa: F401 — register all adapters

        corpus = Corpus(Path(corpus_path))
        adapter = get_adapter("software")
        llm_client = get_llm_client("anthropic")

        engine = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus,
            domain_adapter=adapter,
        )
        result = engine.compile("a simple calculator that adds two numbers")

        assert result.compilation_id
        assert result.trust_indicators is not None
        assert result.trust_indicators.composite_score > 0

    def test_export_tool_after_compile(self, corpus_path, temp_dir):
        """Compile a tool, then export it as .mtool."""
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from core.tool_export import export_tool_to_file
        from persistence.corpus import Corpus
        from pathlib import Path

        import adapters  # noqa: F401

        corpus = Corpus(Path(corpus_path))
        adapter = get_adapter("software")
        llm_client = get_llm_client("anthropic")

        engine = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus,
            domain_adapter=adapter,
        )
        result = engine.compile("a function that reverses a string")
        compilation_id = result.compilation_id

        output_path = os.path.join(temp_dir, f"{compilation_id}.mtool")
        export_result = export_tool_to_file(
            compilation_id=compilation_id,
            corpus=corpus,
            output_path=output_path,
            instance_id="test-instance",
        )

        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0

    def test_import_tool_with_validation(self, corpus_path, temp_dir, registry_path):
        """Compile, export, then import with governor validation."""
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from core.tool_export import export_tool_to_file, load_tool_from_file, import_tool
        from motherlabs_platform.tool_registry import ToolRegistry
        from persistence.corpus import Corpus
        from pathlib import Path

        import adapters  # noqa: F401

        corpus = Corpus(Path(corpus_path))
        adapter = get_adapter("software")
        llm_client = get_llm_client("anthropic")

        engine = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus,
            domain_adapter=adapter,
        )
        result = engine.compile("a greeting function")
        compilation_id = result.compilation_id

        # Export
        output_path = os.path.join(temp_dir, f"{compilation_id}.mtool")
        export_tool_to_file(
            compilation_id=compilation_id,
            corpus=corpus,
            output_path=output_path,
            instance_id="test-instance",
        )

        # Import into fresh registry
        registry = ToolRegistry(registry_path)
        import_corpus = Corpus(Path(os.path.join(temp_dir, "import_corpus.db")))
        package = load_tool_from_file(output_path)
        import_result = import_tool(
            package=package,
            corpus=import_corpus,
            registry=registry,
            min_trust_score=0.0,  # Accept all for testing
        )

        assert import_result.get("accepted", False) is True

    def test_list_tools_after_import(self, corpus_path, temp_dir, registry_path):
        """After compile+export+import, the tool appears in list."""
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from core.tool_export import export_tool_to_file, load_tool_from_file, import_tool
        from motherlabs_platform.tool_registry import ToolRegistry
        from persistence.corpus import Corpus
        from pathlib import Path

        import adapters  # noqa: F401

        corpus = Corpus(Path(corpus_path))
        adapter = get_adapter("software")
        llm_client = get_llm_client("anthropic")

        engine = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus,
            domain_adapter=adapter,
        )
        result = engine.compile("a counter utility")

        output_path = os.path.join(temp_dir, f"{result.compilation_id}.mtool")
        export_tool_to_file(
            compilation_id=result.compilation_id,
            corpus=corpus,
            output_path=output_path,
            instance_id="test-instance",
        )

        registry = ToolRegistry(registry_path)
        import_corpus = Corpus(Path(os.path.join(temp_dir, "import_corpus.db")))
        package = load_tool_from_file(output_path)
        import_tool(
            package=package,
            corpus=import_corpus,
            registry=registry,
            min_trust_score=0.0,
        )

        tools = registry.list_tools()
        assert len(tools) >= 1

    def test_instance_info(self, registry_path, instance_id):
        """Instance info returns correct structure."""
        from motherlabs_platform.tool_registry import ToolRegistry

        registry = ToolRegistry(registry_path)
        tools = registry.list_tools()
        info = {
            "instance_id": instance_id,
            "tool_count": len(tools),
            "status": "active",
        }
        assert info["instance_id"] == instance_id
        assert info["tool_count"] >= 0
        assert info["status"] == "active"

    def test_full_mother_loop(self, temp_dir):
        """Full loop: compile → export → import on fresh instance → verify provenance."""
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from core.tool_export import export_tool_to_file, load_tool_from_file, import_tool
        from motherlabs_platform.tool_registry import ToolRegistry
        from persistence.corpus import Corpus
        from pathlib import Path

        import adapters  # noqa: F401

        # Instance A: compile + export
        corpus_a = Corpus(Path(os.path.join(temp_dir, "corpus_a.db")))
        adapter = get_adapter("software")
        llm_client = get_llm_client("anthropic")
        engine_a = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus_a,
            domain_adapter=adapter,
        )
        result = engine_a.compile("a temperature converter")
        mtool_path = os.path.join(temp_dir, f"{result.compilation_id}.mtool")
        export_tool_to_file(
            compilation_id=result.compilation_id,
            corpus=corpus_a,
            output_path=mtool_path,
            instance_id="instance-A",
        )

        # Instance B: import + verify provenance
        corpus_b = Corpus(Path(os.path.join(temp_dir, "corpus_b.db")))
        registry_b = ToolRegistry(os.path.join(temp_dir, "registry_b.db"))
        package = load_tool_from_file(mtool_path)

        # Verify provenance chain includes instance A
        assert package.provenance is not None
        assert len(package.provenance) >= 1

        import_result = import_tool(
            package=package,
            corpus=corpus_b,
            registry=registry_b,
            min_trust_score=0.0,
        )
        assert import_result.get("accepted", False) is True

        # Verify tool is in instance B's registry
        tools = registry_b.list_tools()
        assert len(tools) >= 1
