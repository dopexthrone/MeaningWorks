"""Tests for runtime scaffold template generators."""

import ast
import re

import pytest

from core.domain_adapter import RuntimeCapabilities
from core.runtime_scaffold import (
    generate_runtime_py,
    generate_state_py,
    generate_tools_py,
    generate_llm_client_py,
    generate_config_py,
    generate_recompile_py,
    generate_compiler_py,
    generate_tool_manager_py,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def full_capabilities():
    """RuntimeCapabilities with everything enabled."""
    return RuntimeCapabilities(
        has_event_loop=True,
        has_llm_client=True,
        has_persistent_state=True,
        has_self_recompile=True,
        has_tool_execution=True,
        event_loop_type="asyncio",
        state_backend="sqlite",
        entry_point="main.py",
        default_port=9090,
        tool_allowlist=("web_search", "file_read", "file_write"),
    )


@pytest.fixture
def disabled_capabilities():
    """RuntimeCapabilities with everything disabled."""
    return RuntimeCapabilities()


@pytest.fixture
def sample_blueprint():
    return {
        "domain": "chat_agent",
        "core_need": "A conversational agent",
        "components": [
            {"name": "ChatAgent", "type": "agent"},
            {"name": "MemoryStore", "type": "state_store"},
        ],
    }


@pytest.fixture
def sample_names():
    return ["ChatAgent", "MemoryStore"]


# =============================================================================
# DISABLED CAPABILITIES — all generators return empty string
# =============================================================================

class TestDisabledCapabilities:
    """When capabilities are disabled, generators return empty string."""

    def test_runtime_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_runtime_py(disabled_capabilities, sample_blueprint, sample_names) == ""

    def test_state_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_state_py(disabled_capabilities, sample_blueprint, sample_names) == ""

    def test_tools_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_tools_py(disabled_capabilities, sample_blueprint, sample_names) == ""

    def test_llm_client_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_llm_client_py(disabled_capabilities, sample_blueprint, sample_names) == ""

    def test_config_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_config_py(disabled_capabilities, sample_blueprint, sample_names) == ""

    def test_recompile_disabled(self, disabled_capabilities, sample_blueprint, sample_names):
        assert generate_recompile_py(disabled_capabilities, sample_blueprint, sample_names) == ""


# =============================================================================
# SYNTAX VALIDITY — all generated code must parse
# =============================================================================

class TestSyntaxValidity:
    """All generated code must be syntactically valid Python."""

    def test_runtime_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)  # Raises SyntaxError if invalid

    def test_state_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_state_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_tools_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_llm_client_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_config_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_config_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_recompile_valid_python(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        ast.parse(code)


# =============================================================================
# RUNTIME.PY CONTENT
# =============================================================================

class TestGenerateRuntime:
    """Test runtime.py generator content."""

    def test_contains_runtime_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "class Runtime:" in code

    def test_contains_async_event_loop(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "asyncio" in code
        assert "async" in code

    def test_contains_dispatch_method(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "dispatch" in code

    def test_contains_start_method(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "async def start" in code

    def test_uses_configured_port(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "9090" in code

    def test_contains_register_method(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "def register" in code

    def test_contains_emit_method(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "def emit" in code

    def test_contains_ready_signal(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "READY" in code


# =============================================================================
# STATE.PY CONTENT
# =============================================================================

class TestGenerateState:
    """Test state.py generator content."""

    def test_contains_state_store_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_state_py(full_capabilities, sample_blueprint, sample_names)
        assert "class StateStore:" in code

    def test_contains_sqlite_connection(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_state_py(full_capabilities, sample_blueprint, sample_names)
        assert "sqlite3" in code

    def test_contains_crud_operations(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_state_py(full_capabilities, sample_blueprint, sample_names)
        assert "async def get" in code
        assert "async def set" in code
        assert "async def delete" in code
        assert "async def query" in code

    def test_json_backend(self, sample_blueprint, sample_names):
        """JSON state backend should use file storage."""
        cap = RuntimeCapabilities(
            has_persistent_state=True,
            state_backend="json",
        )
        code = generate_state_py(cap, sample_blueprint, sample_names)
        assert "class StateStore:" in code
        assert "json" in code
        assert "sqlite3" not in code


# =============================================================================
# TOOLS.PY CONTENT
# =============================================================================

class TestGenerateTools:
    """Test tools.py generator content."""

    def test_contains_tool_executor_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        assert "class ToolExecutor:" in code

    def test_respects_allowlist(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        assert "TOOL_ALLOWLIST" in code
        assert "web_search" in code
        assert "file_read" in code
        assert "file_write" in code

    def test_allowlist_enforcement(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        assert "not in self.allowlist" in code

    def test_subprocess_sandbox(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        assert "subprocess" in code or "create_subprocess" in code

    def test_path_traversal_protection(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_tools_py(full_capabilities, sample_blueprint, sample_names)
        assert ".." in code  # path traversal check

    def test_empty_allowlist(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(has_tool_execution=True, tool_allowlist=())
        code = generate_tools_py(cap, sample_blueprint, sample_names)
        assert "TOOL_ALLOWLIST" in code
        assert "frozenset(" in code


# =============================================================================
# LLM_CLIENT.PY CONTENT
# =============================================================================

class TestGenerateLLMClient:
    """Test llm_client.py generator content."""

    def test_contains_llm_client_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "class LLMClient:" in code

    def test_env_var_configuration(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "LLM_PROVIDER" in code
        assert "LLM_API_KEY" in code

    def test_retry_logic(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "max_retries" in code
        assert "retry" in code.lower() or "attempt" in code.lower()

    def test_async_methods(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "async def chat" in code
        assert "async def complete" in code

    def test_anthropic_provider(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "anthropic" in code

    def test_openai_provider(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_llm_client_py(full_capabilities, sample_blueprint, sample_names)
        assert "openai" in code


# =============================================================================
# CONFIG.PY CONTENT
# =============================================================================

class TestGenerateConfig:
    """Test config.py generator content."""

    def test_contains_config_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_config_py(full_capabilities, sample_blueprint, sample_names)
        assert "class Config:" in code

    def test_env_overrides(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_config_py(full_capabilities, sample_blueprint, sample_names)
        assert "os.environ.get" in code

    def test_configured_port(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_config_py(full_capabilities, sample_blueprint, sample_names)
        assert "9090" in code

    def test_contains_recompile_url(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_config_py(full_capabilities, sample_blueprint, sample_names)
        assert "recompile" in code.lower()


# =============================================================================
# RECOMPILE.PY CONTENT
# =============================================================================

class TestGenerateRecompile:
    """Test recompile.py generator content."""

    def test_contains_self_recompiler_class(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "class SelfRecompiler:" in code

    def test_contains_detect_gap(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "detect_gap" in code

    def test_contains_request_recompilation(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "request_recompilation" in code

    def test_contains_safe_deploy(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "safe_deploy" in code

    def test_embeds_blueprint(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "chat_agent" in code  # domain from blueprint

    def test_calls_recompile_endpoint(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "v2/recompile" in code

    def test_rollback_on_failure(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "backup" in code.lower()
        assert "rollback" in code.lower()


# =============================================================================
# NO MOTHERLABS IMPORTS — generated code is standalone
# =============================================================================

class TestStandaloneCode:
    """Verify generated code has no motherlabs imports."""

    GENERATORS = [
        generate_runtime_py,
        generate_state_py,
        generate_tools_py,
        generate_llm_client_py,
        generate_config_py,
        generate_recompile_py,
    ]

    def test_no_motherlabs_imports(self, full_capabilities, sample_blueprint, sample_names):
        for gen_fn in self.GENERATORS:
            code = gen_fn(full_capabilities, sample_blueprint, sample_names)
            if not code:
                continue
            assert "from core." not in code, f"{gen_fn.__name__} has motherlabs import"
            assert "from motherlabs" not in code, f"{gen_fn.__name__} has motherlabs import"
            assert "import core" not in code, f"{gen_fn.__name__} has motherlabs import"

    def test_no_engine_imports(self, full_capabilities, sample_blueprint, sample_names):
        for gen_fn in self.GENERATORS:
            code = gen_fn(full_capabilities, sample_blueprint, sample_names)
            if not code:
                continue
            assert "from engine" not in code, f"{gen_fn.__name__} imports engine"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and partial capability configs."""

    def test_only_state_enabled(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(has_persistent_state=True, state_backend="sqlite")
        code = generate_state_py(cap, sample_blueprint, sample_names)
        assert len(code) > 0
        ast.parse(code)

    def test_only_llm_enabled(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(has_llm_client=True)
        code = generate_llm_client_py(cap, sample_blueprint, sample_names)
        assert len(code) > 0
        ast.parse(code)

    def test_only_tools_enabled(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(has_tool_execution=True, tool_allowlist=("shell_exec",))
        code = generate_tools_py(cap, sample_blueprint, sample_names)
        assert len(code) > 0
        ast.parse(code)
        assert "shell_exec" in code

    def test_config_with_single_capability(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(has_llm_client=True)
        code = generate_config_py(cap, sample_blueprint, sample_names)
        assert len(code) > 0
        ast.parse(code)

    def test_empty_blueprint(self, full_capabilities, sample_names):
        code = generate_recompile_py(full_capabilities, {}, sample_names)
        assert len(code) > 0
        ast.parse(code)

    def test_empty_component_names(self, full_capabilities, sample_blueprint):
        code = generate_runtime_py(full_capabilities, sample_blueprint, [])
        assert len(code) > 0
        ast.parse(code)

    def test_special_characters_in_component_names(self, full_capabilities, sample_blueprint):
        names = ["Chat Agent", "web-search-tool", "Memory_Store"]
        code = generate_runtime_py(full_capabilities, sample_blueprint, names)
        assert len(code) > 0
        ast.parse(code)


# =============================================================================
# RESTART MECHANISM TESTS
# =============================================================================

class TestRestartMechanism:
    """Tests for restart_process() in recompile.py and signal handling in runtime.py."""

    def test_recompile_has_restart_process(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "restart_process" in code

    def test_recompile_uses_execv(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "os.execv" in code
        assert "sys.executable" in code

    def test_safe_deploy_calls_restart(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "self.restart_process()" in code

    def test_safe_deploy_has_drain_delay(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_recompile_py(full_capabilities, sample_blueprint, sample_names)
        assert "asyncio.sleep(2)" in code

    def test_runtime_has_signal_handler(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "signal.SIGTERM" in code
        assert "signal.SIGINT" in code
        assert "_shutdown" in code


# =============================================================================
# _SYSTEM META-TARGET TESTS
# =============================================================================

class TestSystemMetaTarget:
    """Tests for _system meta-target in generated runtime."""

    def test_runtime_has_handle_system(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "_handle_system" in code

    def test_system_dispatches_to_handler(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert '"_system"' in code

    def test_system_supports_status_action(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert '"status"' in code
        assert "self.components.keys()" in code

    def test_system_supports_health_action(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert '"health"' in code
        assert "uptime" in code
        assert "_message_count" in code

    def test_system_supports_learn_action(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert '"learn"' in code
        assert "SelfRecompiler" in code
        assert "request_recompilation" in code

    def test_system_learn_handles_missing_skill(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "Missing 'skill'" in code

    def test_runtime_tracks_message_count(self, full_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(full_capabilities, sample_blueprint, sample_names)
        assert "_message_count" in code
        assert "self._message_count += 1" in code


# =============================================================================
# MOTHER AGENT: RuntimeCapabilities new fields
# =============================================================================

class TestRuntimeCapabilitiesNewFields:
    """Test new can_compile, can_share_tools, corpus_path fields."""

    def test_defaults_are_disabled(self):
        cap = RuntimeCapabilities()
        assert cap.can_compile is False
        assert cap.can_share_tools is False
        assert cap.corpus_path == ""

    def test_can_set_mother_fields(self):
        cap = RuntimeCapabilities(
            can_compile=True,
            can_share_tools=True,
            corpus_path="~/motherlabs/corpus.db",
        )
        assert cap.can_compile is True
        assert cap.can_share_tools is True
        assert cap.corpus_path == "~/motherlabs/corpus.db"

    def test_frozen_cannot_modify(self):
        cap = RuntimeCapabilities(can_compile=True)
        with pytest.raises(AttributeError):
            cap.can_compile = False


# =============================================================================
# COMPILER.PY CONTENT
# =============================================================================

class TestGenerateCompiler:
    """Test compiler.py generator content."""

    @pytest.fixture
    def mother_capabilities(self):
        return RuntimeCapabilities(
            has_event_loop=True, has_llm_client=True,
            can_compile=True, corpus_path="~/motherlabs/corpus.db",
        )

    def test_disabled_returns_empty(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(can_compile=False)
        assert generate_compiler_py(cap, sample_blueprint, sample_names) == ""

    def test_enabled_returns_code(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert len(code) > 0

    def test_valid_python(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_contains_tool_compiler_class(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "class ToolCompiler:" in code

    def test_contains_compile_tool_method(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def compile_tool" in code

    def test_contains_lazy_init(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "_ensure_engine" in code
        assert "_initialized" in code

    def test_uses_corpus_path(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "~/motherlabs/corpus.db" in code

    def test_uses_run_in_executor(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "run_in_executor" in code

    def test_imports_from_motherlabs(self, mother_capabilities, sample_blueprint, sample_names):
        """Compiler.py DOES import from motherlabs — this is intentional for mother agents."""
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "from core.engine" in code
        assert "from core.llm" in code

    def test_returns_compilation_result_dict(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_compiler_py(mother_capabilities, sample_blueprint, sample_names)
        assert "compilation_id" in code
        assert "trust_score" in code
        assert "verification_badge" in code


# =============================================================================
# TOOL_MANAGER.PY CONTENT
# =============================================================================

class TestGenerateToolManager:
    """Test tool_manager.py generator content."""

    @pytest.fixture
    def mother_capabilities(self):
        return RuntimeCapabilities(
            has_event_loop=True, has_llm_client=True,
            can_share_tools=True, corpus_path="~/motherlabs/corpus.db",
        )

    def test_disabled_returns_empty(self, sample_blueprint, sample_names):
        cap = RuntimeCapabilities(can_share_tools=False)
        assert generate_tool_manager_py(cap, sample_blueprint, sample_names) == ""

    def test_enabled_returns_code(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert len(code) > 0

    def test_valid_python(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        ast.parse(code)

    def test_contains_tool_manager_class(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "class ToolManager:" in code

    def test_contains_list_tools(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def list_tools" in code

    def test_contains_search_tools(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def search_tools" in code

    def test_contains_export_tool(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def export_tool" in code

    def test_contains_import_tool(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def import_tool" in code

    def test_contains_get_instance_info(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "async def get_instance_info" in code

    def test_uses_lazy_imports(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "_ensure_registry" in code
        assert "_ensure_corpus" in code

    def test_imports_from_motherlabs(self, mother_capabilities, sample_blueprint, sample_names):
        """ToolManager DOES import from motherlabs — intentional for mother agents."""
        code = generate_tool_manager_py(mother_capabilities, sample_blueprint, sample_names)
        assert "from motherlabs_platform.tool_registry" in code
        assert "from core.tool_export" in code


# =============================================================================
# RUNTIME _SYSTEM COMPILE/TOOLS/INSTANCE ACTIONS
# =============================================================================

class TestSystemMotherActions:
    """Tests for compile/tools/instance actions in _handle_system."""

    @pytest.fixture
    def mother_capabilities(self):
        return RuntimeCapabilities(
            has_event_loop=True, has_llm_client=True,
            has_persistent_state=True, has_tool_execution=True,
            has_self_recompile=True,
            event_loop_type="asyncio", state_backend="sqlite",
            default_port=8080,
            can_compile=True, can_share_tools=True,
            corpus_path="~/motherlabs/corpus.db",
        )

    def test_runtime_accepts_compiler_param(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert "compiler=None" in code
        assert "self.compiler = compiler" in code

    def test_runtime_accepts_tool_manager_param(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert "tool_manager=None" in code
        assert "self.tool_manager = tool_manager" in code

    def test_system_supports_compile_action(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert '"compile"' in code
        assert "self.compiler.compile_tool" in code

    def test_system_compile_guards_none(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert "self.compiler is None" in code
        assert '"Compilation not available"' in code

    def test_system_supports_tools_action(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert '"tools"' in code
        assert "self.tool_manager" in code

    def test_system_supports_instance_action(self, mother_capabilities, sample_blueprint, sample_names):
        code = generate_runtime_py(mother_capabilities, sample_blueprint, sample_names)
        assert '"instance"' in code
        assert "get_instance_info" in code
