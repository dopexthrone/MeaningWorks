"""
Tests for core/build_loop.py.

Phase 27: Runtime Build Loop — build loop tests.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from core.build_loop import (
    BuildConfig,
    BuildResult,
    BuildIteration,
    FixAttempt,
    build_fix_prompt,
    identify_components_to_fix,
    apply_fix_to_project,
    serialize_build_result,
    BUILD_FIX_PREAMBLE,
)
from core.runtime_validator import (
    RuntimeConfig,
    CommandResult,
    ComponentError,
    ValidationResult,
)
from core.project_writer import ProjectManifest


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_BLUEPRINT = {
    "domain": "Task Management",
    "core_need": "A task management system.",
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item with title and status"},
        {"name": "TaskManager", "type": "process", "description": "Manages task lifecycle"},
    ],
    "relationships": [
        {"from": "TaskManager", "to": "Task", "type": "manages"},
    ],
    "constraints": [
        {"description": "Tasks must have unique IDs", "applies_to": ["Task"]},
    ],
}

SAMPLE_CODE = {
    "Task": 'class Task:\n    """A work item."""\n    def __init__(self):\n        self.name = ""\n',
    "TaskManager": 'class TaskManager:\n    """Manages tasks."""\n    def __init__(self):\n        self.tasks = []\n',
}


def _make_component_error(name="Task", error_type="import", msg="ImportError: No module named 'x'"):
    return ComponentError(
        component_name=name,
        error_type=error_type,
        error_message=msg,
        file_path=f"/tmp/project/{name.lower()}.py",
        line_number=1,
    )


def _make_validation_result(success=False, errors=None, unmapped=None):
    return ValidationResult(
        success=success,
        component_errors=tuple(errors or []),
        unmapped_errors=tuple(unmapped or []),
    )


# =============================================================================
# BuildConfig tests
# =============================================================================

class TestBuildConfig:
    def test_defaults(self):
        cfg = BuildConfig()
        assert cfg.max_iterations == 10
        assert cfg.max_fixes_per_component == 5
        assert isinstance(cfg.runtime_config, RuntimeConfig)

    def test_custom(self):
        cfg = BuildConfig(max_iterations=5, max_fixes_per_component=3)
        assert cfg.max_iterations == 5
        assert cfg.max_fixes_per_component == 3

    def test_frozen(self):
        cfg = BuildConfig()
        with pytest.raises(AttributeError):
            cfg.max_iterations = 10


# =============================================================================
# FixAttempt tests
# =============================================================================

class TestFixAttempt:
    def test_basic(self):
        fa = FixAttempt(
            component_name="Task",
            iteration=1,
            error="ImportError",
            prompt="fix this",
            original_code="class Task: pass",
            fixed_code="class Task:\n    pass\n",
            succeeded=True,
        )
        assert fa.succeeded is True
        assert fa.iteration == 1

    def test_frozen(self):
        fa = FixAttempt("Task", 1, "err", "p", "old", "new", True)
        with pytest.raises(AttributeError):
            fa.succeeded = False


# =============================================================================
# BuildIteration tests
# =============================================================================

class TestBuildIteration:
    def test_basic(self):
        vr = _make_validation_result(success=True)
        bi = BuildIteration(iteration=1, validation=vr)
        assert bi.iteration == 1
        assert bi.fixes_attempted == ()
        assert bi.components_fixed == ()

    def test_with_fixes(self):
        vr = _make_validation_result(success=False, errors=[_make_component_error()])
        fa = FixAttempt("Task", 1, "err", "p", "old", "new", True)
        bi = BuildIteration(
            iteration=1,
            validation=vr,
            fixes_attempted=(fa,),
            components_fixed=("Task",),
        )
        assert len(bi.fixes_attempted) == 1
        assert "Task" in bi.components_fixed


# =============================================================================
# BuildResult tests
# =============================================================================

class TestBuildResult:
    def test_success(self):
        br = BuildResult(success=True, final_code={"Task": "code"})
        assert br.success is True
        assert br.total_fix_attempts == 0

    def test_with_unfixed(self):
        br = BuildResult(
            success=False,
            components_fixed=("Task",),
            components_unfixed=("TaskManager",),
            total_fix_attempts=3,
        )
        assert "TaskManager" in br.components_unfixed
        assert br.total_fix_attempts == 3

    def test_frozen(self):
        br = BuildResult(success=True)
        with pytest.raises(AttributeError):
            br.success = False


# =============================================================================
# build_fix_prompt tests
# =============================================================================

class TestBuildFixPrompt:
    def test_basic_prompt(self):
        error = _make_component_error()
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT)
        assert "Task" in prompt
        assert "ImportError" in prompt
        assert "class Task" in prompt
        assert BUILD_FIX_PREAMBLE in prompt

    def test_includes_component_info(self):
        error = _make_component_error()
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT)
        assert "entity" in prompt
        assert "work item" in prompt.lower()

    def test_includes_constraints(self):
        error = _make_component_error()
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT)
        assert "unique IDs" in prompt

    def test_includes_interface_contracts(self):
        interfaces = {
            "contracts": {
                "Task": {
                    "provides": ["get_status()"],
                    "requires": ["TaskManager"],
                }
            }
        }
        error = _make_component_error()
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT, interfaces)
        assert "get_status" in prompt
        assert "Interface Contract" in prompt

    def test_missing_component_in_blueprint(self):
        error = _make_component_error(name="Unknown")
        prompt = build_fix_prompt("Unknown", error, "class Unknown: pass", SAMPLE_BLUEPRINT)
        assert "Unknown" in prompt
        # Should still work, just without component info

    def test_error_details_included(self):
        error = ComponentError(
            component_name="Task",
            error_type="syntax",
            error_message="SyntaxError: unexpected EOF",
            file_path="/tmp/task.py",
            line_number=42,
        )
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT)
        assert "syntax" in prompt.lower()
        assert "42" in prompt
        assert "/tmp/task.py" in prompt


# =============================================================================
# identify_components_to_fix tests
# =============================================================================

class TestIdentifyComponentsToFix:
    def test_basic_identification(self):
        errors = [_make_component_error("Task"), _make_component_error("TaskManager")]
        vr = _make_validation_result(errors=errors)
        to_fix = identify_components_to_fix(vr, {})
        assert len(to_fix) == 2

    def test_deduplicates_components(self):
        errors = [
            _make_component_error("Task", msg="error 1"),
            _make_component_error("Task", msg="error 2"),
        ]
        vr = _make_validation_result(errors=errors)
        to_fix = identify_components_to_fix(vr, {})
        assert len(to_fix) == 1

    def test_respects_max_fixes(self):
        errors = [_make_component_error("Task")]
        vr = _make_validation_result(errors=errors)
        history = {"Task": 2}
        to_fix = identify_components_to_fix(vr, history, max_fixes=2)
        assert len(to_fix) == 0

    def test_prioritizes_fewer_attempts(self):
        errors = [
            _make_component_error("Task"),
            _make_component_error("TaskManager"),
        ]
        vr = _make_validation_result(errors=errors)
        history = {"Task": 1, "TaskManager": 0}
        to_fix = identify_components_to_fix(vr, history)
        assert to_fix[0].component_name == "TaskManager"

    def test_empty_errors(self):
        vr = _make_validation_result(errors=[])
        to_fix = identify_components_to_fix(vr, {})
        assert len(to_fix) == 0

    def test_all_exceeded_max(self):
        errors = [_make_component_error("Task")]
        vr = _make_validation_result(errors=errors)
        history = {"Task": 5}
        to_fix = identify_components_to_fix(vr, history, max_fixes=2)
        assert len(to_fix) == 0


# =============================================================================
# apply_fix_to_project tests
# =============================================================================

class TestApplyFixToProject:
    def test_basic_apply(self):
        tmpdir = tempfile.mkdtemp()
        project_dir = os.path.join(tmpdir, "myproject")
        os.makedirs(project_dir)

        code = dict(SAMPLE_CODE)
        new_code = 'class Task:\n    """Fixed."""\n    def __init__(self):\n        self.id = 0\n'

        updated, manifest = apply_fix_to_project(
            project_dir, "Task", new_code, code, SAMPLE_BLUEPRINT,
        )
        assert updated["Task"] == new_code
        assert updated["TaskManager"] == SAMPLE_CODE["TaskManager"]
        assert manifest is not None

    def test_preserves_other_components(self):
        tmpdir = tempfile.mkdtemp()
        project_dir = os.path.join(tmpdir, "myproject")
        os.makedirs(project_dir)

        code = dict(SAMPLE_CODE)
        updated, _ = apply_fix_to_project(
            project_dir, "Task", "class Task: pass", code, SAMPLE_BLUEPRINT,
        )
        assert updated["TaskManager"] == SAMPLE_CODE["TaskManager"]


# =============================================================================
# serialize_build_result tests
# =============================================================================

class TestSerializeBuildResult:
    def test_success_result(self):
        br = BuildResult(success=True, final_code={"Task": "code"})
        d = serialize_build_result(br)
        assert d["success"] is True
        assert d["iterations"] == []
        assert d["total_fix_attempts"] == 0

    def test_with_iterations(self):
        err = _make_component_error()
        vr = _make_validation_result(errors=[err])
        fa = FixAttempt("Task", 1, "err", "p", "old", "new", True)
        bi = BuildIteration(
            iteration=1,
            validation=vr,
            fixes_attempted=(fa,),
            components_fixed=("Task",),
            components_still_broken=(),
        )
        br = BuildResult(
            success=True,
            iterations=(bi,),
            components_fixed=("Task",),
            total_fix_attempts=1,
        )
        d = serialize_build_result(br)
        assert len(d["iterations"]) == 1
        assert d["iterations"][0]["iteration"] == 1
        assert len(d["iterations"][0]["fixes_attempted"]) == 1
        assert d["iterations"][0]["fixes_attempted"][0]["succeeded"] is True

    def test_serializable(self):
        """Result should be JSON-serializable."""
        import json
        br = BuildResult(
            success=False,
            components_unfixed=("TaskManager",),
            total_fix_attempts=2,
        )
        d = serialize_build_result(br)
        json_str = json.dumps(d)
        assert "TaskManager" in json_str

    def test_component_errors_serialized(self):
        err = _make_component_error("Task", "import", "No module")
        vr = _make_validation_result(errors=[err], unmapped=["mystery error"])
        bi = BuildIteration(iteration=1, validation=vr)
        br = BuildResult(success=False, iterations=(bi,))
        d = serialize_build_result(br)
        assert len(d["iterations"][0]["component_errors"]) == 1
        assert d["iterations"][0]["component_errors"][0]["component"] == "Task"
        assert "mystery error" in d["iterations"][0]["unmapped_errors"]


# =============================================================================
# Integration: protocol_spec BuildSpec
# =============================================================================

class TestBuildSpec:
    def test_protocol_has_build_spec(self):
        from core.protocol_spec import PROTOCOL
        assert hasattr(PROTOCOL, "build")
        assert PROTOCOL.build.max_iterations == 10
        assert PROTOCOL.build.max_fixes_per_component == 5
        assert PROTOCOL.build.subprocess_timeout_seconds == 300
        assert PROTOCOL.build.pip_install_timeout_seconds == 300
        assert PROTOCOL.build.smoke_test_timeout_seconds == 30
        assert PROTOCOL.build.create_venv is True


# =============================================================================
# Integration: exceptions + error_catalog
# =============================================================================

class TestBuildError:
    def test_build_error_creation(self):
        from core.exceptions import BuildError
        err = BuildError("test failure", iteration=2, phase="import")
        assert err.iteration == 2
        assert err.phase == "import"
        assert err.error_code == "E9001"

    def test_build_error_catalog_lookup(self):
        from core.error_catalog import get_entry
        entry = get_entry("E9001")
        assert entry is not None
        assert "validation" in entry.title.lower()

    def test_e9002_catalog(self):
        from core.error_catalog import get_entry
        entry = get_entry("E9002")
        assert entry is not None
        assert "install" in entry.title.lower()

    def test_e9003_catalog(self):
        from core.error_catalog import get_entry
        entry = get_entry("E9003")
        assert entry is not None
        assert "iteration" in entry.title.lower() or "max" in entry.title.lower()

    def test_build_error_to_user_dict(self):
        from core.exceptions import BuildError
        err = BuildError("fail", error_code="E9001")
        d = err.to_user_dict()
        assert d["error_code"] == "E9001"
        assert "error" in d

    def test_build_error_in_catalog_mapping(self):
        from core.error_catalog import get_code_for_exception
        code = get_code_for_exception("BuildError")
        assert code == "E9001"


# =============================================================================
# ENHANCED FIX PROMPT TESTS
# =============================================================================


class TestEnhancedFixPrompt:
    """Tests for build_fix_prompt with generated_code parameter."""

    def test_fix_prompt_adjacent_code_import_error(self):
        """ImportError includes referenced module snippet."""
        error = _make_component_error(
            name="TaskManager",
            error_type="import",
            msg="ImportError: cannot import name 'Task' from 'models'",
        )
        generated = {
            "Task": "class Task:\n    def __init__(self):\n        self.name = ''\n",
            "TaskManager": "from .models import Task\nclass TaskManager:\n    pass\n",
        }
        prompt = build_fix_prompt(
            "TaskManager", error, generated["TaskManager"], SAMPLE_BLUEPRINT,
            generated_code=generated,
        )
        assert "Referenced Module: Task" in prompt
        assert "class Task:" in prompt

    def test_fix_prompt_no_adjacent_syntax_error(self):
        """SyntaxError → no adjacent section."""
        error = _make_component_error(
            name="Task",
            error_type="syntax",
            msg="SyntaxError: invalid syntax",
        )
        prompt = build_fix_prompt(
            "Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT,
            generated_code=SAMPLE_CODE,
        )
        assert "Referenced Module" not in prompt

    def test_fix_prompt_backward_compatible(self):
        """Calling without generated_code works."""
        error = _make_component_error(
            name="Task",
            error_type="import",
            msg="ImportError: No module named 'x'",
        )
        prompt = build_fix_prompt("Task", error, SAMPLE_CODE["Task"], SAMPLE_BLUEPRINT)
        assert "## Component: Task" in prompt
        assert "Referenced Module" not in prompt
