"""
Tests for core/runtime_validator.py.

Phase 27: Runtime Build Loop — runtime validation tests.
"""

import os
import sys
import tempfile
import textwrap
import pytest
from unittest.mock import patch, MagicMock

from core.runtime_validator import (
    RuntimeConfig,
    CommandResult,
    ComponentError,
    ValidationResult,
    create_project_venv,
    run_command,
    install_dependencies,
    check_imports,
    run_tests,
    run_smoke_test,
    parse_traceback,
    map_errors_to_components,
    validate_project,
)


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_BLUEPRINT = {
    "domain": "Task Management",
    "core_need": "A task management system.",
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item"},
        {"name": "TaskManager", "type": "process", "description": "Manages tasks"},
    ],
    "relationships": [
        {"from": "TaskManager", "to": "Task", "type": "manages"},
    ],
    "constraints": [],
}

SAMPLE_CODE = {
    "Task": 'class Task:\n    """A work item."""\n    def __init__(self):\n        self.name = ""\n',
    "TaskManager": 'class TaskManager:\n    """Manages tasks."""\n    def __init__(self):\n        self.tasks = []\n',
}


def _make_project(code_dict=None, add_tests=True, add_requirements=True):
    """Create a temporary project directory with generated code."""
    tmpdir = tempfile.mkdtemp()
    code = code_dict or SAMPLE_CODE

    for name, content in code.items():
        filepath = os.path.join(tmpdir, f"{name.lower()}.py")
        with open(filepath, "w") as f:
            f.write(content)

    # __init__.py
    with open(os.path.join(tmpdir, "__init__.py"), "w") as f:
        f.write("")

    if add_requirements:
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("")

    if add_tests:
        tests_dir = os.path.join(tmpdir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        with open(os.path.join(tests_dir, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(tests_dir, "test_stub.py"), "w") as f:
            f.write('def test_pass():\n    assert True\n')

    return tmpdir


# =============================================================================
# RuntimeConfig tests
# =============================================================================

class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.subprocess_timeout_seconds == 60
        assert cfg.pip_install_timeout_seconds == 120
        assert cfg.smoke_test_timeout_seconds == 10
        assert cfg.install_deps is True
        assert cfg.run_tests is True
        assert cfg.run_smoke_test is False
        assert cfg.create_venv is True

    def test_custom_values(self):
        cfg = RuntimeConfig(subprocess_timeout_seconds=30, create_venv=False)
        assert cfg.subprocess_timeout_seconds == 30
        assert cfg.create_venv is False

    def test_frozen(self):
        cfg = RuntimeConfig()
        with pytest.raises(AttributeError):
            cfg.create_venv = False


# =============================================================================
# CommandResult tests
# =============================================================================

class TestCommandResult:
    def test_defaults(self):
        cr = CommandResult(command="echo", returncode=0, stdout="ok", stderr="")
        assert cr.timed_out is False
        assert cr.duration == 0.0

    def test_timed_out(self):
        cr = CommandResult(command="sleep", returncode=-1, stdout="", stderr="timeout", timed_out=True)
        assert cr.timed_out is True

    def test_frozen(self):
        cr = CommandResult(command="x", returncode=0, stdout="", stderr="")
        with pytest.raises(AttributeError):
            cr.returncode = 1


# =============================================================================
# ComponentError tests
# =============================================================================

class TestComponentError:
    def test_basic(self):
        err = ComponentError(
            component_name="Task",
            error_type="import",
            error_message="No module named 'task'",
        )
        assert err.component_name == "Task"
        assert err.error_type == "import"
        assert err.file_path == ""
        assert err.line_number == 0

    def test_with_file_info(self):
        err = ComponentError(
            component_name="Task",
            error_type="syntax",
            error_message="invalid syntax",
            file_path="/tmp/task.py",
            line_number=42,
        )
        assert err.file_path == "/tmp/task.py"
        assert err.line_number == 42


# =============================================================================
# ValidationResult tests
# =============================================================================

class TestValidationResult:
    def test_success(self):
        vr = ValidationResult(success=True)
        assert vr.success is True
        assert vr.component_errors == ()
        assert vr.unmapped_errors == ()

    def test_with_errors(self):
        err = ComponentError("Task", "import", "failed")
        vr = ValidationResult(
            success=False,
            component_errors=(err,),
            unmapped_errors=("unknown error",),
        )
        assert len(vr.component_errors) == 1
        assert len(vr.unmapped_errors) == 1


# =============================================================================
# run_command tests
# =============================================================================

class TestRunCommand:
    def test_successful_command(self):
        result = run_command([sys.executable, "-c", "print('hello')"], cwd="/tmp")
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert result.timed_out is False

    def test_failing_command(self):
        result = run_command([sys.executable, "-c", "import sys; sys.exit(1)"], cwd="/tmp")
        assert result.returncode == 1

    def test_timeout(self):
        result = run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd="/tmp",
            timeout=1,
        )
        assert result.timed_out is True
        assert result.returncode == -1

    def test_command_not_found(self):
        result = run_command(["nonexistent_command_xyz"], cwd="/tmp")
        assert result.returncode == -1
        assert "not found" in result.stderr.lower() or "Command not found" in result.stderr

    def test_duration_recorded(self):
        result = run_command([sys.executable, "-c", "pass"], cwd="/tmp")
        assert result.duration >= 0

    def test_env_override(self):
        result = run_command(
            [sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR', ''))"],
            cwd="/tmp",
            env={"TEST_VAR": "hello_test"},
        )
        assert result.returncode == 0
        assert "hello_test" in result.stdout


# =============================================================================
# install_dependencies tests
# =============================================================================

class TestInstallDependencies:
    def test_no_requirements_file(self):
        tmpdir = tempfile.mkdtemp()
        result = install_dependencies(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "No requirements.txt" in result.stdout

    def test_empty_requirements(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("")
        result = install_dependencies(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "Empty" in result.stdout

    def test_whitespace_only_requirements(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("   \n  \n")
        result = install_dependencies(sys.executable, tmpdir)
        assert result.returncode == 0


# =============================================================================
# check_imports tests
# =============================================================================

class TestCheckImports:
    def test_valid_imports(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "hello.py"), "w") as f:
            f.write("x = 1\n")
        result = check_imports(sys.executable, tmpdir)
        assert result.returncode == 0

    def test_no_python_files(self):
        tmpdir = tempfile.mkdtemp()
        result = check_imports(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "No Python files" in result.stdout

    def test_import_error(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "bad.py"), "w") as f:
            f.write("import nonexistent_module_xyz\n")
        result = check_imports(sys.executable, tmpdir)
        assert result.returncode != 0


# =============================================================================
# run_tests tests
# =============================================================================

class TestRunTests:
    def test_no_tests_dir(self):
        tmpdir = tempfile.mkdtemp()
        result = run_tests(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "No tests/" in result.stdout

    def test_passing_tests(self):
        tmpdir = _make_project()
        result = run_tests(sys.executable, tmpdir)
        assert result.returncode == 0


# =============================================================================
# run_smoke_test tests
# =============================================================================

class TestRunSmokeTest:
    def test_no_main_py(self):
        tmpdir = tempfile.mkdtemp()
        result = run_smoke_test(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "No main.py" in result.stdout

    def test_successful_main(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write('print("running")\n')
        result = run_smoke_test(sys.executable, tmpdir)
        assert result.returncode == 0
        assert "running" in result.stdout

    def test_failing_main(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write('raise ValueError("boom")\n')
        result = run_smoke_test(sys.executable, tmpdir)
        assert result.returncode != 0


# =============================================================================
# parse_traceback tests
# =============================================================================

class TestParseTraceback:
    def test_import_error(self):
        output = textwrap.dedent('''\
            Traceback (most recent call last):
              File "/tmp/project/task.py", line 1, in <module>
                import nonexistent
            ModuleNotFoundError: No module named 'nonexistent'
        ''')
        results = parse_traceback(output)
        assert len(results) >= 1
        filename, line, msg = results[0]
        assert "task.py" in filename
        assert "ModuleNotFoundError" in msg

    def test_syntax_error(self):
        output = textwrap.dedent('''\
              File "/tmp/project/bad.py", line 5
                def foo(
                       ^
            SyntaxError: unexpected EOF while parsing
        ''')
        results = parse_traceback(output)
        assert len(results) >= 1
        assert "SyntaxError" in results[0][2]

    def test_attribute_error(self):
        output = textwrap.dedent('''\
            Traceback (most recent call last):
              File "/tmp/project/main.py", line 10, in <module>
                obj.missing_method()
            AttributeError: 'Task' object has no attribute 'missing_method'
        ''')
        results = parse_traceback(output)
        assert len(results) >= 1
        assert "AttributeError" in results[0][2]

    def test_no_traceback(self):
        results = parse_traceback("Everything is fine")
        assert results == []

    def test_empty_input(self):
        results = parse_traceback("")
        assert results == []

    def test_multiple_errors(self):
        output = textwrap.dedent('''\
            Traceback (most recent call last):
              File "/tmp/a.py", line 1
            ImportError: No module named 'x'
              File "/tmp/b.py", line 2
            NameError: name 'y' is not defined
        ''')
        results = parse_traceback(output)
        assert len(results) >= 2


# =============================================================================
# map_errors_to_components tests
# =============================================================================

class TestMapErrorsToComponents:
    def test_filename_match(self):
        errors = [("/tmp/project/task.py", 1, "ImportError: No module named 'x'")]
        mapped, unmapped = map_errors_to_components(errors, SAMPLE_CODE, SAMPLE_BLUEPRINT)
        assert len(mapped) == 1
        assert mapped[0].component_name == "Task"
        assert mapped[0].error_type == "import"

    def test_error_message_match(self):
        errors = [("", 0, "NameError: name 'TaskManager' is not defined")]
        mapped, unmapped = map_errors_to_components(errors, SAMPLE_CODE, SAMPLE_BLUEPRINT)
        assert len(mapped) == 1
        assert mapped[0].component_name == "TaskManager"

    def test_no_match(self):
        errors = [("/tmp/unknown.py", 1, "SyntaxError: invalid syntax")]
        mapped, unmapped = map_errors_to_components(errors, SAMPLE_CODE, SAMPLE_BLUEPRINT)
        assert len(mapped) == 0
        assert len(unmapped) == 1

    def test_empty_errors(self):
        mapped, unmapped = map_errors_to_components([], SAMPLE_CODE, SAMPLE_BLUEPRINT)
        assert len(mapped) == 0
        assert len(unmapped) == 0

    def test_syntax_error_type(self):
        errors = [("/tmp/task.py", 5, "SyntaxError: invalid syntax")]
        mapped, unmapped = map_errors_to_components(errors, SAMPLE_CODE, SAMPLE_BLUEPRINT)
        if mapped:
            assert mapped[0].error_type == "syntax"

    def test_test_error_type(self):
        errors = [("/tmp/task.py", 10, "FAILED test_task :: AssertionError: assert False")]
        mapped, unmapped = map_errors_to_components(errors, SAMPLE_CODE, SAMPLE_BLUEPRINT)
        if mapped:
            assert mapped[0].error_type == "test"


# =============================================================================
# validate_project tests (mocked venv)
# =============================================================================

class TestValidateProject:
    def test_successful_validation_no_venv(self):
        """Validate a simple project without creating a venv."""
        tmpdir = _make_project()
        config = RuntimeConfig(create_venv=False, install_deps=False, run_smoke_test=False)
        result = validate_project(tmpdir, SAMPLE_CODE, SAMPLE_BLUEPRINT, config)
        assert result.success is True

    def test_import_failure_no_venv(self):
        """Detect import errors without venv."""
        # Create project WITHOUT __init__.py so check_imports tries per-file imports
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "bad_module.py"), "w") as f:
            f.write("import nonexistent_module_xyz_abc\n")
        bad_code = {"bad_module": "import nonexistent_module_xyz_abc\n"}
        config = RuntimeConfig(create_venv=False, install_deps=False, run_tests=False)
        result = validate_project(tmpdir, bad_code, SAMPLE_BLUEPRINT, config)
        # The import check should have failed (non-zero returncode)
        assert result.import_check is not None
        assert result.import_check.returncode != 0

    def test_venv_creation_failure(self):
        """Handle venv creation failure gracefully."""
        config = RuntimeConfig(create_venv=True)
        with patch("core.runtime_validator.create_project_venv", side_effect=Exception("venv failed")):
            result = validate_project("/nonexistent", SAMPLE_CODE, SAMPLE_BLUEPRINT, config)
            assert result.success is False
            assert "Venv creation failed" in result.unmapped_errors[0]

    def test_all_steps_pass(self):
        """All validation steps pass."""
        tmpdir = _make_project()
        config = RuntimeConfig(create_venv=False, install_deps=True, run_tests=True, run_smoke_test=False)
        result = validate_project(tmpdir, SAMPLE_CODE, SAMPLE_BLUEPRINT, config)
        assert result.success is True
        assert result.import_check is not None
        assert result.dep_install is not None

    def test_validation_result_fields(self):
        """ValidationResult has all expected fields."""
        tmpdir = _make_project()
        config = RuntimeConfig(create_venv=False, install_deps=False, run_tests=False)
        result = validate_project(tmpdir, SAMPLE_CODE, SAMPLE_BLUEPRINT, config)
        assert result.dep_install is None
        assert result.import_check is not None
        assert result.test_run is None
        assert result.smoke_test is None
        assert result.venv_path == ""


# =============================================================================
# create_project_venv tests
# =============================================================================

class TestCreateProjectVenv:
    def test_creates_venv(self):
        """Venv is created and python path exists."""
        tmpdir = tempfile.mkdtemp()
        python_path = create_project_venv(tmpdir)
        assert os.path.exists(python_path)
        assert ".venv" in python_path

    def test_venv_has_pip(self):
        """Venv python can run pip."""
        tmpdir = tempfile.mkdtemp()
        python_path = create_project_venv(tmpdir)
        result = run_command([python_path, "-m", "pip", "--version"], cwd=tmpdir)
        assert result.returncode == 0
        assert "pip" in result.stdout
