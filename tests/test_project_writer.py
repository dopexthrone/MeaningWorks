"""
Tests for core/project_writer.py — LEAF MODULE.

Phase 1 of Agent Ship: Project Writer tests.
"""

import os
import sys
import tempfile
import pytest

from core.project_writer import (
    ProjectConfig,
    ProjectManifest,
    write_project,
    _infer_project_name,
    _slugify,
    _group_components,
    _resolve_imports,
    _sanitize_concatenated_code,
    _infer_requirements,
    _import_to_pip_name,
    _IMPORT_TO_PIP,
    _RESERVED_VARS,
    _safe_variable_name,
    _generate_main_py,
    _generate_init_py,
    _generate_readme,
    _generate_pyproject_toml,
    _generate_test_stubs,
    _detect_web_framework,
    _to_pascal,
    _to_snake,
    _get_component_type,
    validate_syntax,
    validate_all_code,
    validate_cross_module,
)


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_BLUEPRINT = {
    "domain": "Task Management",
    "core_need": "A system for tracking tasks with deadlines and team assignments.",
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item with deadline"},
        {"name": "Team", "type": "entity", "description": "Group of members"},
        {"name": "TaskManager", "type": "process", "description": "Manages task lifecycle"},
        {"name": "NotificationService", "type": "agent", "description": "Sends notifications"},
    ],
    "relationships": [
        {"from": "TaskManager", "to": "Task", "type": "manages"},
        {"from": "NotificationService", "to": "Team", "type": "notifies"},
    ],
    "constraints": [],
}

SAMPLE_CODE = {
    "Task": 'class Task:\n    """A work item."""\n    def __init__(self):\n        self.title = ""\n        self.deadline = None\n',
    "Team": 'class Team:\n    """Group of members."""\n    def __init__(self):\n        self.members = []\n',
    "TaskManager": 'class TaskManager:\n    """Manages tasks."""\n    def __init__(self):\n        self.tasks = []\n    def create_task(self, task: "Task"):\n        self.tasks.append(task)\n',
    "NotificationService": 'class NotificationService:\n    """Sends notifications."""\n    def notify(self, team: "Team"):\n        pass\n',
}


# =============================================================================
# FROZEN DATACLASS TESTS
# =============================================================================

class TestProjectConfig:
    def test_frozen(self):
        config = ProjectConfig()
        with pytest.raises(AttributeError):
            config.language = "java"

    def test_defaults(self):
        config = ProjectConfig()
        assert config.language == "python"
        assert config.project_name == ""
        assert config.entry_point is True
        assert config.tests is True
        assert config.per_component_files == 6

    def test_custom_values(self):
        config = ProjectConfig(project_name="myapp", tests=False)
        assert config.project_name == "myapp"
        assert config.tests is False


class TestProjectManifest:
    def test_frozen(self):
        manifest = ProjectManifest(
            project_dir="/tmp/test",
            files_written=("a.py",),
            entry_point="main.py",
            total_lines=10,
        )
        with pytest.raises(AttributeError):
            manifest.project_dir = "/other"

    def test_fields(self):
        manifest = ProjectManifest(
            project_dir="/tmp/test",
            files_written=("a.py", "b.py"),
            entry_point="main.py",
            total_lines=42,
        )
        assert manifest.project_dir == "/tmp/test"
        assert len(manifest.files_written) == 2
        assert manifest.total_lines == 42


# =============================================================================
# INFER PROJECT NAME
# =============================================================================

class TestInferProjectName:
    def test_from_domain(self):
        bp = {"domain": "Task Management"}
        assert _infer_project_name(bp) == "task_management"

    def test_from_core_need(self):
        bp = {"core_need": "A booking system for tattoo studios"}
        name = _infer_project_name(bp)
        assert "booking" in name

    def test_empty_blueprint(self):
        assert _infer_project_name({}) == "project"

    def test_domain_with_special_chars(self):
        bp = {"domain": "E-Commerce (v2)"}
        name = _infer_project_name(bp)
        assert name.isidentifier() or name.replace("_", "").isalnum()


class TestSlugify:
    def test_basic(self):
        assert _slugify("Task Management") == "task_management"

    def test_special_chars(self):
        assert _slugify("E-Commerce (v2)") == "e_commerce_v2"

    def test_empty(self):
        assert _slugify("") == "project"

    def test_leading_number(self):
        result = _slugify("123app")
        assert result[0].isalpha()

    def test_already_slug(self):
        assert _slugify("myapp") == "myapp"


# =============================================================================
# COMPONENT GROUPING
# =============================================================================

class TestGroupComponents:
    def test_entities_and_processes(self):
        grouped = _group_components(SAMPLE_CODE, SAMPLE_BLUEPRINT)
        assert "models.py" in grouped
        assert "services.py" in grouped
        assert "Task" in grouped["models.py"]
        assert "Team" in grouped["models.py"]
        assert "TaskManager" in grouped["services.py"]

    def test_per_component_over_threshold(self):
        big_code = {f"Component{i}": f"class Component{i}: pass" for i in range(8)}
        bp = {"components": [{"name": f"Component{i}", "type": "entity"} for i in range(8)]}
        grouped = _group_components(big_code, bp, threshold=6)
        # Should have per-component files
        assert len(grouped) == 8
        for filename in grouped:
            assert filename.endswith(".py")

    def test_under_threshold_groups(self):
        grouped = _group_components(SAMPLE_CODE, SAMPLE_BLUEPRINT, threshold=10)
        assert len(grouped) <= 2

    def test_empty_code(self):
        grouped = _group_components({}, {})
        assert grouped == {}

    def test_single_component(self):
        code = {"Widget": "class Widget: pass"}
        bp = {"components": [{"name": "Widget", "type": "entity"}]}
        grouped = _group_components(code, bp)
        assert len(grouped) >= 1


class TestGetComponentType:
    def test_found(self):
        assert _get_component_type("Task", SAMPLE_BLUEPRINT) == "entity"
        assert _get_component_type("TaskManager", SAMPLE_BLUEPRINT) == "process"

    def test_not_found(self):
        assert _get_component_type("Unknown", SAMPLE_BLUEPRINT) == "entity"


# =============================================================================
# IMPORT RESOLUTION
# =============================================================================

class TestResolveImports:
    def test_cross_references(self):
        grouped = {
            "models.py": {"Task": 'class Task:\n    pass\n'},
            "services.py": {"TaskManager": 'class TaskManager:\n    def run(self, task: Task):\n        pass\n'},
        }
        resolved = _resolve_imports(grouped, ["Task", "TaskManager"])
        # services.py should import Task from models
        assert "from .models import Task" in resolved["services.py"]

    def test_no_cross_references(self):
        grouped = {
            "models.py": {"Task": 'class Task:\n    pass\n'},
            "services.py": {"Handler": 'class Handler:\n    pass\n'},
        }
        resolved = _resolve_imports(grouped, ["Task", "Handler"])
        # No imports needed
        assert "import" not in resolved.get("services.py", "")

    def test_self_reference_ignored(self):
        grouped = {
            "models.py": {"Task": 'class Task:\n    def clone(self) -> Task:\n        pass\n'},
        }
        resolved = _resolve_imports(grouped, ["Task"])
        # No self-import
        assert "import" not in resolved["models.py"]


class TestToPascal:
    def test_already_pascal(self):
        assert _to_pascal("TaskManager") == "TaskManager"

    def test_from_spaces(self):
        assert _to_pascal("task manager") == "TaskManager"

    def test_from_underscores(self):
        assert _to_pascal("task_manager") == "TaskManager"

    def test_single_word(self):
        assert _to_pascal("task") == "Task"


# =============================================================================
# REQUIREMENTS INFERENCE
# =============================================================================

class TestInferRequirements:
    def test_stdlib_filtered(self):
        code = "import os\nimport json\nimport re\n"
        assert _infer_requirements(code) == []

    def test_third_party_detected(self):
        code = "from fastapi import FastAPI\nimport requests\n"
        reqs = _infer_requirements(code)
        assert "fastapi" in reqs
        assert "requests" in reqs

    def test_relative_imports_ignored(self):
        code = "from .models import Task\nfrom ..utils import helper\n"
        assert _infer_requirements(code) == []

    def test_mixed(self):
        code = "import os\nfrom flask import Flask\nimport json\n"
        reqs = _infer_requirements(code)
        assert "flask" in reqs
        assert "os" not in reqs
        assert "json" not in reqs

    def test_empty_code(self):
        assert _infer_requirements("") == []


# =============================================================================
# FILE GENERATORS
# =============================================================================

class TestDetectWebFramework:
    def test_fastapi(self):
        assert _detect_web_framework("from fastapi import FastAPI") == "fastapi"

    def test_flask(self):
        assert _detect_web_framework("from flask import Flask") == "flask"

    def test_django(self):
        assert _detect_web_framework("from django.db import models") == "django"

    def test_no_framework(self):
        assert _detect_web_framework("import os\nclass Foo: pass") is None


class TestGenerateMainPy:
    def test_web_app(self):
        code = "from fastapi import FastAPI\napp = FastAPI()"
        main = _generate_main_py(SAMPLE_BLUEPRINT, ["Task"], code)
        assert "uvicorn" in main
        assert 'if __name__' in main

    def test_non_web_with_components(self):
        code = "class Task: pass"
        grouped = {"models.py": {"Task": code}}
        main = _generate_main_py(SAMPLE_BLUEPRINT, ["Task"], code, grouped)
        assert 'if __name__' in main
        # Should import from models
        assert "from models import" in main
        assert "Task" in main

    def test_non_web_no_grouped(self):
        """Without grouped_files, no imports but still runs."""
        code = "class Task: pass"
        main = _generate_main_py(SAMPLE_BLUEPRINT, ["Task"], code)
        assert 'if __name__' in main
        # Has component instantiation
        assert "task = Task()" in main

    def test_flask(self):
        code = "from flask import Flask\napp = Flask(__name__)"
        main = _generate_main_py(SAMPLE_BLUEPRINT, ["Task"], code)
        assert "app.run" in main

    def test_wires_entities_and_processes(self):
        """main.py instantiates both entity and process components."""
        code = "class Task: pass\nclass TaskManager: pass"
        names = ["Task", "TaskManager"]
        grouped = {"models.py": {"Task": "class Task: pass"}, "services.py": {"TaskManager": "class TaskManager: pass"}}
        main = _generate_main_py(SAMPLE_BLUEPRINT, names, code, grouped)
        assert "task = Task()" in main
        assert "task_manager = TaskManager()" in main
        assert "from models import Task" in main
        assert "from services import TaskManager" in main

    def test_empty_components(self):
        """No components → fallback print."""
        bp = {"domain": "test", "components": []}
        main = _generate_main_py(bp, [], "")
        assert "is running" in main


class TestToSnake:
    def test_pascal(self):
        assert _to_snake("TaskManager") == "task_manager"

    def test_simple(self):
        assert _to_snake("Task") == "task"

    def test_multi_upper(self):
        # All-caps prefix stays together (HTMLParser → htmlparser)
        assert _to_snake("HTMLParser") == "htmlparser"

    def test_already_lower(self):
        assert _to_snake("task") == "task"


class TestPipNameMapping:
    def test_pil_to_pillow(self):
        assert _import_to_pip_name("PIL") == "Pillow"

    def test_cv2_to_opencv(self):
        assert _import_to_pip_name("cv2") == "opencv-python"

    def test_sklearn_to_scikit(self):
        assert _import_to_pip_name("sklearn") == "scikit-learn"

    def test_yaml_to_pyyaml(self):
        assert _import_to_pip_name("yaml") == "PyYAML"

    def test_unknown_passthrough(self):
        assert _import_to_pip_name("requests") == "requests"

    def test_infer_requirements_uses_pip_names(self):
        code = "from PIL import Image\nimport cv2\nimport requests\n"
        reqs = _infer_requirements(code)
        assert "Pillow" in reqs
        assert "opencv-python" in reqs
        assert "requests" in reqs
        # Should NOT have import names
        assert "PIL" not in reqs
        assert "cv2" not in reqs

    def test_mapping_table_not_empty(self):
        assert len(_IMPORT_TO_PIP) > 10  # Reasonable set of mappings


class TestSyntaxValidation:
    def test_valid_code(self):
        assert validate_syntax("class Foo: pass") is None

    def test_invalid_code(self):
        err = validate_syntax("class Foo:\n  def bar(self\n    pass")
        assert err is not None

    def test_error_includes_line_number(self):
        err = validate_syntax("x = 1\ny = \n", filename="test.py")
        assert err is not None
        assert "test.py" in err

    def test_empty_code(self):
        assert validate_syntax("") is None

    def test_validate_all_code_all_valid(self):
        code = {"Foo": "class Foo: pass", "Bar": "class Bar: pass"}
        assert validate_all_code(code) == []

    def test_validate_all_code_has_errors(self):
        code = {"Foo": "class Foo: pass", "Bad": "def (\n"}
        errors = validate_all_code(code)
        assert len(errors) == 1
        assert "Bad.py" in errors[0]


class TestGenerateInitPy:
    def test_with_exports(self):
        content = _generate_init_py(["Task", "Team"])
        assert "__all__" in content
        assert "Task" in content

    def test_empty_exports(self):
        assert _generate_init_py([]) == ""


class TestGenerateReadme:
    def test_has_domain(self):
        readme = _generate_readme(SAMPLE_BLUEPRINT, "task_management")
        assert "Task Management" in readme

    def test_has_components(self):
        readme = _generate_readme(SAMPLE_BLUEPRINT, "task_management")
        assert "Task" in readme
        assert "Components" in readme

    def test_has_getting_started(self):
        readme = _generate_readme(SAMPLE_BLUEPRINT, "test")
        assert "Getting Started" in readme


class TestGeneratePyprojectToml:
    def test_basic(self):
        toml = _generate_pyproject_toml("myapp", ["fastapi", "requests"])
        assert "myapp" in toml
        assert "fastapi" in toml

    def test_no_deps(self):
        toml = _generate_pyproject_toml("myapp", [])
        assert "myapp" in toml
        assert "dependencies = []" in toml


class TestGenerateTestStubs:
    def test_generates_files(self):
        grouped = {"models.py": {"Task": "class Task: pass", "Team": "class Team: pass"}}
        stubs = _generate_test_stubs(grouped)
        assert "test_models.py" in stubs
        assert "TestTask" in stubs["test_models.py"]
        assert "TestTeam" in stubs["test_models.py"]

    def test_empty(self):
        assert _generate_test_stubs({}) == {}


# =============================================================================
# INTEGRATION: write_project
# =============================================================================

class TestWriteProject:
    def test_writes_complete_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)

            assert os.path.isdir(manifest.project_dir)
            assert manifest.entry_point == "main.py"
            assert manifest.total_lines > 0
            assert len(manifest.files_written) > 0

            # Check key files exist
            for f in ["main.py", "__init__.py", "requirements.txt",
                      "pyproject.toml", "README.md"]:
                assert f in manifest.files_written, f"Missing {f}"
                filepath = os.path.join(manifest.project_dir, f)
                assert os.path.isfile(filepath), f"File not on disk: {f}"

    def test_custom_project_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(project_name="my_app")
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config)
            assert manifest.project_dir.endswith("my_app")

    def test_no_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(tests=False)
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config)
            # No test files
            assert not any(f.startswith("tests/") for f in manifest.files_written
                           if f != "tests/__init__.py")
            # Actually, with tests=False, no tests/ at all
            test_dir = os.path.join(manifest.project_dir, "tests")
            assert not os.path.isdir(test_dir)

    def test_no_entry_point(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProjectConfig(entry_point=False)
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir, config)
            assert "main.py" not in manifest.files_written

    def test_empty_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project({}, SAMPLE_BLUEPRINT, tmpdir)
            assert os.path.isdir(manifest.project_dir)
            # Should still produce basic files
            assert "README.md" in manifest.files_written

    def test_files_have_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            readme_path = os.path.join(manifest.project_dir, "README.md")
            with open(readme_path) as f:
                content = f.read()
            assert "Task Management" in content

    def test_models_and_services_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            assert "models.py" in manifest.files_written
            assert "services.py" in manifest.files_written

    def test_manifest_frozen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            with pytest.raises(AttributeError):
                manifest.project_dir = "/other"

    def test_file_contents_populated(self):
        """file_contents has all written files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            assert isinstance(manifest.file_contents, dict)
            assert len(manifest.file_contents) > 0
            # All files_written should have content entries
            for f in manifest.files_written:
                assert f in manifest.file_contents, f"Missing content for {f}"

    def test_file_contents_match_disk(self):
        """file_contents values match what's on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            for filename, content in manifest.file_contents.items():
                filepath = os.path.join(manifest.project_dir, filename)
                with open(filepath) as f:
                    disk_content = f.read()
                assert content == disk_content, f"Content mismatch for {filename}"

    def test_main_py_has_imports(self):
        """main.py should import from source modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            main_content = manifest.file_contents.get("main.py", "")
            assert "from models import" in main_content
            assert "from services import" in main_content

    def test_main_py_has_component_wiring(self):
        """main.py instantiates components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(SAMPLE_CODE, SAMPLE_BLUEPRINT, tmpdir)
            main_content = manifest.file_contents.get("main.py", "")
            # Should instantiate at least one component
            assert "= Task()" in main_content or "= Team()" in main_content

    def test_requirements_uses_pip_names(self):
        """requirements.txt has pip names, not import names."""
        code_with_pil = {
            "ImageProcessor": "from PIL import Image\nimport cv2\nclass ImageProcessor: pass\n",
        }
        bp = {"domain": "test", "components": [{"name": "ImageProcessor", "type": "process"}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(code_with_pil, bp, tmpdir)
            req_content = manifest.file_contents.get("requirements.txt", "")
            assert "Pillow" in req_content
            assert "opencv-python" in req_content
            assert "PIL" not in req_content


# =============================================================================
# LEAF MODULE CONSTRAINT
# =============================================================================

class TestLeafModuleConstraint:
    def test_no_engine_imports(self):
        import core.project_writer as mod
        source_file = mod.__file__
        with open(source_file) as f:
            source = f.read()
        # No imports from engine, protocol, pipeline, llm
        for forbidden in ["from core.engine", "from core.protocol",
                          "from core.pipeline", "from core.llm"]:
            assert forbidden not in source, f"LEAF MODULE violated: found '{forbidden}'"

    def test_stdlib_only_imports(self):
        import core.project_writer as mod
        source_file = mod.__file__
        with open(source_file) as f:
            source = f.read()
        # Only stdlib imports allowed
        import re as _re
        imports = _re.findall(r'^(?:from|import)\s+(\S+)', source, _re.MULTILINE)
        allowed_prefixes = {"os", "re", "sys", "ast", "json", "shutil", "dataclasses", "typing", "core"}
        for imp in imports:
            top = imp.split('.')[0]
            assert top in allowed_prefixes, f"Non-stdlib import: {imp}"


# =============================================================================
# CROSS-MODULE VALIDATION TESTS
# =============================================================================


class TestValidateCrossModule:
    """Tests for validate_cross_module()."""

    def test_validate_cross_module_valid(self):
        """All imports resolve → empty warnings."""
        resolved = {
            "models.py": "class Task:\n    pass\n\nclass Team:\n    pass\n",
            "services.py": "from .models import Task, Team\n\nclass TaskManager:\n    pass\n",
        }
        warnings = validate_cross_module(resolved)
        assert warnings == []

    def test_validate_cross_module_missing_name(self):
        """Missing name → warning returned."""
        resolved = {
            "models.py": "class Task:\n    pass\n",
            "services.py": "from .models import Task, Team\n\nclass TaskManager:\n    pass\n",
        }
        warnings = validate_cross_module(resolved)
        assert len(warnings) == 1
        assert "Team" in warnings[0]
        assert "models" in warnings[0]

    def test_validate_cross_module_skips_non_python(self):
        """Non-.py → empty list."""
        resolved = {
            "models.yaml": "some: yaml",
            "services.yaml": "from .models import Task",
        }
        warnings = validate_cross_module(resolved, file_extension=".yaml")
        assert warnings == []

    def test_manifest_has_cross_module_warnings(self):
        """ProjectManifest.cross_module_warnings populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = {
                "Task": "class Task:\n    pass\n",
                "TaskManager": "class TaskManager:\n    def run(self, task: MissingRef): pass\n",
            }
            blueprint = {
                "domain": "test",
                "core_need": "test",
                "components": [
                    {"name": "Task", "type": "entity", "description": ""},
                    {"name": "TaskManager", "type": "process", "description": ""},
                ],
            }
            manifest = write_project(code, blueprint, tmpdir)
            # cross_module_warnings is a tuple (may be empty or populated)
            assert isinstance(manifest.cross_module_warnings, tuple)


# =============================================================================
# SANITIZE CONCATENATED CODE TESTS
# =============================================================================


class TestSanitizeConcatenatedCode:
    """Tests for _sanitize_concatenated_code() — sterile build Bug 1 fix."""

    def test_future_imports_hoisted(self):
        """from __future__ mid-file → moved to top."""
        code = (
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Foo:\n"
            "    pass\n"
            "\n"
            "from __future__ import annotations\n"
            "\n"
            "@dataclass\n"
            "class Bar:\n"
            "    x: int = 0\n"
        )
        result = _sanitize_concatenated_code(code)
        lines = result.split("\n")
        # __future__ must be first non-empty line
        non_empty = [l for l in lines if l.strip()]
        assert non_empty[0] == "from __future__ import annotations"

    def test_future_imports_deduplicated(self):
        """Multiple from __future__ import annotations → one."""
        code = (
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "\n"
            "class Foo:\n"
            "    pass\n"
            "\n"
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "\n"
            "class Bar:\n"
            "    pass\n"
        )
        result = _sanitize_concatenated_code(code)
        assert result.count("from __future__ import annotations") == 1

    def test_duplicate_imports_removed(self):
        """Repeated import lines from concatenation → deduplicated."""
        code = (
            "from dataclasses import dataclass\n"
            "from typing import Any\n"
            "\n"
            "class Foo:\n"
            "    pass\n"
            "\n"
            "from dataclasses import dataclass\n"
            "from typing import Any\n"
            "\n"
            "class Bar:\n"
            "    pass\n"
        )
        result = _sanitize_concatenated_code(code)
        assert result.count("from dataclasses import dataclass") == 1

    def test_no_future_unchanged(self):
        """Code without __future__ → body preserved."""
        code = "class Foo:\n    pass\n"
        result = _sanitize_concatenated_code(code)
        assert "class Foo:" in result

    def test_result_is_valid_python(self):
        """Sanitized output is parseable Python."""
        import ast
        code = (
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Foo:\n"
            "    x: int = 0\n"
            "\n"
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Bar:\n"
            "    y: str = ''\n"
        )
        result = _sanitize_concatenated_code(code)
        # Must parse without SyntaxError
        ast.parse(result)


# =============================================================================
# RESERVED VARIABLE NAME SHADOWING TESTS
# =============================================================================


class TestReservedVars:
    """Tests for _RESERVED_VARS and _safe_variable_name()."""

    def test_reserved_vars_contains_infrastructure_names(self):
        for name in ("state", "llm", "tools", "runtime", "config", "logging", "asyncio", "port"):
            assert name in _RESERVED_VARS

    def test_safe_variable_name_avoids_llm(self):
        assert _safe_variable_name("Llm") == "llm_component"

    def test_safe_variable_name_avoids_state(self):
        assert _safe_variable_name("State") == "state_component"

    def test_safe_variable_name_avoids_runtime(self):
        assert _safe_variable_name("Runtime") == "runtime_component"

    def test_safe_variable_name_avoids_config(self):
        assert _safe_variable_name("Config") == "config_component"

    def test_safe_variable_name_passes_normal(self):
        assert _safe_variable_name("ChatAgent") == "chat_agent"
        assert _safe_variable_name("TaskManager") == "task_manager"

    def test_safe_variable_name_passes_non_reserved(self):
        assert _safe_variable_name("MessageHandler") == "message_handler"

    def test_async_main_no_shadowing(self):
        """Async main.py should not shadow 'llm' with a component variable."""
        from core.domain_adapter import RuntimeCapabilities
        rt = RuntimeCapabilities(
            has_event_loop=True, has_llm_client=True,
            has_persistent_state=True, has_tool_execution=True,
            event_loop_type="asyncio", state_backend="sqlite",
            default_port=8080,
        )
        bp = {
            "domain": "test",
            "core_need": "test",
            "components": [
                {"name": "LLM", "type": "agent", "description": "An LLM agent"},
            ],
        }
        code = _generate_main_py(bp, ["LLM"], "", {"services.py": {"LLM": "class LLM: pass"}},
                                  runtime_capabilities=rt)
        # The variable for the LLM component should NOT be plain 'llm'
        assert "llm_component = LLM(" in code
        # Infrastructure 'llm = LLMClient(...)' should still exist
        assert "llm = LLMClient(" in code


# =============================================================================
# MOTHER AGENT WIRING TESTS
# =============================================================================

class TestMotherAgentWiring:
    """Test project writer wires compiler/tool_manager into async main.py."""

    def _make_mother_rt(self):
        from core.domain_adapter import RuntimeCapabilities
        return RuntimeCapabilities(
            has_event_loop=True, has_llm_client=True,
            has_persistent_state=True, has_tool_execution=True,
            has_self_recompile=True,
            event_loop_type="asyncio", state_backend="sqlite",
            default_port=8080,
            can_compile=True, can_share_tools=True,
            corpus_path="~/motherlabs/corpus.db",
        )

    def test_async_main_imports_compiler(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "from compiler import ToolCompiler" in code

    def test_async_main_imports_tool_manager(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "from tool_manager import ToolManager" in code
        assert "from motherlabs_platform.instance_identity import InstanceIdentityStore" in code

    def test_async_main_initializes_compiler(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "compiler = ToolCompiler(" in code
        assert "~/motherlabs/corpus.db" in code

    def test_async_main_initializes_tool_manager(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "tool_manager = ToolManager(" in code
        assert "instance_id=instance_id" in code

    def test_async_main_passes_compiler_to_runtime(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "compiler=compiler" in code
        assert "tool_manager=tool_manager" in code

    def test_async_main_identity_setup(self):
        rt = self._make_mother_rt()
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent"}]}
        code = _generate_main_py(bp, ["Agent"], "", {"services.py": {"Agent": "class Agent: pass"}},
                                  runtime_capabilities=rt)
        assert "identity_store = InstanceIdentityStore()" in code
        assert "identity = identity_store.get_or_create_self()" in code
        assert "instance_id = identity.instance_id" in code

    def test_reserved_vars_include_mother_names(self):
        for name in ("compiler", "tool_manager", "instance_id", "identity", "identity_store"):
            assert name in _RESERVED_VARS, f"Missing reserved var: {name}"

    def test_write_project_emits_scaffold_files(self):
        """write_project emits compiler.py and tool_manager.py for mother agents."""
        import tempfile
        rt = self._make_mother_rt()
        config = ProjectConfig(runtime_capabilities=rt)
        bp = {"domain": "test", "core_need": "test",
              "components": [{"name": "Agent", "type": "agent", "description": "test"}]}
        code = {"Agent": "class Agent:\n    async def handle(self, msg): return {}\n"}
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(code, bp, tmpdir, config)
            assert "compiler.py" in manifest.files_written
            assert "tool_manager.py" in manifest.files_written
            # Verify files have content
            assert len(manifest.file_contents.get("compiler.py", "")) > 0
            assert len(manifest.file_contents.get("tool_manager.py", "")) > 0
