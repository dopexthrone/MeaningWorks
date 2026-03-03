"""Tests for mother.integrator module."""

import json
from pathlib import Path

import pytest

from mother.integrator import (
    extract_code_files,
    extract_project_metadata,
    validate_project_structure,
)


@pytest.fixture
def valid_project(tmp_path):
    """Create a valid project structure."""
    project = tmp_path / "test_project"
    project.mkdir()

    # Entry point
    (project / "main.py").write_text("print('hello')")

    # Tests
    tests = project / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_main(): pass")

    return project


@pytest.fixture
def project_with_blueprint(tmp_path):
    """Create project with blueprint.json."""
    project = tmp_path / "bp_project"
    project.mkdir()

    # Entry point
    (project / "app.py").write_text("print('app')")

    # Tests
    tests = project / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_app(): pass")

    # Blueprint
    blueprint = {
        "name": "test-app",
        "description": "A test application",
        "domain": "api",
        "components": [
            {"name": "server", "type": "service"},
            {"name": "client", "type": "tool"},
        ],
    }
    (project / "blueprint.json").write_text(json.dumps(blueprint))

    return project


@pytest.fixture
def project_with_pyproject(tmp_path):
    """Create project with pyproject.toml."""
    project = tmp_path / "py_project"
    project.mkdir()

    # Entry point
    (project / "main.py").write_text("print('main')")

    # Tests
    tests = project / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_main(): pass")

    # pyproject.toml
    toml_content = """
[project]
name = "my-project"
description = "My awesome project"
version = "1.0.0"
"""
    (project / "pyproject.toml").write_text(toml_content)

    return project


class TestValidateProjectStructure:
    """Test validate_project_structure function."""

    def test_valid_project_with_main(self, valid_project):
        """Valid project with main.py passes validation."""
        valid, error = validate_project_structure(valid_project)
        assert valid
        assert error == ""

    def test_valid_project_with_app(self, tmp_path):
        """Valid project with app.py passes validation."""
        project = tmp_path / "app_project"
        project.mkdir()
        (project / "app.py").write_text("print('app')")
        tests = project / "tests"
        tests.mkdir()
        (tests / "test_app.py").write_text("def test(): pass")

        valid, error = validate_project_structure(project)
        assert valid
        assert error == ""

    def test_valid_project_with_single_py(self, tmp_path):
        """Valid project with single .py file passes validation."""
        project = tmp_path / "single_project"
        project.mkdir()
        (project / "script.py").write_text("print('script')")
        tests = project / "tests"
        tests.mkdir()
        (tests / "test_script.py").write_text("def test(): pass")

        valid, error = validate_project_structure(project)
        assert valid
        assert error == ""

    def test_missing_path(self, tmp_path):
        """Non-existent path fails validation."""
        project = tmp_path / "nonexistent"

        valid, error = validate_project_structure(project)
        assert not valid
        assert "does not exist" in error

    def test_path_is_file(self, tmp_path):
        """File instead of directory fails validation."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        valid, error = validate_project_structure(file_path)
        assert not valid
        assert "not a directory" in error

    def test_missing_entry_point(self, tmp_path):
        """Project without entry point fails validation."""
        project = tmp_path / "no_entry"
        project.mkdir()
        tests = project / "tests"
        tests.mkdir()

        valid, error = validate_project_structure(project)
        assert not valid
        assert "No entry point" in error

    def test_missing_tests(self, tmp_path):
        """Project without tests/ fails validation."""
        project = tmp_path / "no_tests"
        project.mkdir()
        (project / "main.py").write_text("print('main')")

        valid, error = validate_project_structure(project)
        assert not valid
        assert "No tests/" in error


class TestExtractProjectMetadata:
    """Test extract_project_metadata function."""

    def test_from_blueprint_json(self, project_with_blueprint):
        """Extract metadata from blueprint.json."""
        metadata = extract_project_metadata(project_with_blueprint)

        assert metadata["name"] == "test-app"
        assert metadata["description"] == "A test application"
        assert metadata["domain"] == "api"
        assert len(metadata["components"]) == 2
        assert metadata["entry_point"] == "app.py"

    def test_from_pyproject_toml(self, project_with_pyproject):
        """Extract metadata from pyproject.toml."""
        metadata = extract_project_metadata(project_with_pyproject)

        assert metadata["name"] == "my-project"
        assert metadata["description"] == "My awesome project"
        assert metadata["domain"] == "software"
        assert metadata["components"] == []
        assert metadata["entry_point"] == "main.py"

    def test_from_structure_inference(self, valid_project):
        """Extract metadata from structure when no config files exist."""
        metadata = extract_project_metadata(valid_project)

        assert metadata["name"] == "test_project"
        assert metadata["description"] == ""
        assert metadata["domain"] == "software"
        assert metadata["components"] == []
        assert metadata["entry_point"] == "main.py"

    def test_invalid_blueprint_json(self, tmp_path):
        """Fall back to structure inference if blueprint.json is invalid."""
        project = tmp_path / "bad_bp"
        project.mkdir()
        (project / "main.py").write_text("print('main')")
        tests = project / "tests"
        tests.mkdir()
        (project / "blueprint.json").write_text("invalid json{")

        metadata = extract_project_metadata(project)

        assert metadata["name"] == "bad_bp"
        assert metadata["domain"] == "software"


class TestExtractCodeFiles:
    """Test extract_code_files function."""

    def test_extract_single_file(self, valid_project):
        """Extract single Python file."""
        code_files = extract_code_files(valid_project)

        assert "main.py" in code_files
        assert code_files["main.py"] == "print('hello')"

    def test_extract_nested_files(self, tmp_path):
        """Extract nested Python files."""
        project = tmp_path / "nested"
        project.mkdir()

        (project / "main.py").write_text("# main")

        subdir = project / "lib"
        subdir.mkdir()
        (subdir / "utils.py").write_text("# utils")

        tests = project / "tests"
        tests.mkdir()
        (tests / "test_main.py").write_text("# test")

        code_files = extract_code_files(project)

        assert "main.py" in code_files
        assert "lib/utils.py" in code_files
        assert "tests/test_main.py" in code_files
        assert len(code_files) == 3

    def test_skip_pycache(self, tmp_path):
        """Skip __pycache__ directories."""
        project = tmp_path / "pycache"
        project.mkdir()

        (project / "main.py").write_text("# main")

        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_text("bytecode")

        tests = project / "tests"
        tests.mkdir()

        code_files = extract_code_files(project)

        assert "main.py" in code_files
        assert not any("__pycache__" in path for path in code_files.keys())

    def test_skip_venv(self, tmp_path):
        """Skip .venv directories."""
        project = tmp_path / "venv"
        project.mkdir()

        (project / "main.py").write_text("# main")

        venv = project / ".venv"
        venv.mkdir()
        lib = venv / "lib"
        lib.mkdir()
        (lib / "something.py").write_text("# lib")

        tests = project / "tests"
        tests.mkdir()

        code_files = extract_code_files(project)

        assert "main.py" in code_files
        assert not any(".venv" in path for path in code_files.keys())

    def test_skip_git(self, tmp_path):
        """Skip .git directories."""
        project = tmp_path / "git"
        project.mkdir()

        (project / "main.py").write_text("# main")

        git = project / ".git"
        git.mkdir()
        (git / "config").write_text("git config")

        tests = project / "tests"
        tests.mkdir()

        code_files = extract_code_files(project)

        assert "main.py" in code_files
        assert not any(".git" in path for path in code_files.keys())

    def test_handle_unreadable_files(self, tmp_path):
        """Skip files that can't be read."""
        project = tmp_path / "unreadable"
        project.mkdir()

        (project / "main.py").write_text("# main")

        # Create file with invalid UTF-8 (binary)
        bad_file = project / "bad.py"
        bad_file.write_bytes(b"\x80\x81\x82")

        tests = project / "tests"
        tests.mkdir()

        code_files = extract_code_files(project)

        # Should have main.py but not bad.py
        assert "main.py" in code_files
        assert "bad.py" not in code_files
