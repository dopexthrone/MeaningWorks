"""Tests for mother/appendage_builder.py — scaffold + prompt + validation."""

import os

import pytest

from mother.appendage_builder import (
    BuildSpec,
    scaffold_project_dir,
    generate_build_prompt,
    validate_agent_script,
)


class TestBuildSpec:
    def test_frozen(self):
        spec = BuildSpec(name="test", description="d", capability_gap="g")
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_defaults(self):
        spec = BuildSpec(name="x", description="d", capability_gap="g")
        assert spec.capabilities == ()
        assert spec.constraints == ""


class TestScaffoldProjectDir:
    def test_creates_directory_and_main(self, tmp_path):
        project_dir = scaffold_project_dir(str(tmp_path), "file-counter")
        assert os.path.isdir(project_dir)
        main_path = os.path.join(project_dir, "main.py")
        assert os.path.isfile(main_path)

        content = open(main_path).read()
        assert "handle_request" in content
        assert '"ready"' in content
        assert "sys.stdin" in content
        assert "json.dumps" in content

    def test_does_not_overwrite_existing(self, tmp_path):
        project_dir = scaffold_project_dir(str(tmp_path), "test-agent")
        main_path = os.path.join(project_dir, "main.py")

        # Write custom content
        with open(main_path, "w") as f:
            f.write("# custom implementation\n")

        # Scaffold again — should NOT overwrite
        scaffold_project_dir(str(tmp_path), "test-agent")
        content = open(main_path).read()
        assert content == "# custom implementation\n"

    def test_nested_directory(self, tmp_path):
        base = str(tmp_path / "deep" / "nested")
        project_dir = scaffold_project_dir(base, "agent")
        assert os.path.isdir(project_dir)


class TestGenerateBuildPrompt:
    def test_includes_spec_fields(self):
        spec = BuildSpec(
            name="screen-recorder",
            description="Records the user's screen",
            capability_gap="screen recording capability",
            capabilities=["screen", "record", "capture"],
        )
        prompt = generate_build_prompt(spec)
        assert "screen-recorder" in prompt
        assert "Records the user's screen" in prompt
        assert "screen recording capability" in prompt
        assert '"screen"' in prompt
        assert '"record"' in prompt
        assert '"capture"' in prompt

    def test_includes_constraints(self):
        spec = BuildSpec(
            name="test",
            description="d",
            capability_gap="g",
            constraints="Must use only macOS APIs",
        )
        prompt = generate_build_prompt(spec)
        assert "Must use only macOS APIs" in prompt

    def test_includes_protocol_reminder(self):
        spec = BuildSpec(name="test", description="d", capability_gap="g")
        prompt = generate_build_prompt(spec)
        assert "PROTOCOL REMINDER" in prompt
        assert "ready" in prompt
        assert "req_001" in prompt


class TestValidateAgentScript:
    def test_valid_scaffold(self, tmp_path):
        project_dir = scaffold_project_dir(str(tmp_path), "valid-agent")
        valid, msg = validate_agent_script(project_dir)
        assert valid is True
        assert msg == "Valid"

    def test_file_not_found(self, tmp_path):
        valid, msg = validate_agent_script(str(tmp_path))
        assert valid is False
        assert "not found" in msg

    def test_syntax_error(self, tmp_path):
        agent_dir = tmp_path / "bad-agent"
        agent_dir.mkdir()
        (agent_dir / "main.py").write_text("def broken(\n")

        valid, msg = validate_agent_script(str(agent_dir))
        assert valid is False
        assert "Syntax error" in msg

    def test_missing_json_import(self, tmp_path):
        agent_dir = tmp_path / "no-json"
        agent_dir.mkdir()
        (agent_dir / "main.py").write_text(
            'import sys\nfor line in sys.stdin:\n    print("ready")\n'
        )

        valid, msg = validate_agent_script(str(agent_dir))
        assert valid is False
        assert "import json" in msg

    def test_missing_ready_signal(self, tmp_path):
        agent_dir = tmp_path / "no-ready"
        agent_dir.mkdir()
        (agent_dir / "main.py").write_text(
            "import json\nimport sys\nfor line in sys.stdin:\n    pass\n"
        )

        valid, msg = validate_agent_script(str(agent_dir))
        assert valid is False
        assert "ready signal" in msg

    def test_missing_stdin(self, tmp_path):
        agent_dir = tmp_path / "no-stdin"
        agent_dir.mkdir()
        (agent_dir / "main.py").write_text(
            'import json\nprint("ready")\n'
        )

        valid, msg = validate_agent_script(str(agent_dir))
        assert valid is False
        assert "stdin" in msg
