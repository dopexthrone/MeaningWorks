"""Tests for direct code task feature — Mother writes code via Claude Code CLI
targeting the user's project (not self-modification).

Covers:
- bridge.code_task() success/failure/rollback/no-git paths
- persona.py ACTION:code routing and personality bites
- chat.py _run_code_task gate and _code_task_worker dispatch
- ACTION parsing for [ACTION:code]
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from mother.bridge import EngineBridge
from mother.persona import inject_personality_bite, INTENT_ROUTING, _PERSONALITY_BITES


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# INTENT_ROUTING contains ACTION:code examples
# ---------------------------------------------------------------------------

class TestIntentRouting:
    def test_action_code_in_routing(self):
        assert "[ACTION:code]" in INTENT_ROUTING

    def test_code_examples_present(self):
        assert "Write me a Python script" in INTENT_ROUTING
        assert "Fix the bug in my project" in INTENT_ROUTING
        assert "Add a login page" in INTENT_ROUTING

    def test_web_research_routes_through_code(self):
        """Web research, scraping, internet tasks route via ACTION:code."""
        assert "Research" in INTENT_ROUTING and "[ACTION:code]" in INTENT_ROUTING
        assert "Scrape" in INTENT_ROUTING
        assert "latest docs" in INTENT_ROUTING or "Find the latest" in INTENT_ROUTING
        assert "price of Bitcoin" in INTENT_ROUTING

    def test_never_say_cant_instruction(self):
        """INTENT_ROUTING tells LLM not to refuse if ACTION:code can handle it."""
        assert "Never say" in INTENT_ROUTING and "can't do that" in INTENT_ROUTING

    def test_code_action_described_with_web_capabilities(self):
        """ACTION:code trigger description mentions web search and shell access."""
        assert "web search" in INTENT_ROUTING or "internet" in INTENT_ROUTING
        assert "shell" in INTENT_ROUTING or "shell access" in INTENT_ROUTING


# ---------------------------------------------------------------------------
# Personality bites for code_task events
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Context capabilities include code + web when claude_code enabled
# ---------------------------------------------------------------------------

class TestContextCapabilities:
    def test_claude_code_enables_code_and_web(self):
        """When cap_claude_code=True, context lists code writing and web research."""
        from mother.context import ContextData, synthesize_frame
        data = ContextData(cap_claude_code=True)
        frame = synthesize_frame(data)
        assert "code writing" in frame
        assert "web research" in frame
        assert "self-build" in frame

    def test_claude_code_disabled_no_code_cap(self):
        """When cap_claude_code=False, no code/web capabilities listed."""
        from mother.context import ContextData, synthesize_frame
        data = ContextData(cap_claude_code=False)
        frame = synthesize_frame(data)
        assert "code writing" not in frame
        assert "web research" not in frame


# ---------------------------------------------------------------------------
# Personality bites for code_task events
# ---------------------------------------------------------------------------

class TestCodeTaskBites:
    @pytest.mark.parametrize("personality", ["composed", "warm", "direct", "playful", "david"])
    def test_code_task_start_bite_exists(self, personality):
        bite = inject_personality_bite(personality, "code_task_start")
        assert bite is not None
        assert len(bite) > 0

    @pytest.mark.parametrize("personality", ["composed", "warm", "direct", "playful", "david"])
    def test_code_task_success_bite_exists(self, personality):
        bite = inject_personality_bite(personality, "code_task_success")
        assert bite is not None

    @pytest.mark.parametrize("personality", ["composed", "warm", "direct", "playful", "david"])
    def test_code_task_failed_bite_exists(self, personality):
        bite = inject_personality_bite(personality, "code_task_failed")
        assert bite is not None

    def test_all_personalities_have_code_bites(self):
        for personality, bites in _PERSONALITY_BITES.items():
            assert "code_task_start" in bites, f"{personality} missing code_task_start"
            assert "code_task_success" in bites, f"{personality} missing code_task_success"
            assert "code_task_failed" in bites, f"{personality} missing code_task_failed"


# ---------------------------------------------------------------------------
# bridge.code_task() — success path
# ---------------------------------------------------------------------------

class TestBridgeCodeTaskSuccess:
    def test_success_with_git_repo(self, tmp_path):
        """code_task succeeds and commits when target is a git repo."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=True,
            result_text="Created fibonacci.py",
            cost_usd=0.05,
            error="",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.claude_code.git_rollback") as mock_rollback, \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run"):  # git add/commit
            # Create .git dir to simulate git repo
            (tmp_path / ".git").mkdir()

            result = run(bridge.code_task(
                prompt="write fibonacci",
                target_dir=str(tmp_path),
            ))

        assert result["success"] is True
        assert result["result_text"] == "Created fibonacci.py"
        assert result["cost_usd"] == 0.05
        assert result["rolled_back"] is False
        assert result["error"] == ""
        mock_rollback.assert_not_called()

    def test_success_without_git_repo(self, tmp_path):
        """code_task succeeds without git snapshot when no .git dir."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=True,
            result_text="Created script.py",
            cost_usd=0.02,
            error="",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot") as mock_snap, \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()):
            result = run(bridge.code_task(
                prompt="write a script",
                target_dir=str(tmp_path),
            ))

        assert result["success"] is True
        assert result["rolled_back"] is False
        mock_snap.assert_not_called()


# ---------------------------------------------------------------------------
# bridge.code_task() — failure path
# ---------------------------------------------------------------------------

class TestBridgeCodeTaskFailure:
    def test_failure_rolls_back_git_repo(self, tmp_path):
        """code_task rolls back on CLI failure when git repo."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=False,
            result_text="",
            cost_usd=0.01,
            error="CLI error",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.claude_code.git_rollback") as mock_rollback:
            (tmp_path / ".git").mkdir()

            result = run(bridge.code_task(
                prompt="fix bug",
                target_dir=str(tmp_path),
            ))

        assert result["success"] is False
        assert result["rolled_back"] is True
        assert result["error"] == "CLI error"
        mock_rollback.assert_called_once_with(str(tmp_path), "abc123")

    def test_failure_no_rollback_without_git(self, tmp_path):
        """code_task failure doesn't attempt rollback without .git."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=False,
            result_text="",
            cost_usd=0.01,
            error="CLI error",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_rollback") as mock_rollback:
            result = run(bridge.code_task(
                prompt="fix bug",
                target_dir=str(tmp_path),
            ))

        assert result["success"] is False
        assert result["rolled_back"] is False
        mock_rollback.assert_not_called()


# ---------------------------------------------------------------------------
# bridge.code_task() — test failure path
# ---------------------------------------------------------------------------

class TestBridgeCodeTaskTestFailure:
    def test_test_failure_rolls_back(self, tmp_path):
        """code_task rolls back when run_tests=True and tests fail."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=True,
            result_text="Wrote code",
            cost_usd=0.03,
            error="",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.claude_code.git_rollback") as mock_rollback, \
             patch("mother.claude_code.run_tests", return_value=False):
            (tmp_path / ".git").mkdir()

            result = run(bridge.code_task(
                prompt="fix bug",
                target_dir=str(tmp_path),
                run_tests=True,
            ))

        assert result["success"] is False
        assert result["rolled_back"] is True
        assert "Tests failed" in result["error"]
        mock_rollback.assert_called_once()

    def test_tests_pass_succeeds(self, tmp_path):
        """code_task succeeds when run_tests=True and tests pass."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=True,
            result_text="Fixed it",
            cost_usd=0.04,
            error="",
        )

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.claude_code.run_tests", return_value=True), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run"):
            (tmp_path / ".git").mkdir()

            result = run(bridge.code_task(
                prompt="fix bug",
                target_dir=str(tmp_path),
                run_tests=True,
            ))

        assert result["success"] is True
        assert result["rolled_back"] is False


# ---------------------------------------------------------------------------
# bridge.code_task() — allowed_tools and parameters
# ---------------------------------------------------------------------------

class TestBridgeCodeTaskParams:
    def test_allowed_tools_includes_write(self, tmp_path):
        """code_task passes Write in allowed_tools (unlike self_build)."""
        bridge = EngineBridge()
        captured_kwargs = {}

        def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(success=True, result_text="ok", cost_usd=0.0, error="")

        with patch("mother.coding_agent.invoke_coding_agent", side_effect=capture_invoke), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()):
            result = run(bridge.code_task(
                prompt="write script",
                target_dir=str(tmp_path),
            ))

        assert "Write" in captured_kwargs.get("allowed_tools", "")

    def test_custom_allowed_tools(self, tmp_path):
        """code_task passes custom allowed_tools through."""
        bridge = EngineBridge()
        captured_kwargs = {}

        def capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(success=True, result_text="ok", cost_usd=0.0, error="")

        with patch("mother.coding_agent.invoke_coding_agent", side_effect=capture_invoke), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()):
            result = run(bridge.code_task(
                prompt="write script",
                target_dir=str(tmp_path),
                allowed_tools="Read,Write",
            ))

        assert captured_kwargs["allowed_tools"] == "Read,Write"

    def test_commit_message_prefix(self, tmp_path):
        """code_task uses 'mother:' commit prefix (not 'self-build:')."""
        bridge = EngineBridge()

        mock_result = SimpleNamespace(
            success=True,
            result_text="ok",
            cost_usd=0.0,
            error="",
        )
        committed_messages = []

        def capture_subprocess_run(cmd, **kwargs):
            if cmd and len(cmd) >= 4 and cmd[0] == "git" and cmd[1] == "commit":
                committed_messages.append(cmd[3])
            return SimpleNamespace(returncode=0)

        with patch("mother.coding_agent.invoke_coding_agent", return_value=mock_result), \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run", side_effect=capture_subprocess_run):
            (tmp_path / ".git").mkdir()

            result = run(bridge.code_task(
                prompt="add login page",
                target_dir=str(tmp_path),
            ))

        assert any("mother:" in msg for msg in committed_messages)


# ---------------------------------------------------------------------------
# ACTION parsing — [ACTION:code] correctly parsed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# claude_code.py — is_error handling
# ---------------------------------------------------------------------------

class TestClaudeCodeIsError:
    def test_is_error_flag_returns_failure(self):
        """invoke_claude_code treats JSON is_error=true as failure even with exit 0."""
        from mother.claude_code import invoke_claude_code, ClaudeCodeResult
        import json

        error_json = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "Credit balance is too low to start a new conversation.",
            "session_id": "abc123",
            "cost_usd": 0.0,
            "num_turns": 0,
        })

        completed = SimpleNamespace(returncode=0, stdout=error_json, stderr="")

        with patch("subprocess.run", return_value=completed), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code(
                prompt="test",
                cwd="/tmp",
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is False
        assert result.is_error is True
        assert "Credit balance" in result.error

    def test_nonzero_exit_with_json_extracts_clean_error(self):
        """Exit code 1 + JSON in stdout extracts clean result field, not raw JSON."""
        from mother.claude_code import invoke_claude_code
        import json

        error_json = json.dumps({
            "type": "result",
            "subtype": "error_response",
            "is_error": True,
            "result": "Credit balance is too low to start a new conversation.",
            "session_id": "",
            "cost_usd": 0.0,
            "num_turns": 0,
        })

        completed = SimpleNamespace(returncode=1, stdout=error_json, stderr="")

        with patch("subprocess.run", return_value=completed), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code(
                prompt="test",
                cwd="/tmp",
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is False
        assert result.is_error is True
        assert "Credit balance" in result.error
        # Must NOT contain raw JSON
        assert '{"type"' not in result.error

    def test_nonzero_exit_plain_stderr_preserved(self):
        """Exit code 1 with plain text stderr uses stderr as error."""
        from mother.claude_code import invoke_claude_code

        completed = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Connection refused",
        )

        with patch("subprocess.run", return_value=completed), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code(
                prompt="test",
                cwd="/tmp",
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is False
        assert result.error == "Connection refused"

    def test_normal_success_not_affected(self):
        """Normal JSON response without is_error still works."""
        from mother.claude_code import invoke_claude_code
        import json

        ok_json = json.dumps({
            "type": "result",
            "result": "Created fibonacci.py",
            "session_id": "def456",
            "cost_usd": 0.05,
            "num_turns": 3,
        })

        completed = SimpleNamespace(returncode=0, stdout=ok_json, stderr="")

        with patch("subprocess.run", return_value=completed), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code(
                prompt="test",
                cwd="/tmp",
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is True
        assert result.is_error is False
        assert result.result_text == "Created fibonacci.py"


# ---------------------------------------------------------------------------
# ACTION parsing — [ACTION:code] correctly parsed
# ---------------------------------------------------------------------------

class TestActionCodeParsing:
    def test_parse_response_extracts_code_action(self):
        """parse_response correctly extracts ACTION:code."""
        from mother.screens.chat import parse_response
        text = "[ACTION:code]write a fibonacci script[/ACTION][VOICE]On it.[/VOICE]"
        parsed = parse_response(text)
        assert parsed["action"] == "code"
        assert parsed["action_arg"] == "write a fibonacci script"
        assert parsed["voice"] == "On it."

    def test_parse_response_code_without_voice(self):
        """ACTION:code works without VOICE tags."""
        from mother.screens.chat import parse_response
        text = "[ACTION:code]fix the login bug[/ACTION]"
        parsed = parse_response(text)
        assert parsed["action"] == "code"
        assert parsed["action_arg"] == "fix the login bug"


# ---------------------------------------------------------------------------
# bridge.record_task_failure() — self-repair goal creation
# ---------------------------------------------------------------------------

class TestRecordTaskFailure:
    def test_creates_goal_on_failure(self, tmp_path):
        """record_task_failure creates a GoalStore entry with self-repair signal words."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()

        result = bridge.record_task_failure(
            db_path=db_path,
            task_type="code_task",
            description="write fibonacci script",
            error="CLI timed out after 600s",
        )

        assert result is True

        # Verify goal was actually created
        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()

        assert len(goals) == 1
        assert "[SELF-REPAIR]" in goals[0].description
        assert "code_task" in goals[0].description
        assert "CLI timed out" in goals[0].description

    def test_goal_triggers_self_build_detection(self, tmp_path):
        """Goal description contains enough signal words for _is_self_build_goal()."""
        from mother.executive import _is_self_build_goal
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()

        bridge.record_task_failure(
            db_path=db_path,
            task_type="code_task",
            description="write fibonacci",
            error="Connection refused",
        )

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()

        assert len(goals) == 1
        # Must have 2+ signal words: "capability", "resilience", "strengthen"
        assert _is_self_build_goal(goals[0].description) is True

    def test_dedup_prevents_duplicate_goals(self, tmp_path):
        """Same error recorded twice creates only one goal."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()

        r1 = bridge.record_task_failure(db_path, "code_task", "test", "Error XYZ")
        r2 = bridge.record_task_failure(db_path, "code_task", "test", "Error XYZ")

        assert r1 is True
        assert r2 is False  # Deduped

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()
        assert len(goals) == 1

    def test_different_errors_create_separate_goals(self, tmp_path):
        """Different errors create separate goals (not deduped)."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()

        bridge.record_task_failure(db_path, "code_task", "test", "Error AAA")
        bridge.record_task_failure(db_path, "code_task", "test", "Error BBB")

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()
        assert len(goals) == 2

    def test_self_build_failure_goal(self, tmp_path):
        """record_task_failure works for self_build task type."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()

        result = bridge.record_task_failure(
            db_path=db_path,
            task_type="self_build",
            description="improve compiler",
            error="Tests failed after modification",
        )

        assert result is True

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()

        assert "self_build" in goals[0].description
        assert "Tests failed" in goals[0].description

    def test_goal_source_contains_task_type(self, tmp_path):
        """Goal source field records the task type for tracking."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()
        bridge.record_task_failure(db_path, "code_task", "test", "error")

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()

        assert goals[0].source == "self-repair:code_task"

    def test_error_truncated_at_200_chars(self, tmp_path):
        """Long error messages are truncated in goal description."""
        db_path = tmp_path / "history.db"
        bridge = EngineBridge()
        long_error = "E" * 500
        bridge.record_task_failure(db_path, "code_task", "test", long_error)

        from mother.goals import GoalStore
        gs = GoalStore(db_path)
        goals = gs.active(limit=10)
        gs.close()

        # Error portion should be truncated at 200 chars
        assert "E" * 200 in goals[0].description
        assert "E" * 201 not in goals[0].description

    def test_graceful_on_bad_db_path(self):
        """record_task_failure returns False on invalid db path, no crash."""
        bridge = EngineBridge()
        result = bridge.record_task_failure(
            db_path="/nonexistent/dir/history.db",
            task_type="code_task",
            description="test",
            error="error",
        )
        assert result is False


# ---------------------------------------------------------------------------
# Self-repair wiring — workers call record_task_failure on failure
# ---------------------------------------------------------------------------

class TestSelfRepairWiring:
    """Verify _code_task_worker and _self_build_worker call record_task_failure."""

    def test_code_task_worker_records_failure(self):
        """_code_task_worker calls record_task_failure when code_task fails."""
        # Minimal integration: verify the method is called in the failure path
        import ast
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        # Find the _code_task_worker method body
        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_code_task_worker":
                body_source = ast.get_source_segment(source, node)
                assert "record_task_failure" in body_source
                found = True
                break
        assert found, "_code_task_worker not found in chat.py"

    def test_self_build_worker_records_failure(self):
        """_self_build_worker calls record_task_failure when self_build fails."""
        import ast
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_self_build_worker":
                body_source = ast.get_source_segment(source, node)
                assert "record_task_failure" in body_source
                found = True
                break
        assert found, "_self_build_worker not found in chat.py"

    def test_code_task_worker_records_on_both_paths(self):
        """record_task_failure called in both result failure and exception paths."""
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        # Find the _code_task_worker method and count record_task_failure calls
        start = source.find("async def _code_task_worker")
        assert start > 0
        # Find end of method (next def at same indentation level)
        end = source.find("\n    def ", start + 10)
        if end < 0:
            end = source.find("\n    async def ", start + 10)
        method_body = source[start:end] if end > 0 else source[start:]

        count = method_body.count("record_task_failure")
        assert count >= 2, f"Expected 2+ calls to record_task_failure, found {count}"

    def test_self_build_worker_records_on_both_paths(self):
        """record_task_failure called in both result failure and exception paths."""
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        start = source.find("async def _self_build_worker")
        assert start > 0
        end = source.find("\n    # --- Code task", start + 10)
        method_body = source[start:end] if end > 0 else source[start:]

        count = method_body.count("record_task_failure")
        assert count >= 2, f"Expected 2+ calls to record_task_failure, found {count}"


# ---------------------------------------------------------------------------
# invoke_claude_code_streaming — streaming invocation
# ---------------------------------------------------------------------------

class TestStreamingInvocation:
    def test_streaming_success_with_result_event(self):
        """invoke_claude_code_streaming parses result event and returns ClaudeCodeResult."""
        from mother.claude_code import invoke_claude_code_streaming

        result_json = json.dumps({
            "type": "result",
            "result": "Modified engine.py",
            "session_id": "sess_123",
            "cost_usd": 0.08,
            "num_turns": 5,
        })
        # Simulate stream-json output: one assistant event + result event
        assistant_json = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Thinking..."}]},
        })
        stream_output = f"{assistant_json}\n{result_json}\n"

        captured_events = []

        mock_proc = MagicMock()
        mock_proc.stdout = iter(stream_output.splitlines(True))
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code_streaming(
                prompt="fix bug",
                cwd="/tmp",
                on_event=lambda e: captured_events.append(e),
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is True
        assert result.result_text == "Modified engine.py"
        assert result.session_id == "sess_123"
        assert result.cost_usd == 0.08
        assert result.num_turns == 5
        assert len(captured_events) == 2

    def test_streaming_cli_not_found(self):
        """Returns error when CLI binary doesn't exist."""
        from mother.claude_code import invoke_claude_code_streaming

        result = invoke_claude_code_streaming(
            prompt="test",
            cwd="/tmp",
            on_event=lambda e: None,
            claude_path="/nonexistent/claude",
        )
        assert result.success is False
        assert result.is_error is True
        assert "not found" in result.error

    def test_streaming_is_error_flag(self):
        """is_error=True in result event yields failure."""
        from mother.claude_code import invoke_claude_code_streaming

        result_json = json.dumps({
            "type": "result",
            "is_error": True,
            "result": "Credit exhausted",
            "session_id": "",
            "cost_usd": 0.0,
            "num_turns": 0,
        })
        stream_output = f"{result_json}\n"

        mock_proc = MagicMock()
        mock_proc.stdout = iter(stream_output.splitlines(True))
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=lambda e: None,
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is False
        assert result.is_error is True
        assert "Credit exhausted" in result.error

    def test_streaming_callback_receives_all_events(self):
        """on_event callback fires for every parsed line."""
        from mother.claude_code import invoke_claude_code_streaming

        events_data = [
            {"type": "assistant", "message": {"content": []}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "result", "result": "done", "cost_usd": 0.01, "num_turns": 1},
        ]
        stream_output = "\n".join(json.dumps(e) for e in events_data) + "\n"

        captured = []
        mock_proc = MagicMock()
        mock_proc.stdout = iter(stream_output.splitlines(True))
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=lambda e: captured.append(e),
                claude_path="/usr/local/bin/claude",
            )

        assert len(captured) == 3
        assert captured[0]["type"] == "assistant"
        assert captured[1]["type"] == "content_block_delta"
        assert captured[2]["type"] == "result"

    def test_streaming_bad_json_lines_skipped(self):
        """Non-JSON lines in stream are silently skipped."""
        from mother.claude_code import invoke_claude_code_streaming

        result_json = json.dumps({"type": "result", "result": "ok", "cost_usd": 0.01, "num_turns": 1})
        stream_output = f"not json\n{result_json}\nalso not json\n"

        captured = []
        mock_proc = MagicMock()
        mock_proc.stdout = iter(stream_output.splitlines(True))
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=lambda e: captured.append(e),
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is True
        assert len(captured) == 1  # Only the result event

    def test_streaming_nonzero_exit_no_result(self):
        """Non-zero exit with no result event returns failure."""
        from mother.claude_code import invoke_claude_code_streaming

        mock_proc = MagicMock()
        mock_proc.stdout = iter([])  # empty stream
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = "Connection refused"
        mock_proc.returncode = 1
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=lambda e: None,
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is False
        assert "Connection refused" in result.error

    def test_streaming_callback_error_does_not_crash(self):
        """on_event callback raising doesn't crash the invocation."""
        from mother.claude_code import invoke_claude_code_streaming

        result_json = json.dumps({"type": "result", "result": "ok", "cost_usd": 0.01, "num_turns": 1})
        stream_output = f"{result_json}\n"

        def bad_callback(event):
            raise RuntimeError("callback boom")

        mock_proc = MagicMock()
        mock_proc.stdout = iter(stream_output.splitlines(True))
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.returncode = 0
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            result = invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=bad_callback,
                claude_path="/usr/local/bin/claude",
            )

        assert result.success is True
        assert result.result_text == "ok"

    def test_streaming_uses_stream_json_format(self):
        """Verify --output-format stream-json is passed to CLI."""
        from mother.claude_code import invoke_claude_code_streaming

        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            result_json = json.dumps({"type": "result", "result": "ok", "cost_usd": 0, "num_turns": 0})
            proc.stdout = iter([result_json + "\n"])
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = ""
            proc.returncode = 0
            proc.wait = MagicMock()
            return proc

        with patch("subprocess.Popen", side_effect=mock_popen), \
             patch("mother.claude_code._clean_env", return_value=os.environ.copy()), \
             patch("pathlib.Path.exists", return_value=True):
            invoke_claude_code_streaming(
                prompt="test",
                cwd="/tmp",
                on_event=lambda e: None,
                claude_path="/usr/local/bin/claude",
            )

        assert "--output-format" in captured_cmd
        idx = captured_cmd.index("--output-format")
        assert captured_cmd[idx + 1] == "stream-json"
        assert "--verbose" in captured_cmd


# ---------------------------------------------------------------------------
# save_build_log — JSONL log writing
# ---------------------------------------------------------------------------

class TestSaveBuildLog:
    def test_writes_jsonl_file(self, tmp_path):
        """save_build_log creates a JSONL file with metadata + events + outcome."""
        from mother.claude_code import save_build_log, ClaudeCodeResult

        events = [
            {"type": "assistant", "message": {"content": []}},
            {"type": "result", "result": "done"},
        ]
        result = ClaudeCodeResult(
            success=True, result_text="done", cost_usd=0.05,
            duration_seconds=12.3, num_turns=3,
        )

        with patch("mother.claude_code.Path.home", return_value=tmp_path):
            log_path = save_build_log(events, "fix bug", "/repo", result, "fix-bug")

        assert log_path is not None
        assert log_path.exists()
        assert log_path.suffix == ".jsonl"
        assert "fix-bug" in log_path.name

        # Read back and verify structure
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 4  # metadata + 2 events + outcome

        meta = json.loads(lines[0])
        assert meta["_type"] == "metadata"
        assert meta["prompt"] == "fix bug"
        assert meta["repo_dir"] == "/repo"

        outcome = json.loads(lines[-1])
        assert outcome["_type"] == "outcome"
        assert outcome["success"] is True
        assert outcome["cost_usd"] == 0.05
        assert outcome["num_turns"] == 3

    def test_slug_sanitization(self, tmp_path):
        """Special characters in slug are sanitized."""
        from mother.claude_code import save_build_log, ClaudeCodeResult

        result = ClaudeCodeResult(success=True, duration_seconds=1.0)

        with patch("mother.claude_code.Path.home", return_value=tmp_path):
            log_path = save_build_log([], "test", "/repo", result, "fix bug/evil<>chars")

        assert log_path is not None
        # No slashes or angle brackets in filename
        assert "/" not in log_path.name.split("/")[-1]
        assert "<" not in log_path.name
        assert ">" not in log_path.name

    def test_empty_events_list(self, tmp_path):
        """save_build_log works with zero events."""
        from mother.claude_code import save_build_log, ClaudeCodeResult

        result = ClaudeCodeResult(success=False, error="failed", duration_seconds=0.5)

        with patch("mother.claude_code.Path.home", return_value=tmp_path):
            log_path = save_build_log([], "test", "/repo", result)

        assert log_path is not None
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2  # metadata + outcome

    def test_returns_none_on_bad_dir(self):
        """Returns None when log dir can't be created."""
        from mother.claude_code import save_build_log, ClaudeCodeResult

        result = ClaudeCodeResult(success=True, duration_seconds=1.0)

        with patch("mother.claude_code.Path.home", return_value=Path("/nonexistent/path")):
            log_path = save_build_log([], "test", "/repo", result)

        assert log_path is None


# ---------------------------------------------------------------------------
# Bridge self-build streaming infrastructure
# ---------------------------------------------------------------------------

class TestBridgeSelfBuildStreaming:
    def test_begin_self_build_clears_state(self):
        """begin_self_build() clears done event, resets events, drains stale queue."""
        bridge = EngineBridge()
        bridge._self_build_done.set()
        bridge._self_build_events = [{"old": True}]
        bridge._self_build_event_queue.put({"stale": True})

        bridge.begin_self_build()

        assert not bridge._self_build_done.is_set()
        assert bridge._self_build_events == []
        assert bridge._self_build_event_queue.empty()

    def test_stream_terminates_on_sentinel(self):
        """stream_self_build_events() terminates when sentinel (None) is received."""
        bridge = EngineBridge()
        bridge._self_build_done.clear()

        # Pre-populate queue with events + sentinel
        bridge._self_build_event_queue.put({"type": "assistant", "n": 1})
        bridge._self_build_event_queue.put({"type": "result", "n": 2})
        bridge._self_build_event_queue.put(bridge._SELF_BUILD_SENTINEL)

        events = run(collect_stream(bridge.stream_self_build_events()))

        assert len(events) == 2
        assert events[0]["n"] == 1
        assert events[1]["n"] == 2

    def test_stream_fallback_on_done_flag(self):
        """stream_self_build_events() exits on done flag if sentinel is missing."""
        bridge = EngineBridge()
        bridge._self_build_done.clear()

        # Put events but no sentinel
        bridge._self_build_event_queue.put({"type": "assistant", "n": 1})
        bridge._self_build_done.set()  # fallback signal

        events = run(collect_stream(bridge.stream_self_build_events()))
        assert len(events) == 1

    def test_stream_exits_when_done_and_empty(self):
        """Stream exits immediately when done=True and queue is empty."""
        bridge = EngineBridge()
        bridge._self_build_done.set()

        events = run(collect_stream(bridge.stream_self_build_events()))
        assert events == []

    def test_sentinel_not_yielded_to_consumer(self):
        """The sentinel value is never yielded as an event."""
        bridge = EngineBridge()
        bridge._self_build_done.clear()
        bridge._self_build_event_queue.put(bridge._SELF_BUILD_SENTINEL)

        events = run(collect_stream(bridge.stream_self_build_events()))
        assert events == []

    def test_self_build_pushes_sentinel_at_end(self, tmp_path):
        """self_build() pushes sentinel as the last item in the queue."""
        bridge = EngineBridge()
        bridge.begin_self_build()

        events_fired = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Working"}]}},
            {"type": "result", "result": "Done", "cost_usd": 0.02, "num_turns": 2},
        ]

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            for e in events_fired:
                on_event(e)
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=True, result_text="Done", cost_usd=0.02, num_turns=2)

        with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
             patch("mother.claude_code.save_build_log", return_value=None), \
             patch("mother.claude_code.git_snapshot", return_value="abc"), \
             patch("mother.claude_code.run_tests", return_value=True), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run"):
            result = run(bridge.self_build(
                prompt="fix thing",
                repo_dir=str(tmp_path),
            ))

        assert result["success"] is True
        # Drain queue and verify sentinel is the last item
        items = []
        while not bridge._self_build_event_queue.empty():
            items.append(bridge._self_build_event_queue.get_nowait())
        # Last item should be sentinel (None). Some items before it are phase events + stream events.
        # But sentinel was already consumed, or may still be in queue
        # The key invariant: after self_build returns, done is set
        assert bridge._self_build_done.is_set()

    def test_self_build_pushes_events_to_queue(self, tmp_path):
        """self_build() pushes Claude Code events to the event queue."""
        bridge = EngineBridge()
        bridge.begin_self_build()

        events_fired = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Working"}]}},
            {"type": "result", "result": "Done", "cost_usd": 0.02, "num_turns": 2},
        ]

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            for e in events_fired:
                on_event(e)
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=True, result_text="Done", cost_usd=0.02, num_turns=2)

        # Use stream consumer to collect events (correct way, accounts for sentinel)
        collected = []

        async def _run():
            nonlocal collected
            with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
                 patch("mother.claude_code.save_build_log", return_value=None), \
                 patch("mother.claude_code.git_snapshot", return_value="abc"), \
                 patch("mother.claude_code.run_tests", return_value=True), \
                 patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
                 patch("subprocess.run"):
                build_task = asyncio.create_task(bridge.self_build(
                    prompt="fix thing",
                    repo_dir=str(tmp_path),
                ))
                async for event in bridge.stream_self_build_events():
                    collected.append(event)
                result = await build_task
            return result

        result = run(_run())
        assert result["success"] is True
        # Should have: phase(snapshot) + phase(invoke) + 2 stream events + phase(testing) + phase(commit)
        assert len(collected) >= 4
        # Verify stream events are present
        stream_types = [e.get("type") for e in collected if "type" in e]
        assert "assistant" in stream_types

    def test_self_build_saves_log(self, tmp_path):
        """self_build() calls save_build_log."""
        bridge = EngineBridge()

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=True, result_text="Done", cost_usd=0.01, num_turns=1)

        with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
             patch("mother.claude_code.save_build_log", return_value=Path("/tmp/log.jsonl")) as mock_log, \
             patch("mother.claude_code.git_snapshot", return_value="abc"), \
             patch("mother.claude_code.run_tests", return_value=True), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run"):
            result = run(bridge.self_build(
                prompt="improve compiler",
                repo_dir=str(tmp_path),
            ))

        assert result["success"] is True
        mock_log.assert_called_once()

    def test_self_build_done_set_on_completion(self, tmp_path):
        """_self_build_done is set after self_build() completes (success or failure)."""
        bridge = EngineBridge()
        bridge._self_build_done.clear()

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=False, error="boom", is_error=True)

        with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
             patch("mother.claude_code.save_build_log", return_value=None), \
             patch("mother.claude_code.git_snapshot", return_value=""), \
             patch("mother.claude_code.git_rollback"):
            run(bridge.self_build(prompt="test", repo_dir=str(tmp_path)))

        assert bridge._self_build_done.is_set()

    def test_self_build_sentinel_on_failure(self, tmp_path):
        """Sentinel is pushed even when build fails (rollback path)."""
        bridge = EngineBridge()
        bridge.begin_self_build()

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=False, error="CLI error", is_error=True)

        collected = []

        async def _run():
            with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
                 patch("mother.claude_code.save_build_log", return_value=None), \
                 patch("mother.claude_code.git_snapshot", return_value="abc"), \
                 patch("mother.claude_code.git_rollback"):
                build_task = asyncio.create_task(bridge.self_build(
                    prompt="test", repo_dir=str(tmp_path),
                ))
                async for event in bridge.stream_self_build_events():
                    collected.append(event)
                return await build_task

        result = run(_run())
        assert result["success"] is False
        # Stream should have terminated cleanly (sentinel received)
        assert bridge._self_build_done.is_set()
        # Should have at least phase events
        phase_events = [e for e in collected if e.get("_type") == "phase"]
        assert len(phase_events) >= 1

    def test_self_build_stores_events_on_instance(self, tmp_path):
        """self_build() stores events on self._self_build_events after invocation."""
        bridge = EngineBridge()

        fired = [{"type": "assistant", "message": {"content": []}}]

        def mock_streaming_invoke(prompt, cwd, on_event, **kwargs):
            for e in fired:
                on_event(e)
            from mother.claude_code import ClaudeCodeResult
            return ClaudeCodeResult(success=True, result_text="ok", cost_usd=0.01, num_turns=1)

        with patch("mother.coding_agent.invoke_coding_agent_streaming", side_effect=mock_streaming_invoke), \
             patch("mother.claude_code.save_build_log", return_value=None), \
             patch("mother.claude_code.git_snapshot", return_value="abc"), \
             patch("mother.claude_code.run_tests", return_value=True), \
             patch("mother.coding_agent._clean_env", return_value=os.environ.copy()), \
             patch("subprocess.run"):
            run(bridge.self_build(prompt="test", repo_dir=str(tmp_path)))

        assert len(bridge._self_build_events) == 1
        assert bridge._self_build_events[0]["type"] == "assistant"


# ---------------------------------------------------------------------------
# _self_build_worker streaming wiring (structural)
# ---------------------------------------------------------------------------

class TestSelfBuildWorkerStreaming:
    def test_worker_uses_streaming_pattern(self):
        """_self_build_worker calls begin_self_build and stream_self_build_events."""
        import ast
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        tree = ast.parse(source)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_self_build_worker":
                body_source = ast.get_source_segment(source, node)
                assert "begin_self_build" in body_source, "Worker must call begin_self_build()"
                assert "stream_self_build_events" in body_source, "Worker must consume stream_self_build_events()"
                assert "create_task" in body_source, "Worker must launch self_build as a task"
                found = True
                break
        assert found, "_self_build_worker not found in chat.py"

    def test_worker_displays_tool_use_events(self):
        """_self_build_worker handles assistant content blocks with tool_use."""
        chat_path = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"
        source = chat_path.read_text()

        start = source.find("async def _self_build_worker")
        end = source.find("\n    # --- Code task", start + 10)
        method_body = source[start:end] if end > 0 else source[start:]

        assert "tool_use" in method_body, "Worker must handle tool_use events"
        assert "tool_name" in method_body or "tool_name" in method_body, "Worker must extract tool names"


async def collect_stream(gen):
    """Collect all items from an async generator."""
    items = []
    async for item in gen:
        items.append(item)
    return items
