"""Tests for mother.coding_agent — provider-agnostic coding CLI failover."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mother.coding_agent import (
    CodingAgentProvider,
    CodingAgentResult,
    available_providers,
    default_providers,
    invoke_coding_agent,
    invoke_coding_agent_streaming,
    _is_credit_error,
    _parse_claude_result,
    _parse_codex_result,
    _parse_kimi_result,
    _parse_generic_result,
    _clean_env,
    _INVOKERS,
)


# ---------------------------------------------------------------------------
# CodingAgentResult
# ---------------------------------------------------------------------------

class TestCodingAgentResult:
    def test_frozen(self):
        r = CodingAgentResult(success=True, provider="claude")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = CodingAgentResult()
        assert r.success is False
        assert r.result_text == ""
        assert r.provider == ""
        assert r.cost_usd == 0.0
        assert r.is_error is False

    def test_all_fields(self):
        r = CodingAgentResult(
            success=True,
            result_text="done",
            session_id="s1",
            cost_usd=0.05,
            duration_seconds=12.3,
            num_turns=5,
            error="",
            is_error=False,
            provider="codex",
        )
        assert r.provider == "codex"
        assert r.num_turns == 5


# ---------------------------------------------------------------------------
# CodingAgentProvider
# ---------------------------------------------------------------------------

class TestCodingAgentProvider:
    def test_frozen(self):
        p = CodingAgentProvider(name="test", binary="/bin/test", env_var="TEST_KEY")
        with pytest.raises(AttributeError):
            p.name = "other"

    def test_is_available_no_key(self):
        p = CodingAgentProvider(name="test", binary="/bin/ls", env_var="NONEXISTENT_KEY_12345")
        # No env var set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NONEXISTENT_KEY_12345", None)
            assert p.is_available() is False

    def test_is_available_no_binary(self):
        p = CodingAgentProvider(
            name="test",
            binary="/nonexistent/binary/path",
            env_var="TEST_KEY_XYZ",
        )
        with patch.dict(os.environ, {"TEST_KEY_XYZ": "key123"}):
            assert p.is_available() is False

    def test_is_available_both_present(self):
        # Use /bin/ls as a known-existing binary
        p = CodingAgentProvider(
            name="test",
            binary="/bin/ls",
            env_var="TEST_KEY_AVAIL",
        )
        with patch.dict(os.environ, {"TEST_KEY_AVAIL": "key123"}):
            assert p.is_available() is True

    def test_is_available_empty_env_var(self):
        """Provider with empty env_var requirement is available if binary exists."""
        p = CodingAgentProvider(name="test", binary="/bin/ls", env_var="")
        assert p.is_available() is True

    def test_is_available_relative_binary_via_which(self):
        p = CodingAgentProvider(name="test", binary="ls", env_var="")
        # ls should be on PATH
        assert p.is_available() is True

    def test_is_available_relative_binary_missing(self):
        p = CodingAgentProvider(name="test", binary="nonexistent_binary_xyz", env_var="")
        assert p.is_available() is False


# ---------------------------------------------------------------------------
# default_providers / available_providers
# ---------------------------------------------------------------------------

class TestDefaultProviders:
    def test_default_providers_has_four(self):
        providers = default_providers()
        assert len(providers) == 4
        names = {p.name for p in providers}
        assert names == {"claude", "codex", "gemini", "kimi"}

    def test_priority_order(self):
        providers = default_providers()
        providers.sort(key=lambda p: p.priority)
        assert providers[0].name == "claude"
        assert providers[1].name == "codex"

    def test_available_providers_filters(self):
        providers = [
            CodingAgentProvider(name="a", binary="/bin/ls", env_var="TEST_A_KEY", priority=1),
            CodingAgentProvider(name="b", binary="/nonexistent", env_var="TEST_B_KEY", priority=0),
        ]
        with patch.dict(os.environ, {"TEST_A_KEY": "k", "TEST_B_KEY": "k"}):
            avail = available_providers(providers)
            assert len(avail) == 1
            assert avail[0].name == "a"

    def test_available_providers_sorted_by_priority(self):
        providers = [
            CodingAgentProvider(name="low", binary="/bin/ls", env_var="TEST_LP_KEY", priority=10),
            CodingAgentProvider(name="high", binary="/bin/ls", env_var="TEST_HP_KEY", priority=0),
        ]
        with patch.dict(os.environ, {"TEST_LP_KEY": "k", "TEST_HP_KEY": "k"}):
            avail = available_providers(providers)
            assert avail[0].name == "high"
            assert avail[1].name == "low"


# ---------------------------------------------------------------------------
# Credit error detection
# ---------------------------------------------------------------------------

class TestCreditErrorDetection:
    def test_credit_balance_too_low(self):
        r = CodingAgentResult(error="Credit balance is too low", is_error=True)
        assert _is_credit_error(r) is True

    def test_insufficient_quota(self):
        r = CodingAgentResult(error="insufficient_quota for this request", is_error=True)
        assert _is_credit_error(r) is True

    def test_exceeded_quota(self):
        r = CodingAgentResult(error="You exceeded your current quota", is_error=True)
        assert _is_credit_error(r) is True

    def test_billing_error(self):
        r = CodingAgentResult(error="billing issue detected", is_error=True)
        assert _is_credit_error(r) is True

    def test_rate_limit(self):
        r = CodingAgentResult(error="rate_limit exceeded", is_error=True)
        assert _is_credit_error(r) is True

    def test_payment_required(self):
        r = CodingAgentResult(error="payment required", is_error=True)
        assert _is_credit_error(r) is True

    def test_402_error(self):
        r = CodingAgentResult(error="HTTP 402 response", is_error=True)
        assert _is_credit_error(r) is True

    def test_normal_error_is_not_credit(self):
        r = CodingAgentResult(error="Syntax error in file.py", is_error=True)
        assert _is_credit_error(r) is False

    def test_timeout_is_not_credit(self):
        r = CodingAgentResult(error="Timed out after 600s", is_error=True)
        assert _is_credit_error(r) is False

    def test_credit_in_result_text(self):
        r = CodingAgentResult(result_text="Credit balance is too low", error="", is_error=True)
        assert _is_credit_error(r) is True


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

class TestParseClaudeResult:
    def test_success_with_result_data(self):
        proc = MagicMock()
        proc.returncode = 0
        result = _parse_claude_result(
            proc=proc, elapsed=5.0,
            result_data={"result": "All done", "session_id": "s1", "cost_usd": 0.02, "num_turns": 3},
            collected_lines=[], provider_name="claude",
        )
        assert result.success is True
        assert result.result_text == "All done"
        assert result.session_id == "s1"
        assert result.cost_usd == 0.02
        assert result.provider == "claude"

    def test_error_with_result_data(self):
        proc = MagicMock()
        proc.returncode = 1
        result = _parse_claude_result(
            proc=proc, elapsed=2.0,
            result_data={"result": "Credit balance is too low", "is_error": True},
            collected_lines=[], provider_name="claude",
        )
        assert result.success is False
        assert result.is_error is True
        assert "Credit" in result.error

    def test_no_result_data_nonzero_exit(self):
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr.read.return_value = "segfault"
        result = _parse_claude_result(
            proc=proc, elapsed=1.0,
            result_data=None,
            collected_lines=[], provider_name="claude",
        )
        assert result.success is False
        assert "segfault" in result.error

    def test_no_result_data_zero_exit(self):
        proc = MagicMock()
        proc.returncode = 0
        result = _parse_claude_result(
            proc=proc, elapsed=1.0,
            result_data=None,
            collected_lines=[], provider_name="claude",
        )
        assert result.success is True


class TestParseCodexResult:
    def test_success_with_turn_completed(self):
        proc = MagicMock()
        proc.returncode = 0
        event = {
            "type": "turn.completed",
            "items": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "text", "text": "Fixed the bug."}
                ]}
            ]
        }
        result = _parse_codex_result(
            proc=proc, elapsed=10.0,
            result_data=None,
            collected_lines=[json.dumps(event)],
            provider_name="codex",
        )
        assert result.success is True
        assert result.result_text == "Fixed the bug."
        assert result.provider == "codex"

    def test_failure_nonzero_exit(self):
        proc = MagicMock()
        proc.returncode = 1
        result = _parse_codex_result(
            proc=proc, elapsed=3.0,
            result_data=None,
            collected_lines=["Error: API key invalid"],
            provider_name="codex",
        )
        assert result.success is False
        assert result.is_error is True

    def test_plain_text_fallback(self):
        proc = MagicMock()
        proc.returncode = 0
        result = _parse_codex_result(
            proc=proc, elapsed=5.0,
            result_data=None,
            collected_lines=["Some plain text output"],
            provider_name="codex",
        )
        assert result.success is True
        assert "Some plain text" in result.result_text


class TestParseKimiResult:
    def test_success_with_result_event(self):
        proc = MagicMock()
        proc.returncode = 0
        result = _parse_kimi_result(
            proc=proc, elapsed=8.0,
            result_data={"result": "Completed refactoring", "is_error": False},
            collected_lines=[], provider_name="kimi",
        )
        assert result.success is True
        assert "refactoring" in result.result_text.lower()

    def test_fallback_to_assistant_content(self):
        proc = MagicMock()
        proc.returncode = 0
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "All tests pass."}]}
        }
        result = _parse_kimi_result(
            proc=proc, elapsed=6.0,
            result_data=None,
            collected_lines=[json.dumps(event)],
            provider_name="kimi",
        )
        assert result.success is True
        assert "tests pass" in result.result_text.lower()


class TestParseGenericResult:
    def test_success(self):
        proc = MagicMock()
        proc.returncode = 0
        result = _parse_generic_result(
            proc=proc, elapsed=4.0,
            result_data=None,
            collected_lines=["line1", "line2", "done"],
            provider_name="gemini",
        )
        assert result.success is True
        assert "done" in result.result_text
        assert result.provider == "gemini"

    def test_failure(self):
        proc = MagicMock()
        proc.returncode = 1
        result = _parse_generic_result(
            proc=proc, elapsed=2.0,
            result_data=None,
            collected_lines=["error: something broke"],
            provider_name="gemini",
        )
        assert result.success is False
        assert result.is_error is True

    def test_truncates_long_output(self):
        proc = MagicMock()
        proc.returncode = 0
        lines = ["x" * 200 for _ in range(20)]  # 4000 chars
        result = _parse_generic_result(
            proc=proc, elapsed=1.0,
            result_data=None,
            collected_lines=lines,
            provider_name="gemini",
        )
        assert len(result.result_text) <= 2000


# ---------------------------------------------------------------------------
# clean_env
# ---------------------------------------------------------------------------

class TestCleanEnv:
    def test_removes_claudecode(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1"}):
            env = _clean_env()
            assert "CLAUDECODE" not in env

    def test_preserves_other_vars(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            env = _clean_env()
            assert env["MY_VAR"] == "hello"


# ---------------------------------------------------------------------------
# Invoker dispatch table
# ---------------------------------------------------------------------------

class TestInvokerDispatch:
    def test_all_providers_have_invokers(self):
        for p in default_providers():
            assert p.name in _INVOKERS, f"No invoker for {p.name}"

    def test_invokers_are_callable(self):
        for name, fn in _INVOKERS.items():
            assert callable(fn), f"Invoker for {name} is not callable"


# ---------------------------------------------------------------------------
# Failover logic (invoke_coding_agent)
# ---------------------------------------------------------------------------

def _mock_invoker(return_value):
    """Create a mock invoker function that returns a fixed CodingAgentResult."""
    mock = MagicMock(return_value=return_value)
    return mock


class TestFailover:
    def test_no_providers_available(self):
        result = invoke_coding_agent(
            prompt="test",
            cwd="/tmp",
            providers=[],
        )
        assert result.success is False
        assert "No coding agents" in result.error

    def test_all_unavailable(self):
        providers = [
            CodingAgentProvider(name="a", binary="/nonexistent", env_var="NO_KEY_A"),
            CodingAgentProvider(name="b", binary="/nonexistent", env_var="NO_KEY_B"),
        ]
        result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is False
        assert "No coding agents" in result.error

    def test_first_provider_succeeds(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=True, result_text="done", provider="claude",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude}):
            result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is True
        assert result.provider == "claude"
        mock_claude.assert_called_once()

    def test_failover_on_credit_error(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=False, error="Credit balance is too low",
            is_error=True, provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=True, result_text="done by codex", provider="codex",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is True
        assert result.provider == "codex"
        mock_claude.assert_called_once()
        mock_codex.assert_called_once()

    def test_failover_on_binary_not_found(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=False, error="claude binary not found: /usr/bin/claude",
            is_error=True, provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=True, result_text="codex did it", provider="codex",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is True
        assert result.provider == "codex"

    def test_no_failover_on_real_error(self):
        """Real errors (not credit/binary) should NOT trigger failover."""
        mock_claude = _mock_invoker(CodingAgentResult(
            success=False, error="Tests failed after modification",
            is_error=True, provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=True, result_text="unused", provider="codex",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is False
        assert result.provider == "claude"
        assert "Tests failed" in result.error
        # Codex should NOT have been called
        mock_codex.assert_not_called()

    def test_all_credit_exhausted(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=False, error="Credit balance is too low",
            is_error=True, provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=False, error="You exceeded your current quota",
            is_error=True, provider="codex",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is False
        assert "All coding agents exhausted" in result.error

    def test_preferred_provider_tried_first(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=True, result_text="unused", provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=True, result_text="codex first", provider="codex",
        ))
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            result = invoke_coding_agent(
                prompt="test", cwd="/tmp", providers=providers, preferred="codex",
            )
        assert result.success is True
        assert result.provider == "codex"
        # Claude should NOT have been called since codex succeeded first
        mock_claude.assert_not_called()

    def test_on_event_receives_provider_attempt(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=True, result_text="done", provider="claude",
        ))
        events = []
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude}):
            result = invoke_coding_agent(
                prompt="test", cwd="/tmp", providers=providers,
                on_event=lambda e: events.append(e),
            )
        assert result.success is True
        attempt_events = [e for e in events if e.get("_type") == "provider_attempt"]
        assert len(attempt_events) == 1
        assert attempt_events[0]["provider"] == "claude"

    def test_on_event_receives_failover_event(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=False, error="Credit balance is too low",
            is_error=True, provider="claude",
        ))
        mock_codex = _mock_invoker(CodingAgentResult(
            success=True, result_text="done", provider="codex",
        ))
        events = []
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
            CodingAgentProvider(name="codex", binary="/bin/ls", env_var="", priority=1),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude, "codex": mock_codex}):
            invoke_coding_agent(
                prompt="test", cwd="/tmp", providers=providers,
                on_event=lambda e: events.append(e),
            )
        failover_events = [e for e in events if e.get("_type") == "provider_failover"]
        assert len(failover_events) == 1
        assert failover_events[0]["failed_provider"] == "claude"
        assert failover_events[0]["reason"] == "credit_exhausted"


# ---------------------------------------------------------------------------
# invoke_coding_agent_streaming alias
# ---------------------------------------------------------------------------

class TestStreamingAlias:
    def test_streaming_alias(self):
        mock_claude = _mock_invoker(CodingAgentResult(
            success=True, result_text="done", provider="claude",
        ))
        events = []
        providers = [
            CodingAgentProvider(name="claude", binary="/bin/ls", env_var="", priority=0),
        ]
        with patch.dict(_INVOKERS, {"claude": mock_claude}):
            result = invoke_coding_agent_streaming(
                prompt="test", cwd="/tmp",
                on_event=lambda e: events.append(e),
                providers=providers,
            )
        assert result.success is True


# ---------------------------------------------------------------------------
# Unknown provider in dispatch
# ---------------------------------------------------------------------------

class TestUnknownProvider:
    def test_unknown_provider_skipped(self):
        providers = [
            CodingAgentProvider(name="unknown_agent", binary="/bin/ls", env_var="", priority=0),
        ]
        result = invoke_coding_agent(prompt="test", cwd="/tmp", providers=providers)
        assert result.success is False
        assert "exhausted" in result.error.lower() or "No coding agents" in result.error
