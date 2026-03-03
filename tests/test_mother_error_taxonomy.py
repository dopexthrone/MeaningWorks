"""
Tests for Mother error taxonomy — structured error classification.

Covers: ErrorClassification frozen dataclass, classify_error rule matching,
fingerprint format, retriable/user_actionable flags, compute_error_impact,
summarize_errors.
"""

import pytest

from mother.error_taxonomy import (
    ErrorClassification,
    classify_error,
    compute_error_impact,
    summarize_errors,
)


class TestErrorClassificationDefaults:

    def test_defaults(self):
        c = ErrorClassification()
        assert c.category == "unknown"
        assert c.severity == 0.5
        assert c.retriable is False
        assert c.user_actionable is False
        assert c.phase == ""
        assert c.fingerprint == ""

    def test_frozen(self):
        c = ErrorClassification()
        with pytest.raises(AttributeError):
            c.category = "auth"


class TestClassifyError:

    def test_connection_error(self):
        c = classify_error(ConnectionError("connection refused"), phase="chat")
        assert c.category == "connection"
        assert c.severity == 0.3
        assert c.retriable is True
        assert c.fingerprint == "connection:chat"

    def test_connection_in_message(self):
        c = classify_error(Exception("could not connect to server"), phase="chat")
        assert c.category == "connection"

    def test_auth_401(self):
        c = classify_error(Exception("HTTP 401 Unauthorized"), phase="chat")
        assert c.category == "auth"
        assert c.user_actionable is True
        assert c.fingerprint == "auth:401"

    def test_auth_unauthorized(self):
        c = classify_error(Exception("unauthorized access"), phase="compile")
        assert c.category == "auth"

    def test_rate_limit_429(self):
        c = classify_error(Exception("HTTP 429 Too Many Requests"), phase="chat")
        assert c.category == "rate_limit"
        assert c.severity == 0.1
        assert c.retriable is True

    def test_cost_cap(self):
        c = classify_error(Exception("cost cap exceeded"), phase="compile")
        assert c.category == "cost"
        assert c.severity == 0.4
        assert c.user_actionable is True

    def test_budget_limit(self):
        c = classify_error(Exception("budget limit reached"), phase="chat")
        assert c.category == "cost"

    def test_timeout_error(self):
        c = classify_error(TimeoutError("operation timed out"), phase="compile")
        assert c.category == "timeout"
        assert c.retriable is True
        assert c.fingerprint == "timeout:compile"

    def test_timeout_in_message(self):
        c = classify_error(Exception("request timeout"), phase="chat")
        assert c.category == "timeout"

    def test_compile_failure_fallthrough(self):
        c = classify_error(RuntimeError("something went wrong"), phase="compile")
        assert c.category == "compile_failure"
        assert c.severity == 0.5
        assert "RuntimeError" in c.fingerprint

    def test_build_failure_fallthrough(self):
        c = classify_error(RuntimeError("build broke"), phase="build")
        assert c.category == "build_failure"
        assert c.severity == 0.6

    def test_launch_failure_fallthrough(self):
        c = classify_error(OSError("can't start"), phase="launch")
        assert c.category == "launch_failure"
        assert c.severity == 0.7

    def test_validation_error(self):
        c = classify_error(Exception("validation failed: invalid schema"), phase="chat")
        assert c.category == "validation"
        assert c.user_actionable is True

    def test_unknown_fallback(self):
        c = classify_error(RuntimeError("weird error"), phase="chat")
        assert c.category == "unknown"
        assert c.severity == 0.5
        assert c.fingerprint == "unknown:RuntimeError"


class TestClassifyErrorFingerprint:

    def test_fingerprint_format(self):
        c = classify_error(ConnectionError("fail"), phase="compile")
        assert ":" in c.fingerprint


class TestClassifyErrorFlags:

    def test_retriable_flagging(self):
        # Connection, rate_limit, timeout should be retriable
        assert classify_error(ConnectionError("x"), "chat").retriable is True
        assert classify_error(Exception("429"), "chat").retriable is True
        assert classify_error(TimeoutError("x"), "chat").retriable is True
        # Auth, cost should NOT be retriable
        assert classify_error(Exception("401 auth"), "chat").retriable is False
        assert classify_error(Exception("cost cap"), "chat").retriable is False

    def test_user_actionable_flagging(self):
        # Auth, cost, validation should be user-actionable
        assert classify_error(Exception("401"), "chat").user_actionable is True
        assert classify_error(Exception("cost"), "chat").user_actionable is True
        assert classify_error(Exception("invalid input"), "chat").user_actionable is True
        # Connection, timeout should NOT be user-actionable
        assert classify_error(ConnectionError("x"), "chat").user_actionable is False
        assert classify_error(TimeoutError("x"), "chat").user_actionable is False


class TestComputeErrorImpact:

    def test_empty(self):
        result = compute_error_impact([])
        assert result["total_severity"] == 0.0
        assert result["max_severity"] == 0.0

    def test_single(self):
        c = ErrorClassification(severity=0.5, retriable=True)
        result = compute_error_impact([c])
        assert result["total_severity"] == 0.5
        assert result["max_severity"] == 0.5
        assert result["retriable_fraction"] == 1.0

    def test_mixed(self):
        classifications = [
            ErrorClassification(severity=0.3, retriable=True, user_actionable=False),
            ErrorClassification(severity=0.7, retriable=False, user_actionable=True),
        ]
        result = compute_error_impact(classifications)
        assert result["total_severity"] == pytest.approx(1.0)
        assert result["max_severity"] == pytest.approx(0.7)
        assert result["retriable_fraction"] == pytest.approx(0.5)
        assert result["user_actionable_fraction"] == pytest.approx(0.5)


class TestSummarizeErrors:

    def test_empty(self):
        assert summarize_errors([]) == ""

    def test_single(self):
        c = classify_error(ConnectionError("fail"), "chat")
        result = summarize_errors([c])
        assert "connection" in result
        assert "1" in result

    def test_mixed(self):
        classifications = [
            classify_error(ConnectionError("fail"), "chat"),
            classify_error(Exception("401 auth"), "chat"),
            classify_error(ConnectionError("retry"), "compile"),
        ]
        result = summarize_errors(classifications)
        assert "connection" in result
        assert "auth" in result
        # Should mention retriable count
        assert "retriable" in result
