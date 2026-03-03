"""Tests for anti-fragile loop — rejection hints fed into context."""

import pytest

from mother.context import ContextData, synthesize_situation


class TestRejectionHintsInContext:

    def test_hints_appear_in_situation(self):
        data = ContextData(
            rejection_hints=["Improve verification scores", "Ensure provenance chain"],
            rejection_count=3,
        )
        situation = synthesize_situation(data)
        assert "Previous compilation issues" in situation
        assert "Improve verification scores" in situation

    def test_no_hints_no_line(self):
        data = ContextData(rejection_hints=[], rejection_count=0)
        situation = synthesize_situation(data)
        assert "Previous compilation issues" not in situation

    def test_hints_capped_at_3(self):
        data = ContextData(
            rejection_hints=["A", "B", "C", "D", "E"],
        )
        situation = synthesize_situation(data)
        # Should only include first 3
        assert "A" in situation
        assert "C" in situation
        # D should NOT appear in the hints line (capped at 3)
        lines = [l for l in situation.split("\n") if "Previous compilation" in l]
        assert len(lines) == 1
        assert "D" not in lines[0]
