"""Tests for provenance-aware trust + Governor refactor (Build 8)."""

import pytest
from core.verification import (
    provenance_integrity_ratio,
    validate_provenance_refs,
    score_traceability,
)
from agents.swarm import GovernorAgent


class TestProvenanceIntegrity:
    """provenance_integrity_ratio scores structural provenance."""

    def test_full_integrity(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"},
            {"name": "B", "derived_from": "grid:EXC.BHV.APP.HOW.SFT"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT", "EXC.BHV.APP.HOW.SFT"]
        ratio = provenance_integrity_ratio(components, postcodes)
        assert ratio == 1.0

    def test_partial_integrity(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"},
            {"name": "B", "derived_from": "user described this"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT"]
        ratio = provenance_integrity_ratio(components, postcodes)
        assert ratio == 0.5

    def test_zero_integrity_text_only(self):
        components = [
            {"name": "A", "derived_from": "user intent"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT"]
        ratio = provenance_integrity_ratio(components, postcodes)
        assert ratio == 0.0

    def test_zero_integrity_no_grid(self):
        components = [{"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"}]
        ratio = provenance_integrity_ratio(components, [])
        assert ratio == 0.0


class TestTraceabilityGridBonus:
    """score_traceability integrates grid provenance bonus."""

    def test_grid_validated_in_details(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT and more text"},
        ]
        result = score_traceability(components, ["STR.ENT.CMP.WHAT.SFT"])
        assert "grid_validated" in result.details

    def test_no_grid_no_mention(self):
        components = [
            {"name": "A", "derived_from": "user said this thing"},
        ]
        result = score_traceability(components)
        assert "grid_validated" not in result.details

    def test_score_increases_with_grid(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT plus context text"},
            {"name": "B", "derived_from": "grid:EXC.BHV.APP.HOW.SFT plus context text"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT", "EXC.BHV.APP.HOW.SFT"]

        score_no = score_traceability(components).score
        score_yes = score_traceability(components, postcodes).score
        assert score_yes >= score_no


class TestGovernorDialogueProvenance:
    """GovernorAgent.validate_dialogue_provenance() checks insight grounding."""

    def _make_state(self, input_text, insights):
        from unittest.mock import MagicMock
        state = MagicMock()
        state.known = {"input": input_text}
        state.insights = insights
        return state

    def test_grounded_insights(self):
        state = self._make_state(
            "Build a task management application with projects",
            ["Task management requires project organization", "Projects contain multiple tasks"]
        )
        gov = GovernorAgent()
        result = gov.validate_dialogue_provenance(state)
        assert result["grounded"] >= 1
        assert result["ratio"] > 0.0

    def test_ungrounded_insights(self):
        state = self._make_state(
            "Build a task manager",
            ["The quantum entanglement framework requires neutron calibration"]
        )
        gov = GovernorAgent()
        result = gov.validate_dialogue_provenance(state)
        assert result["ungrounded"] >= 1

    def test_empty_insights(self):
        state = self._make_state("Build something", [])
        gov = GovernorAgent()
        result = gov.validate_dialogue_provenance(state)
        assert result["ratio"] == 1.0
        assert result["grounded"] == 0

    def test_empty_input(self):
        state = self._make_state("", ["some insight"])
        gov = GovernorAgent()
        result = gov.validate_dialogue_provenance(state)
        assert result["ratio"] == 1.0

    def test_mixed_grounding(self):
        state = self._make_state(
            "Build an authentication system with OAuth tokens and session management",
            [
                "Authentication uses OAuth for external login",
                "Quantum computing enables faster database queries",
                "Session management with tokens provides security for authentication",
            ]
        )
        gov = GovernorAgent()
        result = gov.validate_dialogue_provenance(state)
        # At least 2 of 3 should be grounded (auth+OAuth, session+tokens)
        assert result["grounded"] >= 2
        assert result["ungrounded"] >= 1  # quantum insight is ungrounded
        assert 0.0 < result["ratio"] < 1.0
