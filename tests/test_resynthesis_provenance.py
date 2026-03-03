"""Tests for resynthesis provenance tagging (Build 7)."""

import pytest
from unittest.mock import MagicMock, patch
from core.engine import MotherlabsEngine


class TestResynthesisGapIds:
    """_targeted_resynthesis() tags gaps with IDs for provenance."""

    def _make_engine(self):
        engine = MotherlabsEngine.__new__(MotherlabsEngine)
        engine.synthesis_agent = MagicMock()
        engine._extract_json = MagicMock(return_value={
            "components": [],
            "relationships": [],
        })
        engine._merge_blueprints = MagicMock(side_effect=lambda a, b: a)
        engine.provider_name = "mock"
        return engine

    def _make_state(self):
        state = MagicMock()
        state.known = {}
        state.insights = []
        state.history = []
        state.add_message = MagicMock()
        return state

    def test_gap_ids_in_prompt(self):
        """Resynthesis prompt contains gap IDs like [gap_1]."""
        engine = self._make_engine()
        state = self._make_state()

        # Mock synthesis agent to capture the prompt
        captured_prompts = []
        def capture_run(state, msg, max_tokens=None):
            captured_prompts.append(msg.content)
            return MagicMock(content='{"components": [], "relationships": []}')
        engine.synthesis_agent.run = capture_run

        blueprint = {"components": [{"name": "Auth"}], "relationships": []}
        verification = {
            "completeness": {"gaps": ["Missing: 'database'"]},
            "consistency": {},
            "coherence": {},
        }

        engine._targeted_resynthesis(blueprint, verification, state)

        assert len(captured_prompts) == 1
        assert "[gap_1]" in captured_prompts[0]

    def test_provenance_instruction_in_prompt(self):
        """Resynthesis prompt includes provenance tagging instruction."""
        engine = self._make_engine()
        state = self._make_state()

        captured_prompts = []
        def capture_run(state, msg, max_tokens=None):
            captured_prompts.append(msg.content)
            return MagicMock(content='{"components": [], "relationships": []}')
        engine.synthesis_agent.run = capture_run

        blueprint = {"components": [{"name": "Auth"}], "relationships": []}
        verification = {
            "completeness": {"gaps": ["Missing: 'database'"]},
            "consistency": {},
            "coherence": {},
        }

        engine._targeted_resynthesis(blueprint, verification, state)
        assert "resynthesis:" in captured_prompts[0]

    def test_compression_losses_get_gap_ids(self):
        """Compression losses from closed-loop gate get gap IDs."""
        engine = self._make_engine()
        state = self._make_state()
        state.known["compression_losses"] = ["user roles", "permissions"]

        captured_prompts = []
        def capture_run(state, msg, max_tokens=None):
            captured_prompts.append(msg.content)
            return MagicMock(content='{"components": [], "relationships": []}')
        engine.synthesis_agent.run = capture_run

        blueprint = {"components": [{"name": "Auth"}], "relationships": []}
        verification = {"completeness": {}, "consistency": {}, "coherence": {}}

        engine._targeted_resynthesis(blueprint, verification, state)
        assert "[gap_1]" in captured_prompts[0]
        assert "[gap_2]" in captured_prompts[0]

    def test_no_gaps_returns_blueprint(self):
        """No gaps → blueprint returned unchanged."""
        engine = self._make_engine()
        state = self._make_state()

        blueprint = {"components": [{"name": "Auth"}], "relationships": []}
        verification = {"completeness": {}, "consistency": {}, "coherence": {}}

        result = engine._targeted_resynthesis(blueprint, verification, state)
        assert result == blueprint

    def test_multiple_gap_sources(self):
        """Gaps from different verification dimensions get unique IDs."""
        engine = self._make_engine()
        state = self._make_state()
        state.known["compression_losses"] = ["roles"]

        captured_prompts = []
        def capture_run(state, msg, max_tokens=None):
            captured_prompts.append(msg.content)
            return MagicMock(content='{"components": [], "relationships": []}')
        engine.synthesis_agent.run = capture_run

        blueprint = {"components": [{"name": "Auth"}], "relationships": []}
        verification = {
            "completeness": {"gaps": ["Missing: 'db'"]},
            "consistency": {"conflicts": ["type mismatch"]},
            "coherence": {"suggested_fixes": ["add link"]},
        }

        engine._targeted_resynthesis(blueprint, verification, state)
        prompt = captured_prompts[0]
        # Should have gap_1 (compression), gap_2 (completeness),
        # gap_3 (consistency), gap_4 (coherence)
        assert "[gap_1]" in prompt
        assert "[gap_2]" in prompt
        assert "[gap_3]" in prompt
        assert "[gap_4]" in prompt
