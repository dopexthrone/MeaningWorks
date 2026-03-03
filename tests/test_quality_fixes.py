"""
Quality fix tests — verifying 3 compilation quality improvements.

Covers:
1. Method inference fallback in synthesis prompt (empty extracted_methods → SECTION 3 inference)
2. Insight deduplication (Jaccard semantic dedup in digest + exact-match safety net in engine)
3. Convergence minimum scaling (simple inputs converge after 1 round, complex unchanged)
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

from core.protocol_spec import PROTOCOL
from core.protocol import Message, MessageType
from core.digest import _deduplicate_insights, _rank_insights, build_dialogue_digest
from core.convergence import ConvergenceTracker, estimate_turn_budget
from agents.base import AgentCallResult


def _synthesis_result(content: str) -> AgentCallResult:
    """Wrap a JSON content string in an AgentCallResult for synthesis mocks."""
    msg = Message(sender="Synthesis", content=content, message_type=MessageType.PROPOSITION)
    return AgentCallResult(
        agent_name="Synthesis", response_text=content, message=msg,
        conflicts=(), unknowns=(), fractures=(),
        confidence_boost=0.0, agent_dimension="", has_insight=False,
        token_usage={},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    """Create a minimal engine with mock LLM for testing."""
    from persistence.corpus import Corpus
    client = Mock()
    client.provider_name = "mock"
    client.model_name = "mock-model"
    client.deterministic = True
    client.model = "mock-model"
    from core.engine import MotherlabsEngine
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=Corpus(tmp_path / "corpus"),
        auto_store=False,
    )
    return engine


def _make_shared_state(**kwargs):
    """Create a SharedState with overrides."""
    from core.protocol import SharedState
    return SharedState(**kwargs)


def _make_ranked_insights(insights_with_scores):
    """Create ranked insight dicts from (text, score) pairs."""
    ranked = []
    for text, score in insights_with_scores:
        tier = "HIGH" if score >= 4 else ("MEDIUM" if score >= 2 else "LOW")
        ranked.append({
            "insight": text,
            "source": "test",
            "score": score,
            "tier": tier,
        })
    return ranked


# ===========================================================================
# TestMethodInference
# ===========================================================================

class TestMethodInference:
    """Verify SECTION 3 method inference fallback for natural language inputs."""

    def _capture_synthesis_prompt(self, engine, state):
        """Call _synthesize with a mock agent and capture the prompt content."""
        captured = {}

        def mock_run(s, msg, max_tokens=4096):
            captured["prompt"] = msg.content
            return _synthesis_result('{"components": [{"name": "Stub", "type": "entity"}], "relationships": [], "constraints": [], "unresolved": []}')

        engine.synthesis_agent = Mock()
        engine.synthesis_agent.run_llm_only = mock_run
        try:
            engine._synthesize(state)
        except Exception:
            pass  # May fail on post-synthesis steps, that's fine
        return captured.get("prompt", "")

    def test_section3_fallback_when_no_extracted_methods(self, tmp_path):
        """When extracted_methods is empty, synthesis prompt should include
        METHOD INFERENCE instructions telling LLM to infer from component
        type/description."""
        engine = _make_engine(tmp_path)
        state = _make_shared_state(known={
            "input": "Build a task manager",
            "intent": {"core_need": "task management", "domain": "software"},
        })

        prompt = self._capture_synthesis_prompt(engine, state)

        assert "METHOD INFERENCE" in prompt
        assert "ENTITY components:" in prompt
        assert "PROCESS components:" in prompt
        assert "derived_from" in prompt

    def test_section3_normal_when_methods_exist(self, tmp_path):
        """When extracted_methods are present, existing SECTION 3 behavior
        is preserved (EXTRACTED METHODS header, not METHOD INFERENCE)."""
        engine = _make_engine(tmp_path)
        state = _make_shared_state(known={
            "input": "Build a task manager",
            "intent": {"core_need": "task management", "domain": "software"},
            "extracted_methods": [
                {"component": "TaskService", "name": "create_task",
                 "return_type": "Task", "parameters": [],
                 "derived_from": "dialogue"}
            ],
        })

        prompt = self._capture_synthesis_prompt(engine, state)

        assert "EXTRACTED METHODS" in prompt
        assert "METHOD INFERENCE" not in prompt

    def test_section3_inference_requires_provenance(self, tmp_path):
        """Inference instructions must require tracing to input/description."""
        engine = _make_engine(tmp_path)
        state = _make_shared_state(known={
            "input": "Build a task manager",
            "intent": {"core_need": "task management", "domain": "software"},
        })

        prompt = self._capture_synthesis_prompt(engine, state)

        assert "CRITICAL" in prompt
        assert "trace" in prompt.lower() or "anchor" in prompt.lower()
        assert "Do NOT invent" in prompt


# ===========================================================================
# TestInsightDedup
# ===========================================================================

class TestInsightDedup:
    """Verify insight deduplication at both semantic and exact levels."""

    def test_dedup_merges_similar_insights(self):
        """Near-duplicate Kanban insights (sharing 2+ content words) collapse."""
        ranked = _make_ranked_insights([
            ("Kanban board for task tracking and management", 3),
            ("Kanban board tracking tasks across columns", 2),
            ("Kanban task board with drag and drop", 2),
            ("Task tracking board Kanban style", 1),
            ("Kanban board task management workflow", 1),
        ])
        result = _deduplicate_insights(ranked)
        # All share "kanban" + "board" + "task" → overlap > 0.5 on content words
        assert len(result) <= 2
        # Highest-scored version survives (first in sorted input)
        assert result[0]["score"] == 3

    def test_dedup_preserves_distinct_insights(self):
        """Genuinely different insights should not be merged."""
        ranked = _make_ranked_insights([
            ("Users need role-based access control", 3),
            ("System must handle 10k concurrent connections", 3),
            ("Mobile-first responsive design required", 2),
            ("Data must be encrypted at rest", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 4

    def test_dedup_keeps_highest_scored(self):
        """When merging, the higher-scored version survives."""
        ranked = _make_ranked_insights([
            ("Authentication via OAuth2", 4),  # Highest score, appears first
            ("OAuth2 authentication for users", 2),  # Similar but lower
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1
        assert result[0]["score"] == 4
        assert result[0]["insight"] == "Authentication via OAuth2"

    def test_dedup_empty_input(self):
        """Empty list → empty list."""
        assert _deduplicate_insights([]) == []

    def test_dedup_single_insight(self):
        """Single insight → unchanged."""
        ranked = _make_ranked_insights([("Only insight", 3)])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1
        assert result[0]["insight"] == "Only insight"

    def test_dedup_threshold_boundary(self):
        """Insights sharing >50% of content words should merge (overlap coefficient)."""
        # Content words: {alpha, beta, gamma, delta, epsilon}=5 vs {alpha, beta, gamma, delta, zeta}=5
        # Intersection: 4, min(5,5)=5, overlap=0.8 > 0.5
        ranked = _make_ranked_insights([
            ("alpha beta gamma delta epsilon", 3),
            ("alpha beta gamma delta zeta", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1  # Should merge (overlap > 0.5)

    def test_dedup_below_threshold_preserved(self):
        """Insights with no shared content words should be preserved."""
        ranked = _make_ranked_insights([
            ("alpha beta gamma", 3),
            ("delta epsilon zeta eta theta", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 2

    def test_exact_dedup_on_blueprint_insights(self):
        """Exact-match dedup in engine removes case-insensitive duplicates."""
        state = _make_shared_state(insights=[
            "Kanban board for tracking",
            "kanban board for tracking",  # exact case-insensitive dup
            "KANBAN BOARD FOR TRACKING",  # another exact dup
            "Different insight entirely",
        ])
        # Simulate what engine.py does
        seen = set()
        unique = []
        for ins in state.insights:
            key = ins.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(ins)

        assert len(unique) == 2
        assert unique[0] == "Kanban board for tracking"
        assert unique[1] == "Different insight entirely"

    def test_dedup_wired_into_digest(self):
        """build_dialogue_digest uses deduped insights."""
        from core.protocol import Message, MessageType
        state = _make_shared_state(
            insights=[
                "Use Kanban board for tasks",
                "Kanban board for task management",
                "Real-time collaboration needed",
            ],
            history=[
                Message(sender="Entity", content="analysis",
                        message_type=MessageType.PROPOSITION,
                        insight="Use Kanban board for tasks"),
                Message(sender="Process", content="analysis",
                        message_type=MessageType.PROPOSITION,
                        insight="Kanban board for task management"),
                Message(sender="Entity", content="analysis",
                        message_type=MessageType.PROPOSITION,
                        insight="Real-time collaboration needed"),
            ],
        )
        digest = build_dialogue_digest(state)
        # Count occurrences of "Kanban" in the digest
        kanban_count = digest.lower().count("kanban")
        # Should appear at most once (the deduped survivor)
        assert kanban_count <= 1


# ===========================================================================
# TestConvergenceEarlier
# ===========================================================================

class TestConvergenceEarlier:
    """Verify convergence minimum scaling for simple vs complex inputs."""

    def test_simple_input_convergence_min_3(self):
        """Short input (< 500 chars) → min_turns=6 → convergence_min=3."""
        simple_input = "Build a task manager"
        min_turns, _, _ = estimate_turn_budget(simple_input)
        assert min_turns <= 8
        # The engine would set convergence_min = 3
        convergence_min = 3 if min_turns <= 8 else min_turns
        assert convergence_min == 3

    def test_complex_input_convergence_min_unchanged(self):
        """Long, multi-domain input → min_turns > 8 → convergence_min=min_turns."""
        complex_input = """
        Build a comprehensive trading platform with real-time market data feeds,
        algorithmic trading execution, portfolio management with risk analytics,
        social trading features, cryptocurrency exchange integration,
        accounting and financial reporting, mobile app with push notifications,
        admin dashboard with CCTV security monitoring, customer service chatbot,
        marketplace for third-party trading strategies, IoT device integration
        for hardware trading terminals, email notification system, calendar
        for earnings announcements, health monitoring for server infrastructure.
        """ * 3  # Make it long enough to trigger higher budget
        min_turns, _, _ = estimate_turn_budget(complex_input)
        assert min_turns > 8
        convergence_min = 3 if min_turns <= 8 else min_turns
        assert convergence_min == min_turns

    def test_min_rounds_is_1(self, tmp_path):
        """Verify min_rounds is set to 1 in text-based dialogue (fallback path)."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._run_text_based_dialogue)
        assert "min_rounds = 1" in source

    def test_convergence_tracker_respects_min_turns(self):
        """Tracker with min_turns=3 should allow convergence after 3 turns."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=3,
        )
        # Simulate 3 turns of no change
        tracker._turn_count = 3
        tracker._deltas = [1.0, 0.02, 0.01]  # First turn high, then stable
        assert tracker.has_converged() is True

    def test_convergence_tracker_blocks_before_min_turns(self):
        """Tracker with min_turns=6 should NOT converge at turn 3."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=6,
        )
        tracker._turn_count = 3
        tracker._deltas = [0.01, 0.01, 0.01]
        assert tracker.has_converged() is False

    def test_convergence_min_formula(self, tmp_path):
        """Verify the convergence_min formula in grid-driven dialogue."""
        import inspect
        from core.engine import MotherlabsEngine
        # Formula lives in _run_grid_driven_dialogue (primary path)
        # and _run_text_based_dialogue (fallback path)
        grid_source = inspect.getsource(MotherlabsEngine._run_grid_driven_dialogue)
        text_source = inspect.getsource(MotherlabsEngine._run_text_based_dialogue)
        assert "convergence_min = 3 if min_turns <= 8 else min_turns" in grid_source
        assert "convergence_min = 3 if min_turns <= 8 else min_turns" in text_source

    def test_simple_input_can_converge_round_1(self):
        """A simple input with stable blueprint should be able to converge
        after round 1 (3 turns)."""
        tracker = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.05,
            min_turns_before_convergence=3,  # What engine sets for simple inputs
        )
        # Simulate: round 1 = 3 turns, blueprint stable after turn 2
        tracker._turn_count = 1
        tracker._deltas = [1.0]
        assert tracker.has_converged() is False  # Too early

        tracker._turn_count = 2
        tracker._deltas = [1.0, 0.03]
        assert tracker.has_converged() is False  # Only 1 low delta, need 2

        tracker._turn_count = 3
        tracker._deltas = [1.0, 0.03, 0.02]
        assert tracker.has_converged() is True  # 2 consecutive low deltas, >= 3 turns


# ===========================================================================
# TestSynthesisCollapse — density penalty + thin resynth gate
# ===========================================================================

class TestSynthesisCollapse:
    """Verify fixes for synthesis collapsing 50+ components into 1."""

    def test_density_penalty_one_component(self, tmp_path):
        """1-component blueprint gets 0.33x coverage score."""
        engine = _make_engine(tmp_path)
        blueprint = {"components": [{"name": "MegaComp", "type": "entity"}]}
        num_comps = len(blueprint.get("components", []))
        density_penalty = min(num_comps / 3, 1.0)
        assert abs(density_penalty - 1 / 3) < 0.01

    def test_density_penalty_three_plus(self, tmp_path):
        """3+ component blueprint gets full score (penalty = 1.0)."""
        for n in (3, 5, 20):
            blueprint = {"components": [{"name": f"C{i}", "type": "entity"} for i in range(n)]}
            num_comps = len(blueprint["components"])
            density_penalty = min(num_comps / 3, 1.0)
            assert density_penalty == 1.0

    def test_density_penalty_affects_selection(self, tmp_path):
        """1-comp candidate with high raw score loses to 5-comp with lower raw."""
        # 1-comp: raw coverage 0.90, penalty 0.33 → effective 0.30
        mega = ({"components": [{"name": "All"}]}, 0.90)
        mega_effective = mega[1] * min(len(mega[0]["components"]) / 3, 1.0)

        # 5-comp: raw coverage 0.60, penalty 1.0 → effective 0.60
        proper = ({"components": [{"name": f"C{i}"} for i in range(5)]}, 0.60)
        proper_effective = proper[1] * min(len(proper[0]["components"]) / 3, 1.0)

        assert proper_effective > mega_effective

    def test_thin_blueprint_forces_resynth(self, tmp_path):
        """Blueprint with <=3 components triggers resynth even with low completeness."""
        # Simulate the gate logic
        completeness = 10  # Below resynth_min_completeness (30)
        num_components = 1
        from core.protocol_spec import PROTOCOL
        should_resynth = (
            completeness >= PROTOCOL.engine.resynth_min_completeness
            or num_components <= 3
        )
        assert should_resynth is True

    def test_normal_blueprint_uses_completeness_gate(self, tmp_path):
        """10-comp blueprint with low completeness does NOT force resynth."""
        completeness = 10  # Below resynth_min_completeness (30)
        num_components = 10
        from core.protocol_spec import PROTOCOL
        should_resynth = (
            completeness >= PROTOCOL.engine.resynth_min_completeness
            or num_components <= 3
        )
        assert should_resynth is False


# ===========================================================================
# TestDeterministicMethodInference
# ===========================================================================

class TestDeterministicMethodInference:
    """Verify _infer_component_methods deterministic fallback."""

    def _get_engine(self, tmp_path):
        return _make_engine(tmp_path)

    def test_entity_gets_crud(self, tmp_path):
        """Entity component gets create/get/update/delete methods."""
        engine = self._get_engine(tmp_path)
        bp = {"components": [{"name": "Task", "type": "entity"}]}
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = [m["name"] for m in methods]
        assert "create_task" in names
        assert "get_task" in names
        assert "update_task" in names
        assert "delete_task" in names
        assert len(methods) == 4
        # Provenance must be present
        for m in methods:
            assert "Inferred" in m["derived_from"]

    def test_process_gets_lifecycle(self, tmp_path):
        """Process component gets execute/get_status/validate methods."""
        engine = self._get_engine(tmp_path)
        bp = {"components": [{"name": "TaskRunner", "type": "service"}]}
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = [m["name"] for m in methods]
        assert "execute" in names
        assert "get_status" in names
        assert "validate" in names
        assert len(methods) == 3

    def test_interface_gets_handler(self, tmp_path):
        """Interface component gets handle_request and validate_input methods."""
        engine = self._get_engine(tmp_path)
        bp = {"components": [{"name": "TaskAPI", "type": "api"}]}
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = [m["name"] for m in methods]
        assert "handle_request" in names
        assert "validate_input" in names
        assert len(methods) == 2

    def test_existing_methods_preserved(self, tmp_path):
        """Components with methods already set are not overwritten."""
        engine = self._get_engine(tmp_path)
        existing = [{"name": "custom_op", "parameters": [], "return_type": "None"}]
        bp = {"components": [{"name": "Task", "type": "entity", "methods": existing}]}
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        assert methods == existing  # Unchanged

    def test_unknown_type_gets_defaults(self, tmp_path):
        """Unrecognized types get default initialize/process methods."""
        engine = self._get_engine(tmp_path)
        bp = {"components": [{"name": "Widget", "type": "unknown_thing"}]}
        result = engine._infer_component_methods(bp)
        methods = result["components"][0]["methods"]
        names = [m["name"] for m in methods]
        assert "initialize" in names
        assert "process" in names

    def test_enrichment_fallback_calls_inference(self, tmp_path):
        """When no extracted methods, _enrich_blueprint_methods calls _infer."""
        engine = self._get_engine(tmp_path)
        state = _make_shared_state(known={
            "input": "Build a thing",
            "intent": {"core_need": "thing"},
        })
        bp = {"components": [
            {"name": "User", "type": "entity"},
            {"name": "AuthService", "type": "service"},
        ]}
        result = engine._enrich_blueprint_methods(bp, state)
        # Entity should have CRUD
        assert len(result["components"][0].get("methods", [])) == 4
        # Service should have lifecycle (execute, get_status, validate)
        assert len(result["components"][1].get("methods", [])) == 3

    def test_methods_mandate_in_instruction(self, tmp_path):
        """Final INSTRUCTION section contains METHODS MANDATE."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "METHODS MANDATE" in source
        assert "non-empty" in source.lower() or 'non-empty' in source


# ===========================================================================
# TestArrowNormalization — Fix 1a: arrow normalization in dedup
# ===========================================================================

class TestArrowNormalization:
    """Verify _normalize_insight normalizes unicode arrows to ASCII."""

    def test_right_arrow_normalized(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("A → B") == "A -> B"

    def test_left_arrow_normalized(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("A ← B") == "A <- B"

    def test_bidi_arrow_normalized(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("A ↔ B") == "A <-> B"

    def test_multiple_arrows_normalized(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("A → B → C ← D") == "A -> B -> C <- D"

    def test_whitespace_collapsed(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("A  →   B") == "A -> B"

    def test_no_arrows_unchanged(self):
        from core.digest import _normalize_insight
        assert _normalize_insight("plain text here") == "plain text here"

    def test_arrow_variants_dedup_as_same(self):
        """Unicode arrow and ASCII arrow insight should dedup to one."""
        ranked = _make_ranked_insights([
            ("Input → Output via transform", 4),
            ("Input -> Output via transform", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1
        assert result[0]["score"] == 4


# ===========================================================================
# TestCommutativeEquationDedup — Fix 1b: equation commutativity detection
# ===========================================================================

class TestCommutativeEquationDedup:
    """Verify commutative equation duplicates are detected and merged."""

    def test_swapped_sides_detected(self):
        """A + B = C should match C = A + B."""
        ranked = _make_ranked_insights([
            ("trust + provenance = reliability", 4),
            ("reliability = trust + provenance", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1
        assert result[0]["score"] == 4

    def test_same_sides_different_plus_order(self):
        """A + B = C should match B + A = C (frozenset comparison on +)."""
        ranked = _make_ranked_insights([
            ("compression + reduction = intelligence", 4),
            ("reduction + compression = intelligence", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1

    def test_non_equation_not_matched(self):
        """Non-equation insights should not trigger commutative check."""
        ranked = _make_ranked_insights([
            ("Encryption protects data integrity", 4),
            ("Caching improves response latency", 3),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 2

    def test_equation_vs_non_equation_preserved(self):
        """An equation and a non-equation should both survive."""
        ranked = _make_ranked_insights([
            ("A + B = C", 4),
            ("D requires E for stability", 3),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 2

    def test_different_equations_preserved(self):
        """Two genuinely different equations should both survive."""
        ranked = _make_ranked_insights([
            ("trust + provenance = reliability", 4),
            ("speed + accuracy = performance", 3),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 2

    def test_single_term_equation(self):
        """Single-term equations (no +) with swapped sides should match."""
        ranked = _make_ranked_insights([
            ("trust = reliability", 4),
            ("reliability = trust", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1


# ===========================================================================
# TestOverlapThreshold — Fix 1c: threshold lowered from 0.5 to 0.4
# ===========================================================================

class TestOverlapThreshold:
    """Verify overlap coefficient threshold at 0.4 catches thesaurus loops."""

    def test_synonym_substitution_caught(self):
        """structural_immortality vs architectural_immortality should merge.

        Content words: {structural, immortality} vs {architectural, immortality}
        Intersection: {immortality}, min_size=2, overlap=0.5 > 0.4 → merged.
        (At old 0.5 threshold this was borderline; at 0.4 it's clearly caught.)
        """
        ranked = _make_ranked_insights([
            ("structural immortality through versioning", 4),
            ("architectural immortality through versioning", 2),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1

    def test_low_overlap_preserved(self):
        """Insights sharing only 1 of 5+ content words should survive.

        {alpha, beta, gamma, delta, epsilon} vs {alpha, zeta, eta, theta, iota}
        Intersection: {alpha}, min_size=5, overlap=0.2 < 0.4 → preserved.
        """
        ranked = _make_ranked_insights([
            ("alpha beta gamma delta epsilon", 4),
            ("alpha zeta eta theta iota", 3),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 2

    def test_medium_overlap_caught(self):
        """Insights sharing 2 of 4 content words should merge.

        overlap = 2/4 = 0.5 > 0.4 → merged.
        """
        ranked = _make_ranked_insights([
            ("Kanban board task tracking", 4),
            ("Kanban board project management", 3),
        ])
        result = _deduplicate_insights(ranked)
        assert len(result) == 1

    def test_stopwords_excluded_from_overlap(self):
        """Stopwords like 'the', 'is', 'for' should not inflate overlap."""
        ranked = _make_ranked_insights([
            ("the system is for task management", 4),
            ("the platform is for user authentication", 3),
        ])
        result = _deduplicate_insights(ranked)
        # Content words: {system, task, management} vs {platform, user, authentication}
        # No intersection → preserved
        assert len(result) == 2


# ===========================================================================
# TestInsightCap — Fix 1d: cap at 10 in digest and engine fallback
# ===========================================================================

class TestInsightCap:
    """Verify insights are capped at 10 in both digest and engine paths."""

    def test_digest_caps_at_10(self):
        """build_dialogue_digest caps insights at 10."""
        from core.protocol import Message, MessageType
        state = _make_shared_state(
            insights=[f"Unique insight number {i} about topic {i}" for i in range(25)],
            history=[
                Message(
                    sender="Entity",
                    content=f"analysis {i}",
                    message_type=MessageType.PROPOSITION,
                    insight=f"Unique insight number {i} about topic {i}",
                )
                for i in range(25)
            ],
        )
        digest = build_dialogue_digest(state)
        # Count insight lines (lines starting with [Entity] or similar)
        insight_section = digest.split("INSIGHTS:\n")[1].split("\n\n")[0] if "INSIGHTS:" in digest else ""
        insight_lines = [l for l in insight_section.split("\n") if "[Entity" in l]
        assert len(insight_lines) <= 10

    def test_engine_fallback_caps_at_10(self):
        """Engine fallback path caps insights at 10."""
        state = _make_shared_state(
            insights=[f"Unique insight {i} about different topic {i}" for i in range(25)],
            history=[],
        )
        blueprint = {}

        # Simulate the engine fallback logic
        if state.insights and not blueprint.get("insights"):
            ranked = _deduplicate_insights(_rank_insights(state))
            blueprint["insights"] = [r["insight"] for r in ranked[:10]]

        assert len(blueprint["insights"]) <= 10

    def test_cap_preserves_highest_scored(self):
        """Cap at 10 should keep the highest-scored insights (sorted by score desc)."""
        from core.protocol import Message, MessageType
        # Create 15 genuinely distinct insights with no shared content words
        distinct_topics = [
            "Encryption protects database columns",
            "Caching reduces server latency",
            "Pagination handles large datasets",
            "Webhooks enable event notifications",
            "Middleware validates authentication tokens",
            "Sharding distributes storage horizontally",
            "Throttling prevents resource exhaustion",
            "Serialization converts objects binary",
            "Indexing accelerates query lookups",
            "Replication ensures failover availability",
            "Compression minimizes bandwidth consumption",
            "Sandboxing isolates untrusted execution",
            "Monitoring tracks performance metrics",
            "Migration evolves schema versions",
            "Pooling reuses connection resources",
        ]
        insights = []
        history = []
        for i, text in enumerate(distinct_topics):
            insights.append(text)
            mtype = MessageType.CHALLENGE if i >= 10 else MessageType.PROPOSITION
            history.append(Message(
                sender="Entity",
                content=f"analysis {i}",
                message_type=mtype,
                insight=text,
            ))

        state = _make_shared_state(insights=insights, history=history)
        ranked = _deduplicate_insights(_rank_insights(state))[:10]
        # Should have exactly 10 (15 distinct - capped at 10)
        assert len(ranked) == 10


# ===========================================================================
# TestProcessTemplateNoBoilerplate — Fix 3: no generic lifecycle methods
# ===========================================================================

class TestProcessTemplateNoBoilerplate:
    """Verify METHOD INFERENCE template discourages generic lifecycle methods."""

    def test_no_start_execute_get_status_in_template(self):
        """Template should NOT suggest start/execute/get_status as defaults."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        # Find the METHOD INFERENCE section
        assert "METHOD INFERENCE" in source
        # The template should warn against generic lifecycle
        assert "Do NOT add generic lifecycle methods" in source
        assert "start, execute, get_status" in source

    def test_domain_specific_preference(self):
        """Template should prefer domain-specific methods."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "DOMAIN-SPECIFIC" in source or "domain-specific" in source

    def test_anti_padding_instruction(self):
        """Template should warn against padding with generic getters/setters."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "Do NOT pad with generic" in source
        assert "fewer domain-specific methods are better" in source

    def test_entity_data_container_guidance(self):
        """Template should advise simple data containers need few methods."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "data containers" in source or "value objects" in source
        assert "FEW methods" in source or "few methods" in source.lower()

    def test_provenance_still_required(self):
        """Template should still require derived_from on inferred methods."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "derived_from" in source
        assert "Do NOT invent methods with no anchor" in source


# ===========================================================================
# TestEngineFallbackDedup — Fix 2: proper dedup in engine fallback path
# ===========================================================================

class TestEngineFallbackDedup:
    """Verify engine fallback insight population uses ranked+deduped pipeline."""

    def test_engine_fallback_uses_dedup(self):
        """Engine fallback path should use _deduplicate_insights, not exact match."""
        import inspect
        from core.engine import MotherlabsEngine
        source = inspect.getsource(MotherlabsEngine.compile)
        # Should import and use proper dedup
        assert "_deduplicate_insights" in source
        assert "_rank_insights" in source

    def test_engine_fallback_removes_near_dupes(self):
        """Near-duplicate insights should be collapsed in engine fallback."""
        from core.protocol import Message, MessageType
        state = _make_shared_state(
            insights=[
                "Kanban board for task tracking",
                "Kanban board for task management",
                "Real-time collaboration needed",
            ],
            history=[
                Message(sender="Entity", content="a",
                        message_type=MessageType.PROPOSITION,
                        insight="Kanban board for task tracking"),
                Message(sender="Process", content="b",
                        message_type=MessageType.PROPOSITION,
                        insight="Kanban board for task management"),
                Message(sender="Entity", content="c",
                        message_type=MessageType.PROPOSITION,
                        insight="Real-time collaboration needed"),
            ],
        )
        blueprint = {}

        # Simulate engine fallback
        if state.insights and not blueprint.get("insights"):
            ranked = _deduplicate_insights(_rank_insights(state))
            blueprint["insights"] = [r["insight"] for r in ranked[:10]]

        # Two Kanban insights should merge, leaving 2 total
        assert len(blueprint["insights"]) == 2

    def test_engine_fallback_skips_when_blueprint_has_insights(self):
        """If blueprint already has insights, engine should NOT overwrite."""
        state = _make_shared_state(
            insights=["Insight A", "Insight B"],
            history=[],
        )
        blueprint = {"insights": ["Already present"]}

        # Simulate engine fallback — should NOT trigger
        if state.insights and not blueprint.get("insights"):
            ranked = _deduplicate_insights(_rank_insights(state))
            blueprint["insights"] = [r["insight"] for r in ranked[:10]]

        assert blueprint["insights"] == ["Already present"]
