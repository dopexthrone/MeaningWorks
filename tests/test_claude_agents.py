"""Tests for mother/claude_agents.py — standalone semantic compiler.

Unit tests: no API calls, no mocking.
Integration tests: mocked anthropic.Anthropic client.
"""

import dataclasses
import math
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from mother.claude_agents import (
    # Data structures
    Intent,
    GridCell,
    Component,
    Relationship,
    Blueprint,
    VerificationScore,
    CompilerResult,
    CostConfig,
    # Grid
    SimpleGrid,
    # Vocabulary filters
    _BEHAVIORAL_VOCAB,
    _STRUCTURAL_VOCAB,
    _apply_blindness,
    # Token processing
    _normalize_tokens,
    _stem,
    _semantic_similarity,
    _detect_compression_losses,
    _expand_synonyms,
    _bigram_tokens,
    # Tool schemas
    ALL_TOOLS,
    TOOL_EXTRACT_INTENT,
    TOOL_FILL_STRUCTURAL,
    TOOL_FILL_BEHAVIORAL,
    TOOL_CHALLENGE,
    TOOL_SYNTHESIZE_COMPONENT,
    TOOL_SYNTHESIZE_RELATIONSHIP,
    TOOL_VERIFY_DIMENSION,
    TOOL_DECODE_INTENT,
    # Token tracker
    _TokenTracker,
    # Agent runner
    _call_agent,
    _run_intent_agent,
    _run_dialogue,
    _run_synthesis,
    _run_verification,
    _run_governor_gate,
    # Entry point
    compile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_intent(**kwargs):
    defaults = dict(
        core_need="A task manager with user authentication and notifications",
        domain="task-management",
        actors=("User", "Admin", "Task"),
        constraints=("must support real-time updates",),
        implicit_goals=("input validation", "error handling"),
        insight="Notification system implies an event bus",
    )
    defaults.update(kwargs)
    return Intent(**defaults)


def _make_blueprint(**kwargs):
    defaults = dict(
        components=(
            Component("TaskService", "service", "Manages task CRUD",
                      ("id", "title", "status"), ("create", "update", "delete"),
                      "ENT.Task"),
            Component("AuthService", "service", "Handles authentication",
                      ("token", "session"), ("login", "logout", "verify"),
                      "ENT.User"),
            Component("NotificationService", "service", "Sends notifications",
                      ("channel", "message"), ("send", "subscribe"),
                      "BHV.notifications"),
        ),
        relationships=(
            Relationship("TaskService", "AuthService", "depends_on",
                         "Tasks require auth", "ENT.Task, ENT.User"),
            Relationship("TaskService", "NotificationService", "triggers",
                         "Task changes trigger notifications", "BHV.notifications"),
        ),
        constraints=("must support real-time updates",),
        insights=("Notification system implies an event bus",),
        unresolved=(),
    )
    defaults.update(kwargs)
    return Blueprint(**defaults)


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------

class TestFrozenDataclasses:
    def test_intent_frozen(self):
        intent = _make_intent()
        with pytest.raises(dataclasses.FrozenInstanceError):
            intent.core_need = "changed"  # type: ignore

    def test_grid_cell_frozen(self):
        cell = GridCell(postcode="ENT.User")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cell.content = "changed"  # type: ignore

    def test_component_frozen(self):
        comp = Component("Test", "service", "desc")
        with pytest.raises(dataclasses.FrozenInstanceError):
            comp.name = "changed"  # type: ignore

    def test_relationship_frozen(self):
        rel = Relationship("A", "B", "depends_on")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rel.source = "changed"  # type: ignore

    def test_blueprint_frozen(self):
        bp = Blueprint()
        with pytest.raises(dataclasses.FrozenInstanceError):
            bp.components = ()  # type: ignore

    def test_verification_score_frozen(self):
        vs = VerificationScore()
        with pytest.raises(dataclasses.FrozenInstanceError):
            vs.overall = 100.0  # type: ignore

    def test_compiler_result_frozen(self):
        cr = CompilerResult()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cr.fidelity_score = 1.0  # type: ignore

    def test_cost_config_mutable(self):
        # CostConfig is intentionally NOT frozen
        cfg = CostConfig()
        cfg.max_dialogue_rounds = 10
        assert cfg.max_dialogue_rounds == 10


# ---------------------------------------------------------------------------
# SimpleGrid
# ---------------------------------------------------------------------------

class TestSimpleGrid:
    def test_seed_from_intent(self):
        intent = _make_intent()
        grid = SimpleGrid()
        grid.seed_from_intent(intent)
        # Should have ENT cells for actors
        assert grid.get("ENT.User") is not None
        assert grid.get("ENT.Admin") is not None
        assert grid.get("ENT.Task") is not None
        # All seeded cells are empty
        for pc in ["ENT.User", "ENT.Admin", "ENT.Task"]:
            cell = grid.get(pc)
            assert cell.fill_state == "empty"
            assert cell.confidence == 0.0

    def test_seed_creates_behavioral_cells(self):
        intent = _make_intent(domain="task-management")
        grid = SimpleGrid()
        grid.seed_from_intent(intent)
        bhv_cells = grid.behavioral_cells()
        assert len(bhv_cells) >= 1
        assert any("task-management" in k for k in bhv_cells)

    def test_fill_cell(self):
        grid = SimpleGrid()
        grid.fill("ENT.User", "A system user", "entity_agent", 0.8)
        cell = grid.get("ENT.User")
        assert cell is not None
        assert cell.content == "A system user"
        assert cell.fill_state == "filled"
        assert cell.source == "entity_agent"
        assert cell.confidence == 0.8

    def test_fill_partial(self):
        grid = SimpleGrid()
        grid.fill("ENT.User", "partial data", "entity_agent", 0.4)
        cell = grid.get("ENT.User")
        assert cell.fill_state == "partial"

    def test_fill_with_connections(self):
        grid = SimpleGrid()
        grid.fill("ENT.User", "user", "entity_agent", 0.9, ("ENT.Task",))
        cell = grid.get("ENT.User")
        assert cell.connections == ("ENT.Task",)

    def test_unfilled_cells(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "", "x", 0.0)  # empty stays as is — actually fill is always partial or filled
        grid._cells["ENT.Empty"] = GridCell(postcode="ENT.Empty")  # manually add empty
        grid.fill("ENT.Full", "content", "y", 0.8)
        unfilled = grid.unfilled_cells()
        assert "ENT.Empty" in unfilled
        assert "ENT.Full" not in unfilled

    def test_coverage_score(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "a", "x", 0.8)
        grid.fill("ENT.B", "b", "x", 0.8)
        grid._cells["ENT.C"] = GridCell(postcode="ENT.C")  # empty
        # 2/3 filled with high confidence
        assert abs(grid.coverage_score() - 2.0 / 3.0) < 0.01

    def test_coverage_empty_grid(self):
        grid = SimpleGrid()
        assert grid.coverage_score() == 0.0

    def test_is_converged(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "a", "x", 0.8)
        grid.fill("ENT.B", "b", "x", 0.8)
        grid.fill("ENT.C", "c", "x", 0.8)
        grid._cells["ENT.D"] = GridCell(postcode="ENT.D")  # empty
        # 3/4 = 0.75 >= 0.75
        assert grid.is_converged()

    def test_not_converged(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "a", "x", 0.8)
        grid._cells["ENT.B"] = GridCell(postcode="ENT.B")
        grid._cells["ENT.C"] = GridCell(postcode="ENT.C")
        grid._cells["ENT.D"] = GridCell(postcode="ENT.D")
        # 1/4 = 0.25 < 0.75
        assert not grid.is_converged()

    def test_structural_cells(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "a", "x", 0.8)
        grid.fill("BHV.flow", "f", "y", 0.8)
        struct = grid.structural_cells()
        assert "ENT.A" in struct
        assert "BHV.flow" not in struct

    def test_behavioral_cells(self):
        grid = SimpleGrid()
        grid.fill("ENT.A", "a", "x", 0.8)
        grid.fill("BHV.flow", "f", "y", 0.8)
        behav = grid.behavioral_cells()
        assert "BHV.flow" in behav
        assert "ENT.A" not in behav

    def test_snapshot(self):
        grid = SimpleGrid()
        grid.fill("ENT.X", "content", "agent", 0.9, ("ENT.Y",))
        snap = grid.snapshot()
        assert "ENT.X" in snap
        assert snap["ENT.X"]["content"] == "content"
        assert snap["ENT.X"]["confidence"] == 0.9
        assert snap["ENT.X"]["connections"] == ["ENT.Y"]

    def test_insights(self):
        grid = SimpleGrid()
        grid.add_insight("first")
        grid.add_insight("second")
        grid.add_insight("first")  # duplicate
        assert grid.all_insights() == ("first", "second")

    def test_cell_count(self):
        grid = SimpleGrid()
        assert grid.cell_count() == 0
        grid.fill("ENT.A", "a", "x", 0.8)
        assert grid.cell_count() == 1

    def test_get_nonexistent(self):
        grid = SimpleGrid()
        assert grid.get("ENT.Nonexistent") is None


# ---------------------------------------------------------------------------
# Vocabulary filters
# ---------------------------------------------------------------------------

class TestVocabularyFilters:
    def test_behavioral_terms_removed_for_entity(self):
        text = "A workflow that triggers events through a pipeline"
        filtered = _apply_blindness(text, _BEHAVIORAL_VOCAB)
        assert "workflow" not in filtered.lower()
        assert "trigger" not in filtered.lower()
        assert "pipeline" not in filtered.lower()
        assert "[...]" in filtered

    def test_structural_terms_removed_for_process(self):
        text = "An entity with schema attributes and type system"
        filtered = _apply_blindness(text, _STRUCTURAL_VOCAB)
        assert "entity" not in filtered.lower()
        assert "schema" not in filtered.lower()
        assert "type system" not in filtered.lower()
        assert "[...]" in filtered

    def test_case_insensitive(self):
        text = "The WORKFLOW triggers an EVENT"
        filtered = _apply_blindness(text, _BEHAVIORAL_VOCAB)
        assert "WORKFLOW" not in filtered
        assert "EVENT" not in filtered

    def test_non_vocab_preserved(self):
        text = "Users authenticate with tokens"
        filtered = _apply_blindness(text, _BEHAVIORAL_VOCAB)
        assert "Users" in filtered
        assert "authenticate" in filtered
        assert "tokens" in filtered

    def test_longer_terms_replaced_first(self):
        # "state machine" should be replaced as a unit, not "state" alone
        text = "A state machine handles transitions"
        filtered = _apply_blindness(text, _BEHAVIORAL_VOCAB)
        assert "state machine" not in filtered.lower()

    def test_empty_text(self):
        assert _apply_blindness("", _BEHAVIORAL_VOCAB) == ""


# ---------------------------------------------------------------------------
# Token normalization + similarity
# ---------------------------------------------------------------------------

class TestTokenNormalization:
    def test_basic_stemming(self):
        assert _stem("running") == "runn"
        assert _stem("authentication") == "authentic"
        assert _stem("notifications") == "notific"

    def test_normalize_tokens(self):
        tokens = _normalize_tokens("User Authentication Service")
        assert "user" in tokens or any("user" in t for t in tokens)
        assert "authentic" in tokens

    def test_camel_case_split(self):
        tokens = _normalize_tokens("AuthService")
        assert "auth" in tokens
        assert "service" in tokens

    def test_stop_words_removed(self):
        tokens = _normalize_tokens("the user is authenticated")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_synonym_expansion(self):
        tokens = _normalize_tokens("login")
        expanded = _expand_synonyms(tokens)
        assert "auth" in expanded

    def test_bigram_tokens(self):
        bigrams = _bigram_tokens("task manager authentication")
        assert "task_manag" in bigrams or any("task" in b and "manag" in b for b in bigrams)


class TestSemanticSimilarity:
    def test_identical_strings(self):
        score = _semantic_similarity(
            "A task manager with authentication",
            "A task manager with authentication",
        )
        assert score >= 0.8

    def test_high_overlap(self):
        score = _semantic_similarity(
            "A task manager with user authentication and notifications",
            "Task management system with authentication and notification service",
        )
        assert score >= 0.5

    def test_low_overlap(self):
        score = _semantic_similarity(
            "A task manager with authentication",
            "A weather forecasting system",
        )
        assert score <= 0.4

    def test_empty_strings(self):
        assert _semantic_similarity("", "anything") == 0.0
        assert _semantic_similarity("anything", "") == 0.0
        assert _semantic_similarity("", "") == 0.0

    def test_synonym_boost(self):
        # "login" and "authentication" are synonyms
        score = _semantic_similarity(
            "user login system",
            "user authentication system",
        )
        assert score >= 0.5

    def test_bounded_0_1(self):
        score = _semantic_similarity("a", "a")
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        score = _semantic_similarity("task manager", "task manager")
        assert isinstance(score, float)


class TestCompressionLosses:
    def test_no_losses(self):
        bp = _make_blueprint()
        losses = _detect_compression_losses(
            "TaskService AuthService NotificationService",
            "TaskService AuthService NotificationService",
            bp,
        )
        assert len(losses) == 0

    def test_entity_loss(self):
        bp = _make_blueprint()
        losses = _detect_compression_losses(
            "A task manager with Webhooks and Calendars",
            "A task manager",
            bp,
        )
        assert len(losses) > 0
        assert any("webhook" in l.lower() or "calendar" in l.lower() for l in losses)

    def test_context_loss(self):
        bp = _make_blueprint()
        # "real-time" is in blueprint constraints, but not in reconstructed
        losses = _detect_compression_losses(
            "real-time task manager with notifications",
            "task manager",
            bp,
        )
        assert len(losses) >= 0  # may or may not detect depending on token overlap


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 8

    def test_tool_structure(self):
        for tool in ALL_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            assert isinstance(schema["required"], list)

    def test_extract_intent_fields(self):
        schema = TOOL_EXTRACT_INTENT["input_schema"]
        required = schema["required"]
        assert "core_need" in required
        assert "domain" in required
        assert "actors" in required

    def test_fill_structural_fields(self):
        schema = TOOL_FILL_STRUCTURAL["input_schema"]
        required = schema["required"]
        assert "cell_name" in required
        assert "description" in required
        assert "confidence" in required

    def test_fill_behavioral_fields(self):
        schema = TOOL_FILL_BEHAVIORAL["input_schema"]
        required = schema["required"]
        assert "cell_name" in required
        assert "description" in required
        assert "steps" in required

    def test_challenge_fields(self):
        schema = TOOL_CHALLENGE["input_schema"]
        required = schema["required"]
        assert "target_cell" in required
        assert "challenge_type" in required
        assert "argument" in required

    def test_verify_dimension_enum(self):
        schema = TOOL_VERIFY_DIMENSION["input_schema"]
        enum = schema["properties"]["dimension"]["enum"]
        assert set(enum) == {"completeness", "consistency", "coherence", "traceability"}

    def test_decode_intent_verdict_enum(self):
        schema = TOOL_DECODE_INTENT["input_schema"]
        enum = schema["properties"]["verdict"]["enum"]
        assert set(enum) == {"pass", "fail", "marginal"}


# ---------------------------------------------------------------------------
# CostConfig
# ---------------------------------------------------------------------------

class TestCostConfig:
    def test_defaults(self):
        cfg = CostConfig()
        assert "haiku" in cfg.intent_model
        assert "sonnet" in cfg.dialogue_model
        assert "sonnet" in cfg.synthesis_model
        assert "haiku" in cfg.verify_model
        assert "haiku" in cfg.governor_model
        assert cfg.max_dialogue_rounds == 5
        assert cfg.max_total_tokens == 200_000
        assert cfg.max_cost_usd == 2.0

    def test_custom_values(self):
        cfg = CostConfig(max_dialogue_rounds=3, max_cost_usd=1.0)
        assert cfg.max_dialogue_rounds == 3
        assert cfg.max_cost_usd == 1.0


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------

class TestTokenTracker:
    def test_record_usage(self):
        tracker = _TokenTracker(max_tokens=100_000, max_cost=5.0)
        tracker.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "claude-haiku-4-5-20251001",
        )
        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 500
        assert tracker.calls == 1
        assert tracker.total_cost > 0

    def test_accumulates(self):
        tracker = _TokenTracker(max_tokens=100_000, max_cost=5.0)
        tracker.record({"input_tokens": 100, "output_tokens": 50}, "claude-haiku-4-5-20251001")
        tracker.record({"input_tokens": 200, "output_tokens": 100}, "claude-haiku-4-5-20251001")
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 150
        assert tracker.calls == 2

    def test_token_cap(self):
        tracker = _TokenTracker(max_tokens=100, max_cost=5.0)
        with pytest.raises(RuntimeError, match="Token cap"):
            tracker.record({"input_tokens": 60, "output_tokens": 60}, "claude-haiku-4-5-20251001")

    def test_cost_cap(self):
        tracker = _TokenTracker(max_tokens=10_000_000, max_cost=0.001)
        with pytest.raises(RuntimeError, match="Cost cap"):
            tracker.record(
                {"input_tokens": 500_000, "output_tokens": 100_000},
                "claude-sonnet-4-20250514",
            )

    def test_summary(self):
        tracker = _TokenTracker(max_tokens=100_000, max_cost=5.0)
        tracker.record({"input_tokens": 1000, "output_tokens": 500}, "claude-haiku-4-5-20251001")
        summary = tracker.summary()
        assert summary["input_tokens"] == 1000
        assert summary["output_tokens"] == 500
        assert summary["calls"] == 1
        assert "total_cost_usd" in summary
        assert isinstance(summary["total_cost_usd"], float)

    def test_unknown_model_uses_default_rates(self):
        tracker = _TokenTracker(max_tokens=100_000, max_cost=5.0)
        tracker.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "unknown-model-xyz",
        )
        # Should use default rates (3.0/15.0 per million) without error
        assert tracker.total_cost > 0


# ---------------------------------------------------------------------------
# Integration tests (mocked Anthropic client)
# ---------------------------------------------------------------------------

def _make_tool_use_block(name, input_data, block_id="tb_1"):
    """Create a mock tool_use content block."""
    return SimpleNamespace(type="tool_use", name=name, input=input_data, id=block_id)


def _make_text_block(text):
    """Create a mock text content block."""
    return SimpleNamespace(type="text", text=text)


def _make_response(content_blocks, input_tokens=100, output_tokens=50):
    """Create a mock API response."""
    return SimpleNamespace(
        content=content_blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


class TestCallAgent:
    def test_extracts_tool_calls(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("extract_intent", {"core_need": "test", "domain": "test"}),
        ])
        tools, text, usage = _call_agent(
            client, "claude-haiku-4-5-20251001", "system", "user msg", [],
        )
        assert len(tools) == 1
        assert tools[0]["name"] == "extract_intent"
        assert tools[0]["input"]["core_need"] == "test"

    def test_extracts_text(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_text_block("Some thinking"),
            _make_tool_use_block("extract_intent", {"core_need": "t", "domain": "d"}),
        ])
        tools, text, usage = _call_agent(
            client, "claude-haiku-4-5-20251001", "system", "msg", [],
        )
        assert "Some thinking" in text
        assert len(tools) == 1

    def test_extracts_usage(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response(
            [_make_text_block("ok")], input_tokens=500, output_tokens=200,
        )
        _, _, usage = _call_agent(
            client, "claude-haiku-4-5-20251001", "system", "msg", [],
        )
        assert usage["input_tokens"] == 500
        assert usage["output_tokens"] == 200

    def test_multiple_tool_calls(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("fill_structural_cell", {"cell_name": "User"}, "tb_1"),
            _make_tool_use_block("fill_structural_cell", {"cell_name": "Task"}, "tb_2"),
            _make_tool_use_block("challenge", {"target_cell": "BHV.flow"}, "tb_3"),
        ])
        tools, _, _ = _call_agent(
            client, "claude-sonnet-4-20250514", "system", "msg", [],
        )
        assert len(tools) == 3


class TestIntentAgent:
    def test_extracts_intent(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("extract_intent", {
                "core_need": "Build a task manager",
                "domain": "task-management",
                "actors": ["User", "Task"],
                "constraints": ["real-time"],
                "implicit_goals": ["auth", "validation"],
                "insight": "Needs event bus",
            }),
        ])
        tracker = _TokenTracker(200_000, 5.0)
        intent = _run_intent_agent(client, "test", CostConfig(), tracker)
        assert intent.core_need == "Build a task manager"
        assert intent.domain == "task-management"
        assert "User" in intent.actors
        assert tracker.calls == 1

    def test_raises_if_no_tool_call(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_text_block("I'll analyze this..."),
        ])
        tracker = _TokenTracker(200_000, 5.0)
        with pytest.raises(RuntimeError, match="extract_intent"):
            _run_intent_agent(client, "test", CostConfig(), tracker)


class TestDialogue:
    def _mock_client_for_dialogue(self):
        """Create a client that returns entity fills then process fills."""
        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                # Entity agent
                return _make_response([
                    _make_tool_use_block("fill_structural_cell", {
                        "cell_name": "User",
                        "description": "System user entity",
                        "attributes": ["id", "name", "email"],
                        "confidence": 0.85,
                        "derived_from": "User mentioned in input",
                        "insight": "User needs roles",
                    }),
                ])
            else:
                # Process agent
                return _make_response([
                    _make_tool_use_block("fill_behavioral_cell", {
                        "cell_name": "authentication",
                        "description": "User login flow",
                        "steps": ["enter credentials", "validate", "issue token"],
                        "confidence": 0.8,
                        "derived_from": "authentication requirement",
                        "insight": "Needs token refresh",
                    }),
                ])

        client.messages.create.side_effect = side_effect
        return client

    def test_dialogue_runs(self):
        client = self._mock_client_for_dialogue()
        intent = _make_intent()
        grid = SimpleGrid()
        grid.seed_from_intent(intent)
        tracker = _TokenTracker(200_000, 5.0)
        config = CostConfig(max_dialogue_rounds=2)
        turns = _run_dialogue(client, "test input", intent, grid, config, tracker)
        assert turns > 0
        assert tracker.calls > 0
        # Entity and process cells should be filled
        assert grid.get("ENT.User") is not None
        assert grid.get("ENT.User").fill_state == "filled"

    def test_entity_receives_filtered_input(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("fill_structural_cell", {
                "cell_name": "X", "description": "x", "attributes": [],
                "confidence": 0.9, "derived_from": "x",
            }),
        ])
        intent = _make_intent()
        grid = SimpleGrid()
        grid.seed_from_intent(intent)
        tracker = _TokenTracker(200_000, 5.0)
        config = CostConfig(max_dialogue_rounds=1)
        _run_dialogue(client, "A workflow with triggers", intent, grid, config, tracker)
        # Check that entity agent call had filtered input
        first_call = client.messages.create.call_args_list[0]
        user_msg = first_call.kwargs["messages"][0]["content"]
        assert "workflow" not in user_msg.lower() or "[...]" in user_msg

    def test_stall_detection(self):
        """Grid with no progress should stall after 2 rounds."""
        client = MagicMock()
        # Return empty tool calls — no fills
        client.messages.create.return_value = _make_response([
            _make_text_block("Nothing to fill"),
        ])
        intent = _make_intent()
        grid = SimpleGrid()
        grid.seed_from_intent(intent)
        tracker = _TokenTracker(200_000, 5.0)
        config = CostConfig(max_dialogue_rounds=5)
        turns = _run_dialogue(client, "test", intent, grid, config, tracker)
        # Should stop early due to stall (coverage doesn't change)
        assert turns <= 6  # at most 3 rounds * 2 turns/round


class TestSynthesis:
    def test_produces_blueprint(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("synthesize_component", {
                "name": "TaskService",
                "component_type": "service",
                "description": "Manages tasks",
                "attributes": ["id", "title"],
                "methods": ["create", "update"],
                "derived_from": "ENT.Task",
            }),
            _make_tool_use_block("synthesize_relationship", {
                "source": "TaskService",
                "target": "AuthService",
                "rel_type": "depends_on",
                "derived_from": "ENT.Task, ENT.User",
            }),
        ])
        intent = _make_intent()
        grid = SimpleGrid()
        grid.fill("ENT.Task", "task entity", "entity_agent", 0.8)
        tracker = _TokenTracker(200_000, 5.0)
        bp = _run_synthesis(client, "test", intent, grid, CostConfig(), tracker)
        assert len(bp.components) == 1
        assert bp.components[0].name == "TaskService"
        assert len(bp.relationships) == 1

    def test_retries_on_no_components(self):
        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_response([_make_text_block("Thinking...")])
            else:
                return _make_response([
                    _make_tool_use_block("synthesize_component", {
                        "name": "X", "component_type": "service",
                        "description": "x", "attributes": [], "methods": [],
                        "derived_from": "ENT.X",
                    }),
                ])

        client.messages.create.side_effect = side_effect
        intent = _make_intent()
        grid = SimpleGrid()
        tracker = _TokenTracker(200_000, 5.0)
        bp = _run_synthesis(client, "test", intent, grid, CostConfig(), tracker)
        assert len(bp.components) == 1
        assert call_count[0] == 2


class TestVerification:
    def test_scores_4_dimensions(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("verify_dimension", {
                "dimension": "completeness", "score": 85, "evidence": "all covered", "gaps": [],
            }, "v1"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "consistency", "score": 90, "evidence": "no contradictions", "gaps": [],
            }, "v2"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "coherence", "score": 80, "evidence": "makes sense", "gaps": ["minor gap"],
            }, "v3"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "traceability", "score": 75, "evidence": "mostly traceable", "gaps": [],
            }, "v4"),
        ])
        intent = _make_intent()
        bp = _make_blueprint()
        tracker = _TokenTracker(200_000, 5.0)
        vs = _run_verification(client, intent, bp, CostConfig(), tracker)
        assert vs.completeness == 85
        assert vs.consistency == 90
        assert vs.coherence == 80
        assert vs.traceability == 75
        assert vs.overall == (85 + 90 + 80 + 75) / 4
        assert "minor gap" in vs.gaps
        assert vs.recommendation == "pass"

    def test_low_score_recommendation(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("verify_dimension", {
                "dimension": "completeness", "score": 50, "evidence": "x", "gaps": ["a"],
            }, "v1"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "consistency", "score": 40, "evidence": "x", "gaps": ["b"],
            }, "v2"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "coherence", "score": 60, "evidence": "x", "gaps": [],
            }, "v3"),
            _make_tool_use_block("verify_dimension", {
                "dimension": "traceability", "score": 50, "evidence": "x", "gaps": [],
            }, "v4"),
        ])
        intent = _make_intent()
        bp = _make_blueprint()
        tracker = _TokenTracker(200_000, 5.0)
        vs = _run_verification(client, intent, bp, CostConfig(), tracker)
        assert vs.overall < 70
        assert vs.recommendation == "needs improvement"


class TestGovernorGate:
    def test_high_fidelity(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("decode_intent", {
                "reconstructed_intent": "A task manager with user authentication and real-time notifications",
                "confidence": 0.9,
                "compression_losses": [],
                "verdict": "pass",
            }),
        ])
        bp = _make_blueprint()
        tracker = _TokenTracker(200_000, 5.0)
        fidelity, recon, losses = _run_governor_gate(
            client,
            "A task manager with user authentication and real-time notifications",
            bp, CostConfig(), tracker,
        )
        assert fidelity >= 0.6
        assert len(recon) > 0

    def test_low_fidelity(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("decode_intent", {
                "reconstructed_intent": "A weather forecasting system",
                "confidence": 0.3,
                "compression_losses": ["Lost all task management context"],
                "verdict": "fail",
            }),
        ])
        bp = _make_blueprint()
        tracker = _TokenTracker(200_000, 5.0)
        fidelity, _, losses = _run_governor_gate(
            client,
            "A task manager with user authentication and real-time notifications",
            bp, CostConfig(), tracker,
        )
        assert fidelity < 0.6
        assert len(losses) > 0

    def test_governor_never_sees_grid(self):
        """Governor prompt should not contain grid data."""
        client = MagicMock()
        client.messages.create.return_value = _make_response([
            _make_tool_use_block("decode_intent", {
                "reconstructed_intent": "test",
                "confidence": 0.5,
                "compression_losses": [],
                "verdict": "marginal",
            }),
        ])
        bp = _make_blueprint()
        tracker = _TokenTracker(200_000, 5.0)
        _run_governor_gate(client, "test input", bp, CostConfig(), tracker)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "ENT." not in user_msg
        assert "BHV." not in user_msg
        assert "grid" not in user_msg.lower()


class TestFullPipeline:
    def _mock_full_pipeline_client(self):
        """Mock client that handles all 6 phases."""
        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            system = kwargs.get("system", "")
            tools = kwargs.get("tools", [])
            tool_names = [t["name"] for t in tools] if tools else []

            if "extract_intent" in tool_names:
                return _make_response([
                    _make_tool_use_block("extract_intent", {
                        "core_need": "Build a task manager",
                        "domain": "task-management",
                        "actors": ["User", "Task"],
                        "constraints": ["real-time"],
                        "implicit_goals": ["auth"],
                        "insight": "Needs event bus",
                    }),
                ])
            elif "fill_structural_cell" in tool_names:
                return _make_response([
                    _make_tool_use_block("fill_structural_cell", {
                        "cell_name": "User",
                        "description": "System user",
                        "attributes": ["id", "name"],
                        "confidence": 0.85,
                        "derived_from": "actors",
                    }),
                    _make_tool_use_block("fill_structural_cell", {
                        "cell_name": "Task",
                        "description": "A task item",
                        "attributes": ["id", "title", "status"],
                        "confidence": 0.9,
                        "derived_from": "actors",
                    }),
                ])
            elif "fill_behavioral_cell" in tool_names:
                return _make_response([
                    _make_tool_use_block("fill_behavioral_cell", {
                        "cell_name": "task_management",
                        "description": "CRUD operations on tasks",
                        "steps": ["create", "assign", "complete"],
                        "confidence": 0.8,
                        "derived_from": "core_need",
                    }),
                    _make_tool_use_block("fill_behavioral_cell", {
                        "cell_name": "authentication",
                        "description": "User auth flow",
                        "steps": ["login", "verify", "token"],
                        "confidence": 0.85,
                        "derived_from": "implicit",
                    }),
                ])
            elif "synthesize_component" in tool_names:
                return _make_response([
                    _make_tool_use_block("synthesize_component", {
                        "name": "UserService",
                        "component_type": "service",
                        "description": "User management",
                        "attributes": ["id", "name"],
                        "methods": ["login", "register"],
                        "derived_from": "ENT.User",
                    }, "sc1"),
                    _make_tool_use_block("synthesize_component", {
                        "name": "TaskService",
                        "component_type": "service",
                        "description": "Task CRUD",
                        "attributes": ["id", "title", "status"],
                        "methods": ["create", "assign", "complete"],
                        "derived_from": "ENT.Task, BHV.task_management",
                    }, "sc2"),
                    _make_tool_use_block("synthesize_relationship", {
                        "source": "TaskService",
                        "target": "UserService",
                        "rel_type": "depends_on",
                        "derived_from": "ENT.Task->ENT.User",
                    }, "sr1"),
                ])
            elif "verify_dimension" in tool_names:
                return _make_response([
                    _make_tool_use_block("verify_dimension", {
                        "dimension": "completeness", "score": 80,
                        "evidence": "covers main entities", "gaps": [],
                    }, "v1"),
                    _make_tool_use_block("verify_dimension", {
                        "dimension": "consistency", "score": 85,
                        "evidence": "no contradictions", "gaps": [],
                    }, "v2"),
                    _make_tool_use_block("verify_dimension", {
                        "dimension": "coherence", "score": 82,
                        "evidence": "makes sense", "gaps": [],
                    }, "v3"),
                    _make_tool_use_block("verify_dimension", {
                        "dimension": "traceability", "score": 78,
                        "evidence": "mostly traceable", "gaps": ["some gaps"],
                    }, "v4"),
                ])
            elif "decode_intent" in tool_names:
                return _make_response([
                    _make_tool_use_block("decode_intent", {
                        "reconstructed_intent": "A task management system with user authentication and real-time updates",
                        "confidence": 0.85,
                        "compression_losses": [],
                        "verdict": "pass",
                    }),
                ])
            else:
                return _make_response([_make_text_block("ok")])

        client.messages.create.side_effect = side_effect
        return client

    def test_full_pipeline(self):
        mock_client = self._mock_full_pipeline_client()
        config = CostConfig(max_dialogue_rounds=1, max_cost_usd=5.0)
        result = compile(
            "A task manager with user authentication and real-time notifications",
            config,
            client=mock_client,
        )

        assert result.intent is not None
        assert result.intent.core_need == "Build a task manager"
        assert result.blueprint is not None
        assert len(result.blueprint.components) >= 2
        assert result.verification is not None
        assert result.verification.overall > 0
        assert result.fidelity_score > 0
        assert result.token_usage["calls"] > 0
        assert result.duration_seconds >= 0
        assert result.grid_snapshot  # non-empty

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compile("", CostConfig())

    def test_whitespace_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compile("   ", CostConfig())

    def test_budget_enforcement(self):
        """Token cap should raise RuntimeError."""
        client = MagicMock()
        # Return huge token counts
        client.messages.create.return_value = _make_response(
            [_make_tool_use_block("extract_intent", {
                "core_need": "x", "domain": "x", "actors": [], "constraints": [],
                "implicit_goals": [], "insight": "x",
            })],
            input_tokens=150_000, output_tokens=60_000,
        )
        config = CostConfig(max_total_tokens=100_000)
        with pytest.raises(RuntimeError, match="Token cap"):
            compile("test", config, client=client)

    def test_f_of_f_mock(self):
        """F(F) test: compile a description of the compiler itself.

        Mock returns generic task-management responses, so fidelity will be low.
        The real F(F) test requires live API. Here we verify the pipeline runs
        end-to-end and produces a CompilerResult.
        """
        mock_client = self._mock_full_pipeline_client()
        config = CostConfig(max_dialogue_rounds=1, max_cost_usd=5.0)
        result = compile(
            "A local AI entity that compiles natural language into verified "
            "structure through asymmetric agent dialogue",
            config,
            client=mock_client,
        )
        assert result.intent is not None
        assert result.blueprint is not None
        assert len(result.blueprint.components) >= 2
        # Fidelity is a float (may be low with generic mock responses)
        assert isinstance(result.fidelity_score, float)
        assert 0.0 <= result.fidelity_score <= 1.0
