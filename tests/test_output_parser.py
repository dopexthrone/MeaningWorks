"""
Phase 22: Tests for core/output_parser.py (LEAF MODULE) and engine integration.

Tests:
- ParseResult, FieldSpec, StageSchema frozen dataclasses
- extract_json() 3 strategies + array wrapping
- validate_against_schema() required/optional/type/min_length
- parse_structured_output() orchestrator
- build_repair_prompt() error formatting
- LEAF MODULE constraint (no project imports)
- Engine _parse_structured_output() integration + repair retry
"""

import ast
import json
import pytest
from unittest.mock import Mock, patch

from core.output_parser import (
    ParseResult,
    FieldSpec,
    StageSchema,
    STAGE_SCHEMAS,
    extract_json,
    validate_against_schema,
    parse_structured_output,
    build_repair_prompt,
)


# =============================================================================
# 1. FROZEN DATACLASS TESTS
# =============================================================================


class TestFrozenDataclasses:
    """Verify frozen dataclasses are immutable."""

    def test_parse_result_frozen(self):
        pr = ParseResult(success=True, data={"a": 1}, errors=(), raw_text="x")
        with pytest.raises(AttributeError):
            pr.success = False

    def test_field_spec_defaults(self):
        fs = FieldSpec(name="test")
        assert fs.required is True
        assert fs.expected_type == "any"
        assert fs.min_length == 0

    def test_stage_schema_frozen(self):
        ss = StageSchema(name="test", fields=())
        with pytest.raises(AttributeError):
            ss.name = "other"

    def test_stage_schema_allows_array_default(self):
        ss = StageSchema(name="test", fields=())
        assert ss.allows_array is False


# =============================================================================
# 2. extract_json() TESTS
# =============================================================================


class TestExtractJson:
    """Test the 3 JSON extraction strategies + array wrapping."""

    def test_extract_json_direct(self):
        data = {"key": "value", "num": 42}
        result = extract_json(json.dumps(data))
        assert result == data

    def test_extract_json_code_block(self):
        text = 'Here is the result:\n```json\n{"core_need": "auth"}\n```\nDone.'
        result = extract_json(text)
        assert result["core_need"] == "auth"

    def test_extract_json_brace_extraction(self):
        text = 'Preamble text\n{"components": [], "relationships": []}\nTrailing text'
        result = extract_json(text)
        assert "components" in result

    def test_extract_json_array_wrapping(self):
        text = '[{"name": "Alice"}, {"name": "Bob"}]'
        result = extract_json(text)
        assert "_array" in result
        assert len(result["_array"]) == 2

    def test_extract_json_array_in_code_block(self):
        text = '```json\n[{"name": "A"}]\n```'
        result = extract_json(text)
        assert "_array" in result
        assert result["_array"][0]["name"] == "A"

    def test_extract_json_array_brace_extraction(self):
        """Strategy 3b: array extraction when no dict braces present."""
        text = 'Preamble [1, 2, 3] trailing'
        result = extract_json(text)
        assert "_array" in result
        assert result["_array"] == [1, 2, 3]

    def test_extract_json_failure(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json("This is not JSON at all")

    def test_extract_json_empty_string(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_extract_json_whitespace_handling(self):
        text = '  \n  {"key": "value"}  \n  '
        result = extract_json(text)
        assert result["key"] == "value"


# =============================================================================
# 3. validate_against_schema() TESTS
# =============================================================================


class TestValidateAgainstSchema:
    """Test schema validation: required, type, min_length."""

    def test_validate_required_field_present(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="foo", required=True, expected_type="str"),
        ))
        errors = validate_against_schema({"foo": "bar"}, schema)
        assert errors == ()

    def test_validate_required_field_missing(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="foo", required=True, expected_type="str"),
        ))
        errors = validate_against_schema({}, schema)
        assert len(errors) == 1
        assert "Missing required field" in errors[0]
        assert "'foo'" in errors[0]

    def test_validate_optional_field_missing_ok(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="foo", required=False, expected_type="str"),
        ))
        errors = validate_against_schema({}, schema)
        assert errors == ()

    def test_validate_type_str(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="x", required=True, expected_type="str"),
        ))
        assert validate_against_schema({"x": "hello"}, schema) == ()
        errors = validate_against_schema({"x": 42}, schema)
        assert len(errors) == 1
        assert "expected type 'str'" in errors[0]

    def test_validate_type_list(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="items", required=True, expected_type="list"),
        ))
        assert validate_against_schema({"items": [1, 2]}, schema) == ()
        errors = validate_against_schema({"items": "not a list"}, schema)
        assert len(errors) == 1
        assert "expected type 'list'" in errors[0]

    def test_validate_type_number(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="n", required=True, expected_type="number"),
        ))
        assert validate_against_schema({"n": 42}, schema) == ()
        assert validate_against_schema({"n": 3.14}, schema) == ()
        # bool is not number
        errors = validate_against_schema({"n": True}, schema)
        assert len(errors) == 1

    def test_validate_min_length_str(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="s", required=True, expected_type="str", min_length=3),
        ))
        assert validate_against_schema({"s": "abc"}, schema) == ()
        errors = validate_against_schema({"s": "ab"}, schema)
        assert len(errors) == 1
        assert "too short" in errors[0]

    def test_validate_min_length_list(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="items", required=True, expected_type="list", min_length=2),
        ))
        assert validate_against_schema({"items": [1, 2]}, schema) == ()
        errors = validate_against_schema({"items": [1]}, schema)
        assert len(errors) == 1
        assert "too short" in errors[0]

    def test_validate_multiple_errors(self):
        schema = StageSchema(name="test", fields=(
            FieldSpec(name="a", required=True, expected_type="str"),
            FieldSpec(name="b", required=True, expected_type="list", min_length=1),
        ))
        errors = validate_against_schema({"a": 42, "b": []}, schema)
        assert len(errors) == 2


# =============================================================================
# 4. parse_structured_output() TESTS
# =============================================================================


class TestParseStructuredOutput:
    """Test the orchestrator function."""

    def test_parse_intent_valid(self):
        text = json.dumps({
            "core_need": "Build an auth system",
            "domain": "authentication",
            "actors": ["User"],
        })
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.success is True
        assert result.data["core_need"] == "Build an auth system"
        assert result.errors == ()

    def test_parse_intent_missing_core_need(self):
        text = json.dumps({"domain": "auth", "actors": []})
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.success is False
        assert any("core_need" in e for e in result.errors)

    def test_parse_intent_short_core_need(self):
        text = json.dumps({"core_need": "ab"})
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.success is False
        assert any("too short" in e for e in result.errors)

    def test_parse_synthesis_valid(self):
        text = json.dumps({
            "components": [{"name": "User", "type": "entity"}],
            "relationships": [],
        })
        result = parse_structured_output(text, STAGE_SCHEMAS["synthesis"])
        assert result.success is True

    def test_parse_synthesis_no_components(self):
        text = json.dumps({"components": [], "relationships": []})
        result = parse_structured_output(text, STAGE_SCHEMAS["synthesis"])
        assert result.success is False
        assert any("too short" in e for e in result.errors)

    def test_parse_verification_valid(self):
        text = json.dumps({
            "status": "pass",
            "completeness": 85,
            "consistency": 90,
        })
        result = parse_structured_output(text, STAGE_SCHEMAS["verification"])
        assert result.success is True

    def test_parse_personas_array_input(self):
        """Bare array input gets wrapped for personas schema."""
        text = json.dumps([
            {"name": "Architect", "perspective": "structure", "blind_spots": "none"},
            {"name": "Designer", "perspective": "UX", "blind_spots": "security"},
        ])
        result = parse_structured_output(text, STAGE_SCHEMAS["personas"])
        assert result.success is True
        assert "personas" in result.data
        assert len(result.data["personas"]) == 2

    def test_parse_personas_object_input(self):
        text = json.dumps({
            "personas": [
                {"name": "Architect", "perspective": "structure", "blind_spots": "none"},
            ]
        })
        result = parse_structured_output(text, STAGE_SCHEMAS["personas"])
        assert result.success is True
        assert len(result.data["personas"]) == 1

    def test_parse_no_json(self):
        result = parse_structured_output("no json here", STAGE_SCHEMAS["intent"])
        assert result.success is False
        assert "JSON extraction failed" in result.errors[0]

    def test_parse_array_rejected_for_non_array_schema(self):
        text = json.dumps([1, 2, 3])
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.success is False
        assert "JSON object" in result.errors[0]

    def test_parse_preserves_raw_text(self):
        text = '{"core_need": "test thing", "domain": "x"}'
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.raw_text == text

    def test_parse_repair_hint_on_failure(self):
        text = json.dumps({"domain": "auth"})
        result = parse_structured_output(text, STAGE_SCHEMAS["intent"])
        assert result.success is False
        assert result.repair_hint
        assert "core_need" in result.repair_hint


# =============================================================================
# 5. build_repair_prompt() TESTS
# =============================================================================


class TestBuildRepairPrompt:
    """Test repair prompt construction."""

    def test_build_repair_prompt_includes_errors(self):
        result = ParseResult(
            success=False,
            data={},
            errors=("Missing required field: 'core_need'",),
            raw_text="bad",
            repair_hint="Include the 'core_need' field.",
        )
        prompt = build_repair_prompt(result)
        assert "Missing required field" in prompt
        assert "core_need" in prompt

    def test_build_repair_prompt_includes_hint(self):
        result = ParseResult(
            success=False,
            data={},
            errors=("Field 'x' too short",),
            raw_text="bad",
            repair_hint="Provide more content for 'x'.",
        )
        prompt = build_repair_prompt(result)
        assert "Provide more content" in prompt

    def test_build_repair_prompt_includes_context(self):
        result = ParseResult(
            success=False,
            data={},
            errors=("err",),
            raw_text="bad",
        )
        prompt = build_repair_prompt(result, context="Build an auth system")
        assert "auth system" in prompt

    def test_build_repair_prompt_truncates_context(self):
        result = ParseResult(
            success=False,
            data={},
            errors=("err",),
            raw_text="bad",
        )
        long_context = "x" * 1000
        prompt = build_repair_prompt(result, context=long_context)
        # Context truncated to 500 chars
        assert len(prompt) < 1000 + 200  # errors + formatting < 200


# =============================================================================
# 6. STAGE_SCHEMAS TESTS
# =============================================================================


class TestStageSchemas:
    """Test pre-built schemas exist and have correct structure."""

    def test_all_schemas_present(self):
        assert "intent" in STAGE_SCHEMAS
        assert "personas" in STAGE_SCHEMAS
        assert "synthesis" in STAGE_SCHEMAS
        assert "verification" in STAGE_SCHEMAS

    def test_personas_allows_array(self):
        assert STAGE_SCHEMAS["personas"].allows_array is True

    def test_intent_does_not_allow_array(self):
        assert STAGE_SCHEMAS["intent"].allows_array is False

    def test_synthesis_requires_components(self):
        schema = STAGE_SCHEMAS["synthesis"]
        comp_field = [f for f in schema.fields if f.name == "components"][0]
        assert comp_field.required is True
        assert comp_field.min_length == 1


# =============================================================================
# 7. LEAF MODULE CONSTRAINT
# =============================================================================


class TestLeafModule:
    """Verify output_parser.py has no project imports."""

    def test_output_parser_is_leaf_module(self):
        with open("core/output_parser.py") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "core" not in node.module, f"Non-leaf import: {node.module}"
                assert "api" not in node.module, f"Non-leaf import: {node.module}"
                assert "persistence" not in node.module, f"Non-leaf import: {node.module}"


# =============================================================================
# 8. ENGINE INTEGRATION TESTS
# =============================================================================


class TestEngineIntegration:
    """Test _parse_structured_output() method on engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from persistence.corpus import Corpus

        corpus = Corpus(corpus_path=tmp_path / "corpus")
        return MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

    def test_engine_parse_intent_valid(self, engine):
        text = json.dumps({
            "core_need": "Build a user auth system",
            "domain": "authentication",
            "actors": ["User"],
        })
        result = engine._parse_structured_output(text, "intent")
        assert result["core_need"] == "Build a user auth system"

    def test_engine_parse_synthesis_valid(self, engine):
        text = json.dumps({
            "components": [{"name": "User", "type": "entity"}],
            "relationships": [],
        })
        result = engine._parse_structured_output(text, "synthesis")
        assert len(result["components"]) == 1

    def test_engine_parse_verification_valid(self, engine):
        text = json.dumps({"status": "pass", "completeness": 85})
        result = engine._parse_structured_output(text, "verification")
        assert result["status"] == "pass"

    def test_engine_parse_verification_fallback(self, engine):
        """Bad JSON with no retry agent raises ValueError."""
        with pytest.raises(ValueError):
            engine._parse_structured_output("not json", "verification")

    def test_engine_parse_bad_json_no_agent_raises(self, engine):
        with pytest.raises(ValueError):
            engine._parse_structured_output("garbage", "intent")

    def test_engine_parse_intent_repair_retry(self, tmp_path):
        """When intent parse fails and agent is provided, retry once."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from core.protocol import SharedState, Message, MessageType
        from persistence.corpus import Corpus

        client = MockClient()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        eng = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        state = SharedState()
        msg = Message(sender="User", content="Build auth", message_type=MessageType.PROPOSITION)

        # Mock the intent agent to return valid JSON on retry
        good_response = json.dumps({
            "core_need": "Build a user auth system",
            "domain": "authentication",
            "actors": ["User"],
        })
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = good_response
        mock_agent.run = Mock(return_value=mock_response)

        result = eng._parse_structured_output(
            "bad json here", "intent",
            state=state, agent=mock_agent, original_msg=msg,
        )
        assert result["core_need"] == "Build a user auth system"
        assert mock_agent.run.called

    def test_engine_parse_personas_repair_retry(self, tmp_path):
        """When personas parse fails and agent is provided, retry once."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from core.protocol import SharedState, Message, MessageType
        from persistence.corpus import Corpus

        client = MockClient()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        eng = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        state = SharedState()
        msg = Message(sender="System", content="{}", message_type=MessageType.PROPOSITION)

        good_response = json.dumps({
            "personas": [{"name": "Arch", "perspective": "structure", "blind_spots": "none"}],
        })
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = good_response
        mock_agent.run = Mock(return_value=mock_response)

        result = eng._parse_structured_output(
            "not json", "personas",
            state=state, agent=mock_agent, original_msg=msg,
        )
        assert "personas" in result
        assert len(result["personas"]) == 1

    def test_engine_parse_retry_also_fails_raises(self, tmp_path):
        """When retry also fails, raise ValueError."""
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        from core.protocol import SharedState, Message, MessageType
        from persistence.corpus import Corpus

        client = MockClient()
        corpus = Corpus(corpus_path=tmp_path / "corpus")
        eng = MotherlabsEngine(
            llm_client=client,
            corpus=corpus,
            auto_store=False,
            cache_policy="none",
        )

        state = SharedState()
        msg = Message(sender="User", content="Build auth", message_type=MessageType.PROPOSITION)

        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "still bad json"
        mock_agent.run = Mock(return_value=mock_response)

        with pytest.raises(ValueError):
            eng._parse_structured_output(
                "bad json", "intent",
                state=state, agent=mock_agent, original_msg=msg,
            )
