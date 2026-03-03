"""
Tests for mother/code_engine_adapters.py

Covers the three provider adapters (Claude, OpenAI, Gemini) and the
select_adapter() dispatch function. Only format/parse methods are tested —
call_with_tools() makes real API calls and is NOT tested here.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from mother.code_engine import ToolDef, ToolCall, ParsedResponse
from mother.code_engine_adapters import (
    ClaudeToolAdapter,
    OpenAIToolAdapter,
    GeminiToolAdapter,
    select_adapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_def(
    name: str = "read_file",
    description: str = "Read a file from disk",
    parameters: dict | None = None,
    required: tuple = ("path",),
) -> ToolDef:
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "description": "Encoding"},
            },
        }
    return ToolDef(
        name=name,
        description=description,
        parameters=parameters,
        required=required,
    )


def _make_tool_call(
    id: str = "tc_001",
    name: str = "read_file",
    arguments: dict | None = None,
) -> ToolCall:
    if arguments is None:
        arguments = {"path": "/tmp/foo.py"}
    return ToolCall(id=id, name=name, arguments=arguments)


def _make_parsed_response(
    text: str = "I will read the file.",
    tool_calls: tuple = (),
    stop_reason: str = "end_turn",
) -> ParsedResponse:
    return ParsedResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        raw=None,
    )


def _mock_client():
    """A generic mock LLM client with .client and .model attributes."""
    c = MagicMock()
    c.client = MagicMock()
    c.model = "test-model"
    return c


# ===================================================================
# A. ClaudeToolAdapter Tests
# ===================================================================

class TestClaudeToolAdapter:

    def test_provider_name(self):
        adapter = ClaudeToolAdapter(_mock_client())
        assert adapter.provider_name == "claude"

    def test_format_tools_single(self):
        adapter = ClaudeToolAdapter(_mock_client())
        td = _make_tool_def()
        result = adapter.format_tools([td])

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "read_file"
        assert tool["description"] == "Read a file from disk"
        assert "input_schema" in tool
        assert tool["input_schema"]["required"] == ["path"]
        assert "properties" in tool["input_schema"]

    def test_format_tools_multiple(self):
        adapter = ClaudeToolAdapter(_mock_client())
        tools = [
            _make_tool_def(name="read_file"),
            _make_tool_def(name="write_file", description="Write a file", required=("path", "content")),
        ]
        result = adapter.format_tools(tools)
        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert result[1]["name"] == "write_file"
        assert result[1]["input_schema"]["required"] == ["path", "content"]

    def test_format_tools_empty_list(self):
        adapter = ClaudeToolAdapter(_mock_client())
        result = adapter.format_tools([])
        assert result == []

    def test_format_tool_result(self):
        adapter = ClaudeToolAdapter(_mock_client())
        result = adapter.format_tool_result("toolu_123", "read_file", "file contents here")

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_123"
        assert block["content"] == "file contents here"

    def test_format_assistant_message_with_text_and_tools(self):
        adapter = ClaudeToolAdapter(_mock_client())
        tc = _make_tool_call(id="toolu_abc", name="read_file", arguments={"path": "/x"})
        resp = _make_parsed_response(
            text="Let me read that.",
            tool_calls=(tc,),
            stop_reason="tool_use",
        )
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 2
        # First block: text
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me read that."
        # Second block: tool_use
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "toolu_abc"
        assert content[1]["name"] == "read_file"
        assert content[1]["input"] == {"path": "/x"}

    def test_format_assistant_message_text_only(self):
        adapter = ClaudeToolAdapter(_mock_client())
        resp = _make_parsed_response(text="All done.", tool_calls=())
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "All done."

    def test_format_assistant_message_tools_only_no_text(self):
        adapter = ClaudeToolAdapter(_mock_client())
        tc = _make_tool_call()
        resp = _make_parsed_response(text="", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        # No text block since text is empty
        assert all(block["type"] == "tool_use" for block in msg["content"])

    def test_format_assistant_message_multiple_tool_calls(self):
        adapter = ClaudeToolAdapter(_mock_client())
        tc1 = _make_tool_call(id="tc_1", name="read_file", arguments={"path": "/a"})
        tc2 = _make_tool_call(id="tc_2", name="write_file", arguments={"path": "/b", "content": "x"})
        resp = _make_parsed_response(text="", tool_calls=(tc1, tc2))
        msg = adapter.format_assistant_message(resp)

        assert len(msg["content"]) == 2
        assert msg["content"][0]["name"] == "read_file"
        assert msg["content"][1]["name"] == "write_file"


# ===================================================================
# B. OpenAIToolAdapter Tests
# ===================================================================

class TestOpenAIToolAdapter:

    def test_provider_name_default(self):
        adapter = OpenAIToolAdapter(_mock_client())
        assert adapter.provider_name == "openai"

    def test_provider_name_custom_label(self):
        adapter = OpenAIToolAdapter(_mock_client(), provider_label="grok")
        assert adapter.provider_name == "grok"

    def test_format_tools_single(self):
        adapter = OpenAIToolAdapter(_mock_client())
        td = _make_tool_def()
        result = adapter.format_tools([td])

        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert "function" in tool
        func = tool["function"]
        assert func["name"] == "read_file"
        assert func["description"] == "Read a file from disk"
        assert func["parameters"]["required"] == ["path"]
        assert "properties" in func["parameters"]

    def test_format_tools_multiple(self):
        adapter = OpenAIToolAdapter(_mock_client())
        tools = [
            _make_tool_def(name="read_file"),
            _make_tool_def(name="bash", description="Run a command", required=("command",)),
        ]
        result = adapter.format_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "read_file"
        assert result[1]["function"]["name"] == "bash"
        assert result[1]["function"]["parameters"]["required"] == ["command"]

    def test_format_tools_empty(self):
        adapter = OpenAIToolAdapter(_mock_client())
        assert adapter.format_tools([]) == []

    def test_format_tool_result(self):
        adapter = OpenAIToolAdapter(_mock_client())
        result = adapter.format_tool_result("call_xyz", "read_file", "hello world")

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_xyz"
        assert result["content"] == "hello world"

    def test_format_assistant_message_with_text_and_tools(self):
        adapter = OpenAIToolAdapter(_mock_client())
        tc = _make_tool_call(id="call_1", name="read_file", arguments={"path": "/x"})
        resp = _make_parsed_response(text="Reading...", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Reading..."
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc_out = msg["tool_calls"][0]
        assert tc_out["id"] == "call_1"
        assert tc_out["type"] == "function"
        assert tc_out["function"]["name"] == "read_file"
        # Arguments are JSON-serialized
        assert json.loads(tc_out["function"]["arguments"]) == {"path": "/x"}

    def test_format_assistant_message_text_only(self):
        adapter = OpenAIToolAdapter(_mock_client())
        resp = _make_parsed_response(text="Done.", tool_calls=())
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Done."
        assert "tool_calls" not in msg

    def test_format_assistant_message_no_text_with_tools(self):
        adapter = OpenAIToolAdapter(_mock_client())
        tc = _make_tool_call()
        resp = _make_parsed_response(text="", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "assistant"
        assert msg["content"] is None  # OpenAI sets null when no text
        assert "tool_calls" in msg

    def test_format_assistant_message_arguments_serialized_as_json(self):
        adapter = OpenAIToolAdapter(_mock_client())
        tc = _make_tool_call(arguments={"path": "/tmp/test.py", "count": 42, "flag": True})
        resp = _make_parsed_response(text="", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        raw_args = msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw_args, str)
        parsed = json.loads(raw_args)
        assert parsed["count"] == 42
        assert parsed["flag"] is True


# ===================================================================
# C. GeminiToolAdapter Tests
# ===================================================================

class TestGeminiToolAdapter:

    def test_provider_name(self):
        adapter = GeminiToolAdapter(_mock_client())
        assert adapter.provider_name == "gemini"

    def test_format_tool_result(self):
        adapter = GeminiToolAdapter(_mock_client())
        result = adapter.format_tool_result("gemini_1", "read_file", "contents")

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "function_response"
        assert block["name"] == "read_file"
        assert block["content"] == "contents"

    def test_format_assistant_message_with_text_and_function_calls(self):
        adapter = GeminiToolAdapter(_mock_client())
        tc = _make_tool_call(id="gemini_1", name="read_file", arguments={"path": "/x"})
        resp = _make_parsed_response(text="Reading now.", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "model"
        assert msg["content"] == "Reading now."
        assert "function_calls" in msg
        assert len(msg["function_calls"]) == 1
        fc = msg["function_calls"][0]
        assert fc["name"] == "read_file"
        assert fc["arguments"] == {"path": "/x"}

    def test_format_assistant_message_text_only(self):
        adapter = GeminiToolAdapter(_mock_client())
        resp = _make_parsed_response(text="Finished.", tool_calls=())
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "model"
        assert msg["content"] == "Finished."
        assert "function_calls" not in msg

    def test_format_assistant_message_no_text_with_function_calls(self):
        adapter = GeminiToolAdapter(_mock_client())
        tc = _make_tool_call()
        resp = _make_parsed_response(text="", tool_calls=(tc,))
        msg = adapter.format_assistant_message(resp)

        assert msg["role"] == "model"
        assert "content" not in msg  # empty text is falsy, no content key
        assert "function_calls" in msg

    def test_format_assistant_message_multiple_function_calls(self):
        adapter = GeminiToolAdapter(_mock_client())
        tc1 = _make_tool_call(id="g_1", name="read_file", arguments={"path": "/a"})
        tc2 = _make_tool_call(id="g_2", name="bash", arguments={"command": "ls"})
        resp = _make_parsed_response(text="", tool_calls=(tc1, tc2))
        msg = adapter.format_assistant_message(resp)

        assert len(msg["function_calls"]) == 2
        assert msg["function_calls"][0]["name"] == "read_file"
        assert msg["function_calls"][1]["name"] == "bash"

    def test_format_tools_requires_google_generativeai(self):
        """format_tools raises ImportError if google.generativeai is not installed."""
        adapter = GeminiToolAdapter(_mock_client())
        td = _make_tool_def()
        with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
            with pytest.raises(ImportError, match="google-generativeai"):
                adapter.format_tools([td])

    def test_call_counter_initial_state(self):
        adapter = GeminiToolAdapter(_mock_client())
        assert adapter._call_counter == 0


# ===================================================================
# D. select_adapter() Tests
# ===================================================================

class TestSelectAdapter:
    """Test select_adapter() dispatch logic.

    We patch the imports inside select_adapter so isinstance checks
    work with our mock classes, without needing real API keys.
    """

    def _make_typed_mock(self, cls):
        """Create a MagicMock that passes isinstance checks for cls."""
        mock = MagicMock(spec=cls)
        mock.client = MagicMock()
        mock.model = "test-model"
        return mock

    def test_claude_client_returns_claude_adapter(self):
        from core.llm import ClaudeClient
        client = self._make_typed_mock(ClaudeClient)
        adapter = select_adapter(client)
        assert isinstance(adapter, ClaudeToolAdapter)
        assert adapter.provider_name == "claude"

    def test_openai_client_returns_openai_adapter(self):
        from core.llm import OpenAIClient
        client = self._make_typed_mock(OpenAIClient)
        adapter = select_adapter(client)
        assert isinstance(adapter, OpenAIToolAdapter)
        assert adapter.provider_name == "openai"

    def test_grok_client_returns_openai_adapter_with_grok_label(self):
        from core.llm import GrokClient
        client = self._make_typed_mock(GrokClient)
        adapter = select_adapter(client)
        assert isinstance(adapter, OpenAIToolAdapter)
        assert adapter.provider_name == "grok"

    def test_gemini_client_returns_gemini_adapter(self):
        from core.llm import GeminiClient
        client = self._make_typed_mock(GeminiClient)
        adapter = select_adapter(client)
        assert isinstance(adapter, GeminiToolAdapter)
        assert adapter.provider_name == "gemini"

    def test_failover_client_returns_first_provider_adapter(self):
        from core.llm import FailoverClient, ClaudeClient
        # The FailoverClient mock needs a `providers` list
        failover = self._make_typed_mock(FailoverClient)
        inner_claude = self._make_typed_mock(ClaudeClient)
        failover.providers = [inner_claude]
        adapter = select_adapter(failover)
        assert isinstance(adapter, ClaudeToolAdapter)

    def test_failover_client_skips_bad_provider_uses_second(self):
        """If the first provider in failover raises, tries the next."""
        from core.llm import FailoverClient, OpenAIClient

        failover = self._make_typed_mock(FailoverClient)
        # First provider is unrecognized (plain MagicMock)
        bad_provider = MagicMock()
        bad_provider.__class__ = type("UnknownClient", (), {})
        good_provider = self._make_typed_mock(OpenAIClient)
        failover.providers = [bad_provider, good_provider]
        adapter = select_adapter(failover)
        assert isinstance(adapter, OpenAIToolAdapter)
        assert adapter.provider_name == "openai"

    def test_failover_client_all_bad_raises(self):
        from core.llm import FailoverClient

        failover = self._make_typed_mock(FailoverClient)
        bad1 = MagicMock()
        bad1.__class__ = type("UnknownClient1", (), {})
        bad2 = MagicMock()
        bad2.__class__ = type("UnknownClient2", (), {})
        failover.providers = [bad1, bad2]
        with pytest.raises(ValueError, match="No adapter available"):
            select_adapter(failover)

    def test_unknown_client_raises_value_error(self):
        unknown = MagicMock()
        unknown.__class__ = type("TotallyNewClient", (), {})
        with pytest.raises(ValueError, match="No tool adapter"):
            select_adapter(unknown)


# ===================================================================
# E. Edge cases & cross-cutting
# ===================================================================

class TestEdgeCases:

    def test_format_tools_preserves_parameter_properties(self):
        """Ensure original parameter dict properties are not lost during formatting."""
        params = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path"},
                "line": {"type": "integer", "description": "Line number"},
            },
        }
        td = ToolDef(
            name="edit_file",
            description="Edit a file",
            parameters=params,
            required=("path", "line"),
        )

        # Claude
        claude_tools = ClaudeToolAdapter(_mock_client()).format_tools([td])
        schema = claude_tools[0]["input_schema"]
        assert "path" in schema["properties"]
        assert "line" in schema["properties"]
        assert set(schema["required"]) == {"path", "line"}

        # OpenAI
        openai_tools = OpenAIToolAdapter(_mock_client()).format_tools([td])
        params_out = openai_tools[0]["function"]["parameters"]
        assert "path" in params_out["properties"]
        assert "line" in params_out["properties"]
        assert set(params_out["required"]) == {"path", "line"}

    def test_format_tools_empty_required(self):
        """Tools with no required fields should have empty required list."""
        td = ToolDef(
            name="noop",
            description="Does nothing",
            parameters={"type": "object", "properties": {}},
            required=(),
        )

        claude_result = ClaudeToolAdapter(_mock_client()).format_tools([td])
        assert claude_result[0]["input_schema"]["required"] == []

        openai_result = OpenAIToolAdapter(_mock_client()).format_tools([td])
        assert openai_result[0]["function"]["parameters"]["required"] == []

    def test_tool_result_preserves_arbitrary_content_string(self):
        """Tool result content can be any string — JSON, error messages, etc."""
        content = '{"error": "file not found", "code": 404}'

        claude_msg = ClaudeToolAdapter(_mock_client()).format_tool_result("id1", "read_file", content)
        assert claude_msg["content"][0]["content"] == content

        openai_msg = OpenAIToolAdapter(_mock_client()).format_tool_result("id1", "read_file", content)
        assert openai_msg["content"] == content

        gemini_msg = GeminiToolAdapter(_mock_client()).format_tool_result("id1", "read_file", content)
        assert gemini_msg["content"][0]["content"] == content

    def test_parsed_response_with_empty_fields(self):
        """Adapters handle a ParsedResponse with all empty/default fields."""
        resp = ParsedResponse()  # all defaults

        claude_msg = ClaudeToolAdapter(_mock_client()).format_assistant_message(resp)
        assert claude_msg["role"] == "assistant"
        assert claude_msg["content"] == []  # no text, no tool calls

        openai_msg = OpenAIToolAdapter(_mock_client()).format_assistant_message(resp)
        assert openai_msg["role"] == "assistant"
        assert openai_msg["content"] is None
        assert "tool_calls" not in openai_msg

        gemini_msg = GeminiToolAdapter(_mock_client()).format_assistant_message(resp)
        assert gemini_msg["role"] == "model"
        assert "content" not in gemini_msg
        assert "function_calls" not in gemini_msg
