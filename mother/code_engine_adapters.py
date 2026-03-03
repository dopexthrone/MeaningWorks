"""
Provider-specific tool-call adapters for the native coding agent.

NOT a LEAF module — imports from core/llm.py for provider type inspection
and raw SDK access. Only imported by mother/bridge.py.

Each adapter translates between the canonical tool definitions in
code_engine.py and the provider-specific wire format (Claude tool_use,
OpenAI function_calling, Gemini FunctionDeclaration).
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from mother.code_engine import ToolDef, ToolCall, ParsedResponse

logger = logging.getLogger("mother.code_engine_adapters")


# ---------------------------------------------------------------------------
# Claude adapter (Anthropic Messages API)
# ---------------------------------------------------------------------------

class ClaudeToolAdapter:
    """Adapter for Claude's native tool_use API."""

    def __init__(self, client: Any):
        """Accept a ClaudeClient from core/llm.py."""
        self._client = client

    @property
    def provider_name(self) -> str:
        return "claude"

    def format_tools(self, tools: List[ToolDef]) -> List[Dict]:
        """Convert to Claude tools format."""
        result = []
        for t in tools:
            schema = dict(t.parameters)
            schema["required"] = list(t.required)
            result.append({
                "name": t.name,
                "description": t.description,
                "input_schema": schema,
            })
        return result

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict],
        tools: Any,
        max_tokens: int,
        temperature: float,
    ) -> ParsedResponse:
        """Call Claude API with tool definitions."""
        # Access the raw Anthropic client
        raw_client = self._client.client

        response = raw_client.messages.create(
            model=self._client.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
            tools=tools,
        )

        # Parse response
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        usage = {}
        try:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        except Exception:
            pass

        return ParsedResponse(
            text="\n".join(text_parts),
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason or "",
            usage=usage,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> Dict:
        """Format tool result for Claude message history."""
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result,
            }],
        }

    def format_assistant_message(self, response: ParsedResponse) -> Dict:
        """Format assistant response for Claude message history."""
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# OpenAI adapter (Chat Completions API — also used by Grok)
# ---------------------------------------------------------------------------

class OpenAIToolAdapter:
    """Adapter for OpenAI's function calling API. Also works for Grok (OpenAI-compat)."""

    def __init__(self, client: Any, provider_label: str = "openai"):
        """Accept an OpenAIClient or GrokClient from core/llm.py."""
        self._client = client
        self._provider_label = provider_label

    @property
    def provider_name(self) -> str:
        return self._provider_label

    def format_tools(self, tools: List[ToolDef]) -> List[Dict]:
        """Convert to OpenAI function calling format."""
        result = []
        for t in tools:
            schema = dict(t.parameters)
            schema["required"] = list(t.required)
            result.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                },
            })
        return result

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict],
        tools: Any,
        max_tokens: int,
        temperature: float,
    ) -> ParsedResponse:
        """Call OpenAI/Grok API with function definitions."""
        raw_client = self._client.client

        # Build messages with system
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        kwargs: Dict[str, Any] = {
            "model": self._client.model,
            "messages": full_messages,
            "tools": tools,
            "temperature": temperature,
        }

        # GPT-5+ uses max_completion_tokens
        model = self._client.model
        if model.startswith("gpt-5") or model.startswith("gpt-4.1"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        response = raw_client.chat.completions.create(**kwargs)

        # Parse
        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    raw_args = tc.function.arguments
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif raw_args:
                        args = json.loads(raw_args)
                    else:
                        args = {}
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Failed to parse tool args for %s: %s (raw: %.200s)",
                                   tc.function.name, e, tc.function.arguments)
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = {}
        try:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        except Exception:
            pass

        return ParsedResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            stop_reason=choice.finish_reason or "",
            usage=usage,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> Dict:
        """Format tool result for OpenAI message history."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    def format_assistant_message(self, response: ParsedResponse) -> Dict:
        """Format assistant response for OpenAI message history."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if response.text:
            msg["content"] = response.text
        else:
            msg["content"] = None

        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg


# ---------------------------------------------------------------------------
# Gemini adapter (Google GenAI)
# ---------------------------------------------------------------------------

class GeminiToolAdapter:
    """Adapter for Google Gemini's function calling API."""

    def __init__(self, client: Any):
        """Accept a GeminiClient from core/llm.py."""
        self._client = client
        self._call_counter = 0

    @property
    def provider_name(self) -> str:
        return "gemini"

    def format_tools(self, tools: List[ToolDef]) -> Any:
        """Convert to Gemini FunctionDeclaration format."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai required for Gemini tool adapter")

        declarations = []
        for t in tools:
            # Build Schema from parameters
            props = t.parameters.get("properties", {})
            schema_props = {}
            for pname, pdef in props.items():
                ptype = pdef.get("type", "string").upper()
                if ptype == "INTEGER":
                    ptype = "NUMBER"
                schema_props[pname] = genai.types.Schema(
                    type=getattr(genai.types.Type, ptype, genai.types.Type.STRING),
                    description=pdef.get("description", ""),
                )

            declarations.append(genai.types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties=schema_props,
                    required=list(t.required),
                ),
            ))

        return genai.types.Tool(function_declarations=declarations)

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict],
        tools: Any,
        max_tokens: int,
        temperature: float,
    ) -> ParsedResponse:
        """Call Gemini API with function declarations."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai required")

        # Gemini needs a fresh model with tools and system instruction
        genai.configure(api_key=self._client.api_key)
        model = genai.GenerativeModel(
            self._client.model,
            tools=[tools],
            system_instruction=system,
        )

        # Convert messages to Gemini Content format
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                if isinstance(content, str):
                    contents.append(genai.types.Content(
                        role="user",
                        parts=[genai.types.Part.from_text(content)],
                    ))
                elif isinstance(content, list):
                    # Tool results
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "function_response":
                            parts.append(genai.types.Part.from_function_response(
                                name=item["name"],
                                response={"result": item["content"]},
                            ))
                        elif isinstance(item, str):
                            parts.append(genai.types.Part.from_text(item))
                    if parts:
                        contents.append(genai.types.Content(role="user", parts=parts))
            elif role == "model":
                parts = []
                content_data = msg.get("content", "")
                if isinstance(content_data, str) and content_data:
                    parts.append(genai.types.Part.from_text(content_data))
                # Add function calls if present
                for fc in msg.get("function_calls", []):
                    parts.append(genai.types.Part(
                        function_call=genai.types.FunctionCall(
                            name=fc["name"],
                            args=fc["arguments"],
                        ),
                    ))
                if parts:
                    contents.append(genai.types.Content(role="model", parts=parts))

        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # Parse
        text_parts = []
        tool_calls = []

        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    self._call_counter += 1
                    tool_calls.append(ToolCall(
                        id=f"gemini_{self._call_counter}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))
        except (IndexError, AttributeError):
            pass

        usage = {}
        try:
            um = response.usage_metadata
            usage = {
                "input_tokens": um.prompt_token_count,
                "output_tokens": um.candidates_token_count,
                "total_tokens": um.prompt_token_count + um.candidates_token_count,
            }
        except Exception:
            pass

        return ParsedResponse(
            text="\n".join(text_parts),
            tool_calls=tuple(tool_calls),
            stop_reason="end_turn" if not tool_calls else "tool_use",
            usage=usage,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> Dict:
        """Format tool result for Gemini message history."""
        return {
            "role": "user",
            "content": [{
                "type": "function_response",
                "name": tool_name,
                "content": result,
            }],
        }

    def format_assistant_message(self, response: ParsedResponse) -> Dict:
        """Format assistant response for Gemini message history."""
        msg: Dict[str, Any] = {"role": "model"}
        if response.text:
            msg["content"] = response.text
        function_calls = []
        for tc in response.tool_calls:
            function_calls.append({
                "name": tc.name,
                "arguments": tc.arguments,
            })
        if function_calls:
            msg["function_calls"] = function_calls
        return msg


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------

def select_adapter(client: Any) -> Any:
    """Select the appropriate tool-call adapter for an LLM client.

    Inspects the client type and returns the matching adapter.
    For FailoverClient, uses the current primary provider.
    """
    # Import here to avoid circular imports at module load
    from core.llm import ClaudeClient, OpenAIClient, GrokClient, GeminiClient, FailoverClient

    if isinstance(client, ClaudeClient):
        return ClaudeToolAdapter(client)
    elif isinstance(client, OpenAIClient):
        return OpenAIToolAdapter(client, provider_label="openai")
    elif isinstance(client, GrokClient):
        return OpenAIToolAdapter(client, provider_label="grok")
    elif isinstance(client, GeminiClient):
        return GeminiToolAdapter(client)
    elif isinstance(client, FailoverClient):
        # Use the first available provider's adapter
        for provider_client in client.providers:
            try:
                return select_adapter(provider_client)
            except (ValueError, ImportError):
                continue
        raise ValueError("No adapter available for any FailoverClient provider")
    else:
        raise ValueError(f"No tool adapter for {type(client).__name__}")
