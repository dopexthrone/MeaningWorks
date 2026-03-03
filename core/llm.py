"""
Motherlabs LLM Client - Multi-provider support.

Derived from: PROJECT-PLAN.md Section 0.2

Responsibilities:
- Unified interface for multiple LLM providers
- complete_with_system(system_prompt, user_content) -> str
- Handle basic errors
- Support deterministic mode (C008) via temperature=0

Supported Providers:
- Anthropic Claude (claude-sonnet-4, claude-opus-4, etc.)
- OpenAI (gpt-4o, gpt-4-turbo, etc.)
- Google Gemini (gemini-1.5-pro, gemini-2.0-flash, etc.)
"""

import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generator, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# HTTP status codes that are transient and should be retried
_RETRYABLE_STATUS_CODES = {500, 502, 503, 529}
_MAX_API_RETRIES = 3
_INITIAL_BACKOFF_SECONDS = 1.0


class RouteTier(Enum):
    """LLM routing tiers for cost/quality optimization.

    CRITICAL: synthesis, verification, governor — highest quality model
    STANDARD: dialogue, personas — balanced cost/quality
    LOCAL: emergence, observer — cheapest/fastest
    """
    CRITICAL = "critical"
    STANDARD = "standard"
    LOCAL = "local"


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All providers must implement complete_with_system().
    """

    def __init__(self, deterministic: bool = True):
        self.deterministic = deterministic
        self.last_usage: Dict[str, Any] = {}
        self._thread_local = threading.local()

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        """Generate completion from messages.

        Args:
            tier: Optional routing tier hint. Concrete clients may ignore it;
                  FailoverClient uses it to select preferred providers.
        """
        pass

    @abstractmethod
    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        """Convenience method for single-turn with system prompt."""
        pass

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks. Default: fall back to complete() as single yield."""
        yield self.complete(messages, system=system, max_tokens=max_tokens, temperature=temperature)

    def _store_usage(self, usage: Dict[str, Any]) -> None:
        """Store usage data on both instance and thread-local for concurrent safety."""
        self.last_usage = usage
        self._thread_local.last_usage = dict(usage)

    def _get_temperature(self, temperature: Optional[float]) -> float:
        """Get temperature respecting deterministic setting (C008)."""
        if temperature is not None:
            return temperature
        return 0.0 if self.deterministic else 0.7


class ClaudeClient(BaseLLMClient):
    """
    Client for Anthropic Claude API.

    Derived from: PROJECT-PLAN.md Phase 0.2

    Constraint C008 (Determinism):
    - deterministic=True sets temperature=0
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        deterministic: bool = True
    ):
        super().__init__(deterministic)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load Anthropic client with socket-level timeout."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                import httpx
                # Socket-level timeout: works in worker threads where SIGALRM can't.
                # connect=30s, read=300s (single token gap), write=30s, pool=30s.
                self._client = Anthropic(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(300.0, connect=30.0),
                )
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install: pip install anthropic"
                )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        temperature = self._get_temperature(temperature)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }

        if system:
            kwargs["system"] = system

        # Use streaming to avoid SDK timeout restrictions on large max_tokens
        # Retry on transient API errors (500, 502, 503, 529) with exponential backoff
        last_error = None
        for retry in range(_MAX_API_RETRIES):
            try:
                collected_text = []
                with self.client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        collected_text.append(text)
                    # Phase 21: Extract token usage from final message
                    try:
                        final = stream.get_final_message()
                        self._store_usage({
                            "input_tokens": final.usage.input_tokens,
                            "output_tokens": final.usage.output_tokens,
                            "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
                        })
                    except Exception as e:
                        logger.debug(f"Claude streaming usage extraction skipped: {e}")
                        self._store_usage({})
                return "".join(collected_text)
            except Exception as e:
                last_error = e
                status = getattr(e, 'status_code', None)
                if status in _RETRYABLE_STATUS_CODES and retry < _MAX_API_RETRIES - 1:
                    backoff = _INITIAL_BACKOFF_SECONDS * (2 ** retry)
                    logger.warning(f"Transient API error (HTTP {status}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                raise
        raise last_error  # Should not reach here, but safety net

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks from Claude streaming API."""
        temperature = self._get_temperature(temperature)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }
        if system:
            kwargs["system"] = system

        last_error = None
        for retry in range(_MAX_API_RETRIES):
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    for text in stream.text_stream:
                        yield text
                    try:
                        final = stream.get_final_message()
                        self._store_usage({
                            "input_tokens": final.usage.input_tokens,
                            "output_tokens": final.usage.output_tokens,
                            "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
                        })
                    except Exception as e:
                        logger.debug(f"Claude stream-gen usage extraction skipped: {e}")
                        self._store_usage({})
                return
            except Exception as e:
                last_error = e
                status = getattr(e, 'status_code', None)
                if status in _RETRYABLE_STATUS_CODES and retry < _MAX_API_RETRIES - 1:
                    backoff = _INITIAL_BACKOFF_SECONDS * (2 ** retry)
                    logger.warning(f"Transient API error (HTTP {status}), retrying in {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                raise
        raise last_error


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI API.

    Supports: gpt-5.1, gpt-5, gpt-4o, gpt-4-turbo, etc.
    Default: gpt-5.1 (latest, powerful - released Nov 2025)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        deterministic: bool = True
    ):
        super().__init__(deterministic)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load OpenAI client with socket-level timeout."""
        if self._client is None:
            try:
                from openai import OpenAI
                import httpx
                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(300.0, connect=30.0),
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install: pip install openai"
                )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        temperature = self._get_temperature(temperature)

        # OpenAI uses system message in messages array
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # GPT-5+ models use max_completion_tokens, older use max_tokens
        if self.model.startswith("gpt-5") or self.model.startswith("gpt-4.1"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        # Phase 21: Extract token usage
        try:
            self._store_usage({
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            })
        except Exception as e:
            logger.debug(f"OpenAI usage extraction skipped: {e}")
            self._store_usage({})
        return response.choices[0].message.content

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks from OpenAI streaming API."""
        temperature = self._get_temperature(temperature)

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "stream": True,
        }
        if self.model.startswith("gpt-5") or self.model.startswith("gpt-4.1"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        collected_tokens = 0
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_tokens += 1
                yield chunk.choices[0].delta.content
        # Usage not available in streaming mode for OpenAI — estimate
        self._store_usage({})


class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini API.

    Supports: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, etc.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        deterministic: bool = True
    ):
        super().__init__(deterministic)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self._client = None

        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. Install: pip install google-generativeai"
                )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        temperature = self._get_temperature(temperature)

        # Build content for Gemini
        # Gemini uses system_instruction at model level, but we can prepend to prompt
        content_parts = []

        if system:
            content_parts.append(f"SYSTEM INSTRUCTIONS:\n{system}\n\n---\n\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                content_parts.append(f"USER: {content}\n")
            elif role == "assistant":
                content_parts.append(f"ASSISTANT: {content}\n")

        full_content = "".join(content_parts)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = self.client.generate_content(
            full_content,
            generation_config=generation_config
        )
        # Phase 21: Extract token usage
        try:
            um = response.usage_metadata
            self._store_usage({
                "input_tokens": um.prompt_token_count,
                "output_tokens": um.candidates_token_count,
                "total_tokens": um.prompt_token_count + um.candidates_token_count,
            })
        except Exception as e:
            logger.debug(f"Gemini usage extraction skipped: {e}")
            self._store_usage({})
        return response.text

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks from Gemini streaming API."""
        temperature = self._get_temperature(temperature)

        content_parts = []
        if system:
            content_parts.append(f"SYSTEM INSTRUCTIONS:\n{system}\n\n---\n\n")
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                content_parts.append(f"USER: {content}\n")
            elif role == "assistant":
                content_parts.append(f"ASSISTANT: {content}\n")

        full_content = "".join(content_parts)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = self.client.generate_content(
            full_content,
            generation_config=generation_config,
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
        self._store_usage({})


class OllamaClient(BaseLLMClient):
    """
    Client for local Ollama LLM server.

    Uses Ollama's native /api/chat endpoint (not OpenAI compat) for
    accurate token tracking via prompt_eval_count/eval_count.

    Supports: llama3:8b, mistral, codellama, etc.
    No API key required — runs locally.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        deterministic: bool = True
    ):
        super().__init__(deterministic)
        self.base_url = base_url.rstrip("/")
        self.model = model

    @staticmethod
    def is_available(base_url: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
        """Check if Ollama server is reachable."""
        import requests
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    @staticmethod
    def list_models(base_url: str = "http://localhost:11434", timeout: float = 5.0) -> list:
        """List available models on the Ollama server."""
        import requests
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Ollama model listing failed: {e}")
            return []

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        import requests

        temperature = self._get_temperature(temperature)

        # Build messages list — Ollama uses system role in messages array
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract token usage from native Ollama response
        self._store_usage({
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        })

        message = data.get("message", {})
        return message.get("content", "")

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks from Ollama streaming API (NDJSON)."""
        import requests

        temperature = self._get_temperature(temperature)

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()

        import json as _json
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = _json.loads(line)
            # Yield content from intermediate chunks
            message = chunk.get("message", {})
            content = message.get("content", "")
            if content:
                yield content
            # Final chunk has done: true with token counts
            if chunk.get("done"):
                self._store_usage({
                    "input_tokens": chunk.get("prompt_eval_count", 0),
                    "output_tokens": chunk.get("eval_count", 0),
                    "total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0),
                })


class GrokClient(BaseLLMClient):
    """
    Client for xAI Grok API.

    Supports: grok-3, grok-3-mini, grok-2, etc.
    Uses OpenAI-compatible API format.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-reasoning",
        deterministic: bool = True
    ):
        super().__init__(deterministic)
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.model = model
        self._client = None
        # Session-scoped conversation ID for xAI prompt caching.
        # Reusing the same ID across calls caches input tokens at $0.02/M
        # instead of $2.00/M (100x reduction on repeated system prompts).
        self._conv_id = str(uuid.uuid4())

        if not self.api_key:
            raise ValueError(
                "API key required. Set XAI_API_KEY or pass api_key parameter."
            )

    @property
    def client(self):
        """Lazy-load OpenAI client configured for xAI with socket-level timeout."""
        if self._client is None:
            try:
                from openai import OpenAI
                import httpx
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1",
                    timeout=httpx.Timeout(300.0, connect=30.0),
                    default_headers={"x-grok-conv-id": self._conv_id},
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install: pip install openai"
                )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        temperature = self._get_temperature(temperature)

        # Grok uses OpenAI-compatible format
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Phase 21: Extract token usage (OpenAI-compatible)
        try:
            self._store_usage({
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            })
        except Exception as e:
            logger.debug(f"Grok usage extraction skipped: {e}")
            self._store_usage({})
        return response.choices[0].message.content

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Yield token chunks from Grok streaming API."""
        temperature = self._get_temperature(temperature)

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        self._store_usage({})


class MockClient(BaseLLMClient):
    """Mock client for testing without API."""

    def __init__(self):
        super().__init__(deterministic=True)
        self.call_count = 0

    def complete(self, messages: List[Dict], tier: Optional[RouteTier] = None, **kwargs) -> str:
        self.call_count += 1
        self._store_usage({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        return f"[Mock response #{self.call_count}]"

    def complete_with_system(self, system_prompt: str, user_content: str, tier: Optional[RouteTier] = None, **kwargs) -> str:
        self.call_count += 1
        self._store_usage({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        return f"[Mock response #{self.call_count}]"

    def stream(self, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        """Yield word-by-word for testing."""
        text = f"[Mock response #{self.call_count + 1}]"
        self.call_count += 1
        self._store_usage({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        for word in text.split():
            yield word + " "


class FailoverClient(BaseLLMClient):
    """
    Wraps multiple LLM providers with automatic failover.

    Phase 6.1: Enterprise Scale & Reliability

    When the primary provider fails, automatically tries the next provider
    in the list until one succeeds or all providers are exhausted.

    Example:
        client = FailoverClient([
            GrokClient(),
            ClaudeClient(),
            OpenAIClient()
        ])
        # Will try Grok first, then Claude, then OpenAI
        result = client.complete_with_system("system", "user")
    """

    def __init__(
        self,
        providers: List[BaseLLMClient],
        logger=None,
        tier_map: Optional[Dict[RouteTier, List[int]]] = None,
    ):
        """
        Initialize failover client with a list of providers.

        Args:
            providers: List of LLM clients to try in order
            logger: Optional logger for failover events
            tier_map: Optional mapping from RouteTier to preferred provider
                      indices. When a tier is specified in complete(), providers
                      at those indices are tried first, then remaining providers
                      as fallback. Without tier_map: identical to current behavior.
        """
        if not providers:
            raise ValueError("At least one provider is required")

        super().__init__(deterministic=providers[0].deterministic)
        self.providers = providers
        self.logger = logger
        self.tier_map = tier_map
        self._last_successful_idx = 0

    @property
    def provider_name(self) -> str:
        """Return name of current primary provider."""
        provider = self.providers[self._last_successful_idx]
        if hasattr(provider, 'model'):
            return provider.__class__.__name__.replace('Client', '').lower()
        return "failover"

    @property
    def model_name(self) -> str:
        """Return model name of current primary provider."""
        provider = self.providers[self._last_successful_idx]
        if hasattr(provider, 'model'):
            return provider.model
        return "failover"

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        """
        Complete with automatic failover across providers.

        When tier and tier_map are both set, preferred providers for that tier
        are tried first, then the rest as fallback. Otherwise tries providers
        in their original order.
        """
        from core.exceptions import FailoverExhaustedError, ProviderUnavailableError

        errors = {}
        providers_tried = []

        # Build provider order: tier-preferred first, then remainder
        order = list(range(len(self.providers)))
        if tier and self.tier_map and tier in self.tier_map:
            preferred = [i for i in self.tier_map[tier] if 0 <= i < len(self.providers)]
            rest = [i for i in order if i not in preferred]
            order = preferred + rest

        for i in order:
            provider = self.providers[i]
            provider_name = provider.__class__.__name__
            providers_tried.append(provider_name)

            try:
                result = provider.complete(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                self._last_successful_idx = i
                # Phase 21: Relay token usage from successful provider
                self._store_usage(getattr(provider, 'last_usage', {}))

                if self.logger and i > 0:
                    self.logger.info(
                        f"Failover success: {provider_name} "
                        f"(after {i} failed providers)"
                    )

                return result

            except Exception as e:
                errors[provider_name] = str(e)

                if self.logger:
                    self.logger.warning(
                        f"Provider {provider_name} failed: {e}"
                    )

                # Continue to next provider
                continue

        # All providers failed
        raise FailoverExhaustedError(
            f"All {len(self.providers)} providers failed",
            providers_tried=providers_tried,
            errors=errors
        )

    def complete_with_system(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        tier: Optional[RouteTier] = None,
    ) -> str:
        """
        Complete with system prompt, using automatic failover.
        """
        return self.complete(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tier=tier,
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Stream with automatic failover across providers."""
        from core.exceptions import FailoverExhaustedError

        errors = {}
        providers_tried = []

        for i, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            providers_tried.append(provider_name)
            try:
                yield from provider.stream(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_successful_idx = i
                self._store_usage(getattr(provider, 'last_usage', {}))
                return
            except Exception as e:
                errors[provider_name] = str(e)
                if self.logger:
                    self.logger.warning(f"Provider {provider_name} stream failed: {e}")
                continue

        raise FailoverExhaustedError(
            f"All {len(self.providers)} providers failed",
            providers_tried=providers_tried,
            errors=errors,
        )


def create_client(
    provider: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    deterministic: bool = True,
    base_url: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client.

    Args:
        provider: "claude", "openai", "gemini", "grok", "local", "ollama", or "auto" (detect from env)
        model: Model name (uses provider default if not specified)
        api_key: API key (uses env var if not specified)
        deterministic: Use temperature=0 for reproducibility
        base_url: Base URL for local Ollama server (default: http://localhost:11434)

    Returns:
        Configured LLM client

    Auto-detection order:
    1. ANTHROPIC_API_KEY -> Claude
    2. OPENAI_API_KEY -> OpenAI
    3. XAI_API_KEY -> Grok
    4. GOOGLE_API_KEY or GEMINI_API_KEY -> Gemini
    5. Ollama running locally -> OllamaClient
    """
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY") or api_key:
            provider = "claude"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("XAI_API_KEY"):
            provider = "grok"
        elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif OllamaClient.is_available(
            base_url or os.environ.get("LOCAL_MODEL_URL", "http://localhost:11434")
        ):
            provider = "local"
        else:
            raise ValueError(
                "No API key found and no local Ollama server detected. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GOOGLE_API_KEY, "
                "or start Ollama (ollama serve)"
            )

    provider = provider.lower()

    if provider in ("claude", "anthropic"):
        return ClaudeClient(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            deterministic=deterministic
        )
    elif provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model or "gpt-5.1",
            deterministic=deterministic
        )
    elif provider in ("grok", "xai"):
        return GrokClient(
            api_key=api_key,
            model=model or "grok-4-1-fast-reasoning",
            deterministic=deterministic
        )
    elif provider in ("gemini", "google"):
        return GeminiClient(
            api_key=api_key,
            model=model or "gemini-2.0-flash",
            deterministic=deterministic
        )
    elif provider in ("local", "ollama"):
        return OllamaClient(
            base_url=base_url or os.environ.get("LOCAL_MODEL_URL", "http://localhost:11434"),
            model=model or "llama3:8b",
            deterministic=deterministic
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'openai', 'grok', 'gemini', or 'local'")
