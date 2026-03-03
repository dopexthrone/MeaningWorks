"""
Local LLM server — OpenAI-compatible chat completions endpoint for ElevenLabs.

Lightweight aiohttp server that accepts POST /v1/chat/completions from
ElevenLabs Conversational AI cloud, injects Mother's system prompt, and
streams tokens from the configured LLM provider.

LEAF module. No imports from core/. Handlers are injected callables.
Localhost only — never externally accessible.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger("mother.llm_server")

try:
    from aiohttp import web
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False


def is_llm_server_available() -> bool:
    """True if aiohttp is installed."""
    return _AIOHTTP_AVAILABLE


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the local LLM server."""
    host: str = "127.0.0.1"
    port: int = 11411


class LocalLLMServer:
    """OpenAI-compatible chat completions proxy for ElevenLabs duplex voice.

    Accepts POST /v1/chat/completions, injects Mother's current system prompt,
    routes to the configured LLM provider, and streams SSE tokens back.

    Usage:
        server = LocalLLMServer()
        server.set_handlers(system_prompt_fn, chat_stream_fn)
        await server.start()
        # ... ElevenLabs calls http://127.0.0.1:11411/v1/chat/completions
        await server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11411):
        self._host = host
        self._port = port
        self._system_prompt_fn: Optional[Callable[[], str]] = None
        self._chat_stream_fn: Optional[
            Callable[[List[Dict], str], AsyncGenerator[str, None]]
        ] = None
        self._app: Optional[object] = None
        self._runner: Optional[object] = None
        self._site: Optional[object] = None
        self._running = False
        self._request_count = 0

    @property
    def running(self) -> bool:
        """True if server is accepting requests."""
        return self._running

    @property
    def url(self) -> str:
        """Base URL for the server."""
        return f"http://{self._host}:{self._port}"

    @property
    def completions_url(self) -> str:
        """Full URL for the chat completions endpoint."""
        return f"{self.url}/v1/chat/completions"

    @property
    def request_count(self) -> int:
        """Number of requests handled since start."""
        return self._request_count

    def set_handlers(
        self,
        system_prompt_fn: Callable[[], str],
        chat_stream_fn: Callable[[List[Dict], str], AsyncGenerator[str, None]],
    ) -> None:
        """Inject handler callables. Must be called before start().

        Args:
            system_prompt_fn: Returns Mother's current system prompt string.
            chat_stream_fn: Async generator yielding LLM tokens.
                Takes (messages, system_prompt) -> AsyncGenerator[str, None].
        """
        self._system_prompt_fn = system_prompt_fn
        self._chat_stream_fn = chat_stream_fn

    async def start(self) -> None:
        """Start the server. Raises if handlers not set or aiohttp missing."""
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for LocalLLMServer")
        if self._system_prompt_fn is None or self._chat_stream_fn is None:
            raise RuntimeError("Call set_handlers() before start()")
        if self._running:
            logger.warning("Server already running")
            return

        self._app = web.Application()
        self._app.router.add_post(
            "/v1/chat/completions", self._handle_chat_completions
        )
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        try:
            self._site = web.TCPSite(
                self._runner, self._host, self._port, reuse_address=True
            )
            await self._site.start()
            self._running = True
            self._request_count = 0
            logger.info(f"LLM server started at {self.completions_url}")
        except OSError as e:
            await self._runner.cleanup()
            self._runner = None
            self._app = None
            raise RuntimeError(f"Port {self._port} unavailable: {e}") from e

    async def stop(self) -> None:
        """Stop the server and clean up."""
        if not self._running:
            return

        self._running = False
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("LLM server stopped")

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "requests_handled": self._request_count,
        })

    async def _handle_chat_completions(
        self, request: "web.Request"
    ) -> "web.StreamResponse":
        """Handle POST /v1/chat/completions from ElevenLabs.

        Parses messages, injects system prompt, streams SSE tokens back.
        Supports both streaming and non-streaming modes.
        """
        self._request_count += 1
        request_id = f"chatcmpl-mother-{self._request_count}"

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Malformed request body: {e}")
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            return web.json_response(
                {"error": {"message": "messages field is required", "type": "invalid_request_error"}},
                status=400,
            )

        # Inject Mother's current system prompt
        system_prompt = ""
        if self._system_prompt_fn:
            try:
                system_prompt = self._system_prompt_fn()
            except Exception as e:
                logger.warning(f"System prompt fn error: {e}")

        if stream:
            return await self._stream_response(messages, system_prompt, request_id, request)
        else:
            return await self._non_stream_response(messages, system_prompt, request_id)

    async def _stream_response(
        self,
        messages: List[Dict],
        system_prompt: str,
        request_id: str,
        request: "web.Request",
    ) -> "web.StreamResponse":
        """Stream SSE response in OpenAI chat completions format."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": request_id,
            },
        )
        await response.prepare(request)

        created = int(time.time())
        collected = []

        try:
            async for token in self._chat_stream_fn(messages, system_prompt):
                collected.append(token)
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "mother",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                data = f"data: {json.dumps(chunk)}\n\n"
                await response.write(data.encode("utf-8"))

            # Send finish chunk
            finish_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mother",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            await response.write(
                f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")

        except Exception as e:
            logger.warning(f"Stream error: {e}")
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mother",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f" [Error: {e}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
            await response.write(
                f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")

        return response

    async def _non_stream_response(
        self,
        messages: List[Dict],
        system_prompt: str,
        request_id: str,
    ) -> "web.Response":
        """Non-streaming response — collect all tokens then return."""
        collected = []
        try:
            async for token in self._chat_stream_fn(messages, system_prompt):
                collected.append(token)
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            return web.json_response(
                {"error": {"message": str(e), "type": "server_error"}},
                status=500,
            )

        content = "".join(collected)
        return web.json_response({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mother",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })
