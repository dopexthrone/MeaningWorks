"""Async HTTP client for the Motherlabs API.

Bots call the API over HTTP — never import the engine directly.
Flow: POST /v2/compile/async → poll GET /v2/tasks/{id} → return result.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("motherlabs.bots.api_client")

DEFAULT_API_URL = "http://api:8000"
POLL_INTERVAL = 3.0  # seconds between status polls
MAX_POLL_TIME = 180.0  # max seconds to wait for compilation


@dataclass(frozen=True)
class CompileBotResult:
    """Compilation result as seen by bots."""

    success: bool
    blueprint: Dict[str, Any] = field(default_factory=dict)
    trust: Dict[str, Any] = field(default_factory=dict)
    domain: str = "software"
    duration_seconds: float = 0.0
    error: Optional[str] = None


class MotherlabsAPIClient:
    """Async client that submits compilations and polls for results."""

    def __init__(self, api_url: Optional[str] = None, timeout: float = 10.0):
        self.api_url = (api_url or DEFAULT_API_URL).rstrip("/")
        self._timeout = timeout

    async def compile(
        self,
        description: str,
        domain: str = "software",
    ) -> CompileBotResult:
        """Submit a compilation and wait for the result."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Enqueue
            resp = await client.post(
                f"{self.api_url}/v2/compile/async",
                json={"description": description, "domain": domain},
            )
            resp.raise_for_status()
            data = resp.json()
            task_id = data["task_id"]

            # Poll
            elapsed = 0.0
            while elapsed < MAX_POLL_TIME:
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

                status_resp = await client.get(
                    f"{self.api_url}/v2/tasks/{task_id}",
                )
                status_resp.raise_for_status()
                status = status_resp.json()

                if status["status"] == "complete":
                    result = status.get("result", {})
                    return CompileBotResult(
                        success=result.get("success", False),
                        blueprint=result.get("blueprint", {}),
                        trust=result.get("trust", {}),
                        domain=result.get("domain", domain),
                        duration_seconds=result.get("duration_seconds", 0.0),
                    )

                if status["status"] == "error":
                    return CompileBotResult(
                        success=False,
                        error=status.get("error", "Compilation failed"),
                    )

            return CompileBotResult(
                success=False,
                error=f"Compilation timed out after {MAX_POLL_TIME:.0f}s",
            )

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self.api_url}/v2/health")
            resp.raise_for_status()
            return resp.json()
