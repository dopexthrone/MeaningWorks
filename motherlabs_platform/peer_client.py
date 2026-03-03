"""
Peer client — async HTTP client for communicating with remote Mother instances.

Speaks to the remote instance's existing V2 API via httpx.
"""

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerStatus:
    """Health status of a peer instance."""
    instance_id: str
    reachable: bool
    latency_ms: float
    last_checked: str
    tool_count: int = 0
    error: str = ""


class PeerClient:
    """Async HTTP client for a single remote Mother instance.

    Uses httpx to communicate with the remote V2 API.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_client(self):
        """Lazy-load httpx.AsyncClient."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def health_check(self, instance_id: str = "") -> PeerStatus:
        """Check peer health via GET /v1/health.

        Returns PeerStatus with reachable=True/False and latency.
        """
        start = time.monotonic()
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            client = self._get_client()
            resp = await client.get("/v1/health")
            latency = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                return PeerStatus(
                    instance_id=instance_id,
                    reachable=True,
                    latency_ms=latency,
                    last_checked=now,
                )
            return PeerStatus(
                instance_id=instance_id,
                reachable=False,
                latency_ms=latency,
                last_checked=now,
                error=f"HTTP {resp.status_code}",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return PeerStatus(
                instance_id=instance_id,
                reachable=False,
                latency_ms=latency,
                last_checked=now,
                error=str(e),
            )

    async def get_digest(self) -> Dict[str, Any]:
        """Fetch trust graph digest from GET /v2/instance/digest."""
        client = self._get_client()
        resp = await client.get("/v2/instance/digest")
        resp.raise_for_status()
        return resp.json()

    async def list_tools(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tools from GET /v2/tools/export."""
        client = self._get_client()
        params = {}
        if domain:
            params["domain"] = domain
        resp = await client.get("/v2/tools/export", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("tools", [])

    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search remote tools via GET /v2/tools/search."""
        client = self._get_client()
        resp = await client.get("/v2/tools/search", params={"q": query})
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    async def get_tool(self, package_id: str) -> Dict[str, Any]:
        """Fetch full tool package from GET /v2/tools/{id}."""
        client = self._get_client()
        resp = await client.get(f"/v2/tools/{package_id}")
        resp.raise_for_status()
        return resp.json()

    async def register_self(
        self,
        instance_id: str,
        name: str,
        api_endpoint: str,
    ) -> bool:
        """Register this instance with the remote peer via POST /v2/instance/peers.

        Returns True on success, False on failure.
        """
        client = self._get_client()
        try:
            resp = await client.post(
                "/v2/instance/peers",
                json={
                    "instance_id": instance_id,
                    "name": name,
                    "api_endpoint": api_endpoint,
                },
            )
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.warning(f"Failed to register self with peer: {e}")
            return False

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
