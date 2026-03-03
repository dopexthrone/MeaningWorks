"""Base messaging bridge — TCP client and abstract bridge protocol.

Handles the connection to a running Motherlabs agent system via TCP JSON lines.
Platform-specific bridges (Telegram, Discord) subclass MessageBridge.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TCPClient:
    """Async TCP client that speaks JSON-line protocol to a Motherlabs runtime."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Establish TCP connection to the runtime."""
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port,
        )
        logger.info("Connected to runtime at %s:%s", self.host, self.port)

    async def disconnect(self) -> None:
        """Close the TCP connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
            logger.info("Disconnected from runtime")

    async def send(self, target: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to the runtime and return the response.

        Args:
            target: Component name or "_system" for meta-commands
            payload: Message payload dict

        Returns:
            Response dict from the runtime
        """
        if not self._writer:
            raise ConnectionError("Not connected to runtime")

        msg = json.dumps({"target": target, "payload": payload}) + "\n"
        self._writer.write(msg.encode())
        await self._writer.drain()

        data = await self._reader.readline()
        if not data:
            raise ConnectionError("Runtime closed connection")

        return json.loads(data.decode().strip())

    @property
    def connected(self) -> bool:
        return self._writer is not None


class MessageBridge(ABC):
    """Abstract base for platform-specific messaging bridges.

    Subclasses implement platform connection/polling and message translation.
    The bridge handles routing between platform messages and runtime targets.
    """

    def __init__(
        self,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 8080,
        default_target: str = "Chat Agent",
    ):
        self.tcp = TCPClient(tcp_host, tcp_port)
        self.default_target = default_target

    def translate_to_runtime(self, text: str) -> Dict[str, Any]:
        """Translate a platform message to a runtime message.

        Special commands:
            /learn X  → {"target": "_system", "payload": {"action": "learn", "skill": "X"}}
            /status   → {"target": "_system", "payload": {"action": "status"}}
            /health   → {"target": "_system", "payload": {"action": "health"}}
            other     → {"target": default_target, "payload": {"action": "chat", "message": "..."}}

        Returns:
            Dict with "target" and "payload" keys
        """
        text = text.strip()

        if text.startswith("/learn "):
            skill = text[7:].strip()
            return {
                "target": "_system",
                "payload": {"action": "learn", "skill": skill},
            }
        elif text == "/status":
            return {
                "target": "_system",
                "payload": {"action": "status"},
            }
        elif text == "/health":
            return {
                "target": "_system",
                "payload": {"action": "health"},
            }
        elif text.startswith("/compile:"):
            # /compile:domain description
            rest = text[9:]
            parts = rest.split(None, 1)
            domain = parts[0] if parts else "software"
            desc = parts[1].strip() if len(parts) > 1 else ""
            return {
                "target": "_system",
                "payload": {"action": "compile", "skill": desc, "domain": domain},
            }
        elif text.startswith("/compile "):
            skill = text[9:].strip()
            return {
                "target": "_system",
                "payload": {"action": "compile", "skill": skill},
            }
        elif text.startswith("/tools search "):
            query = text[14:].strip()
            return {
                "target": "_system",
                "payload": {"action": "tools", "subaction": "search", "query": query},
            }
        elif text.startswith("/tools export "):
            compilation_id = text[14:].strip()
            return {
                "target": "_system",
                "payload": {"action": "tools", "subaction": "export", "compilation_id": compilation_id},
            }
        elif text.startswith("/tools import "):
            file_path = text[14:].strip()
            return {
                "target": "_system",
                "payload": {"action": "tools", "subaction": "import", "file_path": file_path},
            }
        elif text == "/tools":
            return {
                "target": "_system",
                "payload": {"action": "tools", "subaction": "list"},
            }
        elif text == "/instance":
            return {
                "target": "_system",
                "payload": {"action": "instance"},
            }
        else:
            return {
                "target": self.default_target,
                "payload": {"action": "chat", "message": text},
            }

    def translate_from_runtime(self, response: Dict[str, Any]) -> str:
        """Translate a runtime response to a human-readable string."""
        if "error" in response:
            return f"Error: {response['error']}"
        if "content" in response:
            return str(response["content"])
        if "response" in response:
            return str(response["response"])
        # Compilation result
        if "compilation_id" in response and "status" in response:
            status = response["status"]
            if status == "compiled":
                lines = [
                    f"Compiled successfully!",
                    f"  ID: {response['compilation_id']}",
                    f"  Trust: {response.get('trust_score', 0):.1f}",
                    f"  Badge: {response.get('verification_badge', 'none')}",
                    f"  Components: {response.get('component_count', 0)}",
                ]
                return "\n".join(lines)
            elif status == "exported":
                return f"Exported to: {response.get('file_path', '?')}"
            elif status == "imported":
                return f"Imported successfully from: {response.get('file_path', '?')}"
            elif status == "rejected":
                return f"Import rejected: {response.get('reason', 'trust threshold not met')}"
            elif status == "failed":
                return f"Failed: {response.get('error', 'unknown error')}"
        # Tool list
        if "tools" in response:
            tools = response["tools"]
            if not tools:
                return "No tools found."
            lines = ["Tools:"]
            for t in tools:
                if "error" in t:
                    lines.append(f"  Error: {t['error']}")
                else:
                    name = t.get("name", t.get("id", "?"))
                    domain = t.get("domain", "")
                    trust = t.get("trust_score", 0)
                    lines.append(f"  - {name} ({domain}) trust={trust:.1f}")
            return "\n".join(lines)
        # Instance info
        if "instance_id" in response and "tool_count" in response:
            lines = [
                f"Instance: {response['instance_id']}",
                f"  Tools: {response['tool_count']}",
                f"  Corpus: {response.get('corpus_path', '?')}",
                f"  Status: {response.get('status', '?')}",
            ]
            return "\n".join(lines)
        if "components" in response:
            # Status response
            comps = response["components"]
            if isinstance(comps, list):
                return "Registered components:\n" + "\n".join(f"  - {c}" for c in comps)
            return str(comps)
        if "uptime" in response:
            return f"Uptime: {response['uptime']:.0f}s, Messages: {response.get('messages', 0)}"
        return json.dumps(response, indent=2)

    async def handle_message(self, text: str) -> str:
        """Process an incoming platform message end-to-end.

        1. Translate to runtime format
        2. Send via TCP
        3. Translate response back

        Returns:
            Human-readable response string
        """
        msg = self.translate_to_runtime(text)
        try:
            response = await self.tcp.send(msg["target"], msg["payload"])
        except (ConnectionError, OSError) as e:
            return f"Cannot reach agent: {e}"
        return self.translate_from_runtime(response)

    @abstractmethod
    async def start(self) -> None:
        """Start the bridge — connect to platform and begin polling."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the bridge gracefully."""
        ...
